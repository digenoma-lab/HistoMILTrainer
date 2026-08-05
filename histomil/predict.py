"""Prediction and Heatmap functions"""
import json
import os
from pathlib import Path
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import logging

from histomil import (
    H5DatasetPredict,
    import_model,
    variable_patches_collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unwrap_transformer(model):
    """Return the underlying Transformer MIL module (handles HF PreTrained wrapper)."""
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model
    return model


def _mha_cls_attention_row(mha: torch.nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
    """
    Softmax attention from the CLS token (index 0) to all tokens, averaged over heads.

    This is equivalent to the CLS row of ``need_weights=True`` / ``average_attn_weights=True``
    in ``nn.MultiheadAttention``, but uses O(L) memory instead of O(L^2).

    Args:
        mha: Multi-head attention module (``batch_first=False``).
        x: Normalized tokens of shape ``(L, B, E)``.

    Returns:
        Attention row of shape ``(B, L)``.
    """
    seq_len, batch_size, embed_dim = x.shape
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise ValueError(
            f"embed_dim={embed_dim} is not divisible by num_heads={num_heads}"
        )

    if mha.in_proj_weight is None:
        q = F.linear(
            x[:1],
            mha.q_proj_weight,
            None if mha.in_proj_bias is None else mha.in_proj_bias[:embed_dim],
        )
        k = F.linear(
            x,
            mha.k_proj_weight,
            None if mha.in_proj_bias is None else mha.in_proj_bias[embed_dim: 2 * embed_dim],
        )
    else:
        w_q, w_k, _ = mha.in_proj_weight.chunk(3, dim=0)
        if mha.in_proj_bias is None:
            b_q = b_k = None
        else:
            b_q, b_k, _ = mha.in_proj_bias.chunk(3, dim=0)
        q = F.linear(x[:1], w_q, b_q)  # (1, B, E)
        k = F.linear(x, w_k, b_k)      # (L, B, E)

    # (B, H, Lq/Lk, D)
    q = q.view(1, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
    k = k.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)

    attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)  # (B, H, 1, L)
    attn = attn.softmax(dim=-1)
    return attn.mean(dim=1).squeeze(1)  # (B, L)


def transformer_cls_attention(model, features: torch.Tensor) -> np.ndarray:
    """
    Extract CLS→patch attention for the first Transformer block without an N×N matrix.

    Matches the scores previously taken as ``attention[0, 1:]`` after
    ``return_attention=True``.
    """
    mil = _unwrap_transformer(model)
    block = mil.blocks[0]

    h = features
    if h.dim() == 2:
        h = h.unsqueeze(0)
    h = mil.patch_embed(h)
    batch_size = h.shape[0]
    cls_tokens = mil.cls_token.expand(batch_size, -1, -1).to(h.device)
    h = torch.cat((cls_tokens, h), dim=1)  # (B, N+1, E)

    # Mirror TransLayer preprocessing: (N+1, 1, E) with batch_first=False
    if h.shape[0] == 1:
        x = h.squeeze(0).unsqueeze(1)
    else:
        x = h.transpose(0, 1)
    norm_x = block.norm(x)
    attn_row = _mha_cls_attention_row(block.attention, norm_x)  # (B, N+1)
    # Same crop as before: drop CLS self-attention entry
    return attn_row[0, 1:].detach().cpu().numpy()


class Predictor:
    SEED = 2
    # Variable-length bags are processed one slide at a time; keep loader batch=1
    # so large WSIs are not held concurrently in memory.
    BATCH_SIZE = 1

    def __init__(self, csv_path, weights_path, features_folder, feature_extractor, results_dir, mil, params_path):
        self.logger = logging.getLogger(__name__)
        self.csv_path = csv_path
        self.weights_path = weights_path
        self.features_folder = features_folder
        self.feature_extractor = feature_extractor
        self.results_dir = results_dir
        self.batch_size = 1 if mil == "clam" else self.BATCH_SIZE
        self.mil = mil
        self.params_path = params_path
        
        self.logger.info(f"Initializing Predictor with parameters:")
        self.logger.info(f"  - CSV path: {csv_path}")
        self.logger.info(f"  - Weights path: {weights_path}")
        self.logger.info(f"  - Features folder: {features_folder}")
        self.logger.info(f"  - Feature extractor: {feature_extractor}")
        self.logger.info(f"  - Results directory: {results_dir}")
        self.logger.info(f"  - MIL model: {mil}")
        self.logger.info(f"  - Params path: {params_path}")
        self.logger.info(f"  - Batch size: {self.batch_size}")
        self.logger.info(f"  - Device: {device}")

    def _load_data(self, csv_path):
        """Load data from csv file."""
        self.logger.info(f"Loading dataset from CSV: {csv_path}")
        dataset_csv = pd.read_csv(csv_path)
        self.logger.info(f"Dataset loaded: {len(dataset_csv)} slides")
        return dataset_csv
    
    def _create_loader(self, dataset_csv):
        """Create DataLoader for a given split."""
        self.logger.debug(f"Creating DataLoader with batch_size={self.batch_size}")
        loader = DataLoader(
            H5DatasetPredict(self.features_folder, dataset_csv, variable_patches=True),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=variable_patches_collate_fn,
        )
        self.logger.debug(f"DataLoader created: {len(loader)} batches")
        return loader

    @staticmethod
    def drop_extension(filepath):
        filename = Path(filepath)
        return filename.stem

    def _forward_with_attention(self, model, features):
        """Run model forward and return (logits_dict, attn_scores ndarray)."""
        if self.mil == "transformer":
            # Logits without materializing N×N attention (uses SDPA / flash path).
            logits, _ = model(features, return_attention=False)
            attn_scores = transformer_cls_attention(model, features)
            return logits, attn_scores

        if self.mil in ["clam", "dftd"]:
            logits, attn = model(
                features,
                torch.tensor([1]).to(device),
                CrossEntropyLoss().to(device),
                return_attention=True,
            )
        else:
            logits, attn = model(features, return_attention=True)

        attn_scores = attn["attention"].squeeze().cpu().numpy()
        if self.mil == "wikg":
            # Geo attention score: average over the patch dimension
            attn_scores = attn_scores.mean(axis=0)
        if self.mil == "dsmil":
            # Only keep the attention score for the second class (positive)
            attn_scores = attn_scores[:, 1]
        if self.mil == "transmil":
            # Transmil returns a N^2 length array of attention scores.
            # We need to crop it to the number of patches.
            attn_scores = attn_scores[: features.shape[1]]
        return logits, attn_scores

    def predict(self, model, test_loader):
        """Predict function: Predicts the class of a set of slides."""
        self.logger.info("Starting prediction process")
        if self.mil == "transformer":
            self.logger.info(
                "Transformer: extracting CLS attention in O(N) memory "
                "(avoids full N×N attention matrices on large WSIs)"
            )
        model.eval()
        all_outputs = []
        all_attentions = []
        total_slides = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # Handle variable patches: batch is a list of feature tensors
                for features in batch:
                    features = features.to(device)  # Shape: (num_patches, feature_dim)
                    # Add batch dimension: (1, num_patches, feature_dim)
                    features = features.unsqueeze(0)
                    num_patches = features.shape[1]
                    self.logger.debug(f"Processing slide with {num_patches} patches")

                    logits, attn_scores = self._forward_with_attention(model, features)
                    probs = torch.softmax(logits["logits"], dim=1)
                    all_outputs.append(probs[0, 1].cpu().item())  # prob. clase 1
                    all_attentions.append(attn_scores)
                    total_slides += 1

                    # Free per-slide GPU tensors; fragmentation from large bags
                    # otherwise accumulates across the validation set.
                    del features, logits, probs
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        # Convert lists to numpy arrays (handles both scalars and arrays)
        # List contains scalars (variable patches mode)
        all_outputs = np.array(all_outputs)
        self.logger.info(f"✓ Prediction completed: {total_slides} slides processed")
        self.logger.debug(f"Output shape: {all_outputs.shape}, Attention scores: {len(all_attentions)}")
        return all_outputs, all_attentions

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("Starting prediction pipeline")
        self.logger.info("=" * 60)
        
        #Import model and load weights
        self.logger.info(f"Loading model parameters from: {self.params_path}")
        with open(self.params_path, "r") as f:
            params_dict = json.load(f)
        self.logger.debug(f"Model parameters: {params_dict}")
        
        self.logger.info(f"Importing {self.mil} model with {self.feature_extractor} feature extractor")
        model = import_model(self.mil, self.feature_extractor, **params_dict).to(device)
        
        self.logger.info(f"Loading model weights from: {self.weights_path}")
        model.load_state_dict(torch.load(self.weights_path))
        self.logger.info("✓ Model loaded successfully")
        
        dataset_df = self._load_data(self.csv_path)
        test_loader = self._create_loader(dataset_df)
        
        #Predict and extract attention scores
        y_prob, all_attentions = self.predict(model, test_loader)
        
        #Export predictions and attention scores
        self.logger.info("Exporting predictions and attention scores")
        results_df = pd.DataFrame({"slide_id": dataset_df["slide_id"], "prob": y_prob})
        results_df["pred"] = results_df["prob"].apply(lambda x: 1 if x > 0.5 else 0)
        
        predictions_file = os.path.join(self.results_dir, "predictions.csv")
        self.logger.info(f"Saving predictions to: {predictions_file}")
        results_df.to_csv(predictions_file, index=False)
        self.logger.info(f"✓ Predictions saved: {len(results_df)} slides")
        
        attention_dir = os.path.join(self.results_dir, "attention_scores")
        self.logger.info(f"Creating attention scores directory: {attention_dir}")
        os.makedirs(attention_dir, exist_ok=True)
        
        self.logger.info(f"Saving attention scores for {len(all_attentions)} slides")
        for slide_id, attn_score in zip(dataset_df["slide_id"], all_attentions):
            attn_file = os.path.join(attention_dir, self.drop_extension(slide_id) + ".h5")
            self.logger.debug(f"Saving attention scores for {slide_id} to {attn_file}")
            with h5py.File(attn_file, "w") as f:
                f.create_dataset("attention_scores", data=attn_score)
        
        self.logger.info("=" * 60)
        self.logger.info("✓ Prediction pipeline completed successfully")
        self.logger.info("=" * 60)
