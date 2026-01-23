"""Prediction and Heatmap functions"""
import json
import os
from pathlib import Path
import h5py
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from trident import OpenSlideWSI, visualize_heatmap

from histomil import (
    H5DatasetPredict,
    import_model,
    variable_patches_collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor:
    SEED = 2
    BATCH_SIZE = 16
    
    def __init__(self, csv_path, weights_path, features_folder, feature_extractor, results_dir, mil, params_path):
        self.csv_path = csv_path
        self.weights_path = weights_path
        self.features_folder = features_folder
        self.feature_extractor = feature_extractor
        self.results_dir = results_dir
        self.batch_size = 1 if mil == "clam" else self.BATCH_SIZE
        self.mil = mil
        self.params_path = params_path

    def _load_data(self, csv_path):
        """Load data from csv file."""
        dataset_csv = pd.read_csv(csv_path)
        return dataset_csv
    
    def _create_loader(self, dataset_csv):
        """Create DataLoader for a given split."""
        return DataLoader(
            H5DatasetPredict(self.features_folder, dataset_csv, variable_patches=True),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=variable_patches_collate_fn,
        )

    @staticmethod
    def drop_extension(filepath):
        filename = Path(filepath)
        return filename.stem

    def predict(self, model, test_loader):
        """Predict function: Predicts the class of a set of slides."""
        model.eval()
        all_outputs = []
        all_attentions = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # Handle variable patches: batch is a list of (features, label) tuples
                # Variable patches mode: process each slide individually
                for features in batch:
                    features = features.to(device)  # Shape: (num_patches, feature_dim)
                    # Add batch dimension: (1, num_patches, feature_dim)
                    features = features.unsqueeze(0)
                    if self.mil in ["clam", "dftd"]:
                        logits, attn = model(features, 
                                          torch.tensor([1]).to(device),
                                          CrossEntropyLoss().to(device),
                                          return_attention=True)
                    else:
                        logits, attn = model(features, return_attention=True)
                    probs = torch.softmax(logits["logits"], dim=1)
                    all_outputs.append(probs[0, 1].cpu().item())  # prob. clase 1
                    all_attentions.append(attn["attention"].squeeze().cpu().numpy())
        # Convert lists to numpy arrays (handles both scalars and arrays)
        # List contains scalars (variable patches mode)
        all_outputs = np.array(all_outputs)
        return all_outputs, all_attentions

    def run(self):
        #Import model and load weights
        with open(self.params_path, "r") as f:
            params_dict = json.load(f)
        model = import_model(self.mil, self.feature_extractor, **params_dict).to(device)
        model.load_state_dict(torch.load(self.weights_path))
        dataset_df = self._load_data(self.csv_path)
        test_loader = self._create_loader(dataset_df)
        #Predict and extract attention scores
        y_prob, all_attentions = self.predict(model, test_loader)
        #Export predictions and attention scores
        results_df = pd.DataFrame({"slide_id": dataset_df["slide_id"], "prob": y_prob})
        results_df["pred"] = results_df["prob"].apply(lambda x: 1 if x > 0.5 else 0)
        results_df.to_csv(self.results_dir + "predictions.csv", index=False)
        os.makedirs(os.path.join(self.results_dir, "attention_scores"), exist_ok=True)
        for slide_id, attn_score in zip(dataset_df["slide_id"], all_attentions):
            with h5py.File(os.path.join(self.results_dir, "attention_scores", self.drop_extension(slide_id) + ".h5"), "w") as f:
                f.create_dataset("attention_scores", data=attn_score)
