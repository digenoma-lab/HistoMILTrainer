"""Fixed-parameter training for MIL models."""

import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from histomil.datasets import H5Dataset, variable_patches_collate_fn
from histomil.models import import_model
from histomil.train import test, train
from histomil.utils import get_weights, seed_torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FixedParameterTrainer:
    """Train one fixed MIL configuration across the requested folds."""

    SEED = 2
    BATCH_SIZE = 16
    TRANSFER_MODES = {"scratch", "head_only", "partial"}

    def __init__(
        self,
        folds,
        features_path,
        splits_dir,
        csv_path,
        results_dir,
        params_path,
        feature_extractor="uni_v2",
        epochs=10,
        learning_rate=4e-4,
        mil="abmil",
        use_class_weights=True,
        transfer_mode="scratch",
        pretrained_checkpoint=None,
    ):
        self.folds = folds
        self.features_path = os.path.realpath(features_path)
        self.splits_dir = os.path.realpath(splits_dir)
        self.csv_path = os.path.realpath(csv_path)
        self.results_dir = os.path.realpath(results_dir)
        self.params_path = os.path.realpath(params_path)
        self.feature_extractor = feature_extractor
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.mil = mil
        self.use_class_weights = use_class_weights
        self.transfer_mode = transfer_mode
        self.pretrained_checkpoint = (
            os.path.realpath(pretrained_checkpoint)
            if pretrained_checkpoint not in (None, "")
            else None
        )
        self.batch_size = 1 if self.mil == "clam" else self.BATCH_SIZE
        self.logger = logging.getLogger(__name__)

        self._validate_inputs()
        self.params = self._load_fixed_params()

        seed_torch(self.SEED)
        os.makedirs(self.results_dir, exist_ok=True)

    def _validate_inputs(self):
        """Validate paths and transfer-learning arguments."""
        if not isinstance(self.folds, int) or self.folds < 1:
            raise ValueError(f"folds must be a positive integer. Received: {self.folds}")

        if self.transfer_mode not in self.TRANSFER_MODES:
            raise ValueError(
                f"transfer_mode must be one of {sorted(self.TRANSFER_MODES)}. "
                f"Received: {self.transfer_mode}"
            )

        required_paths = {
            "features_path": self.features_path,
            "splits_dir": self.splits_dir,
            "csv_path": self.csv_path,
            "params_path": self.params_path,
        }
        for name, path in required_paths.items():
            exists = os.path.isdir(path) if name in {"features_path", "splits_dir"} else os.path.isfile(path)
            if not exists:
                raise FileNotFoundError(f"{name} not found: {path}")

        if self.transfer_mode == "scratch":
            if self.pretrained_checkpoint is not None:
                raise ValueError("scratch must not receive pretrained_checkpoint")
        else:
            if self.pretrained_checkpoint is None:
                raise ValueError(
                    f"{self.transfer_mode} requires pretrained_checkpoint"
                )
            if not os.path.isfile(self.pretrained_checkpoint):
                raise FileNotFoundError(
                    f"pretrained_checkpoint not found: {self.pretrained_checkpoint}"
                )

        for fold in range(self.folds):
            for suffix in ("bool", "descriptor"):
                split_file = os.path.join(
                    self.splits_dir,
                    f"splits_{fold}_{suffix}.csv",
                )
                if not os.path.isfile(split_file):
                    raise FileNotFoundError(
                        f"Split file not found for fold {fold}: {split_file}"
                    )

    def _load_fixed_params(self):
        """Load one fixed model configuration from JSON."""
        with open(self.params_path, "r", encoding="utf-8") as file:
            params = json.load(file)

        if not isinstance(params, dict) or not params:
            raise ValueError(
                "params_path must contain one non-empty JSON object"
            )

        list_parameters = [
            key for key, value in params.items() if isinstance(value, list)
        ]
        if list_parameters:
            raise ValueError(
                "histomil-train accepts one fixed configuration. "
                f"List-valued parameters found: {list_parameters}"
            )

        return params

    def _load_fold_data(self, fold_idx):
        """Load the dataset split and class weights for one fold."""
        splits_file = os.path.join(
            self.splits_dir,
            f"splits_{fold_idx}_bool.csv",
        )
        descriptors_file = os.path.join(
            self.splits_dir,
            f"splits_{fold_idx}_descriptor.csv",
        )

        splits = pd.read_csv(splits_file)
        splits.columns = ["slide_id", "train", "val", "test"]

        descriptors = pd.read_csv(descriptors_file, index_col=0)

        class_weights = None
        if self.use_class_weights:
            class_weights = get_weights(descriptors.train)

        dataset_csv = pd.read_csv(self.csv_path).merge(
            splits,
            on="slide_id",
        )
        return dataset_csv, class_weights

    def _create_loader(self, dataset_csv, split, shuffle):
        """Create a DataLoader for one dataset split."""
        return DataLoader(
            H5Dataset(
                self.features_path,
                dataset_csv,
                split,
                variable_patches=True,
            ),
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=variable_patches_collate_fn,
            worker_init_fn=lambda _: np.random.seed(self.SEED),
        )

    def _save_predictions(self, fold, y_scores, y_true):
        """Save test probabilities and labels using the existing output schema."""
        predictions_file = os.path.join(
            self.results_dir,
            f"predictions_{self.feature_extractor}.{self.mil}_{fold}.csv",
        )
        pd.DataFrame(
            {
                "y_pred": y_scores,
                "y_true": y_true,
            }
        ).to_csv(predictions_file, index=False)

    def run(self):
        """Train and evaluate one fixed parameter configuration across folds."""
        self.logger.info("=" * 60)
        self.logger.info("Starting fixed-parameter training")
        self.logger.info("=" * 60)
        self.logger.info("MIL: %s", self.mil)
        self.logger.info("Feature extractor: %s", self.feature_extractor)
        self.logger.info("Transfer mode: %s", self.transfer_mode)
        self.logger.info("Parameters: %s", self.params)
        self.logger.info("Folds: %d", self.folds)

        training_results = []
        test_results = []

        for fold in range(self.folds):
            self.logger.info("-" * 60)
            self.logger.info("Processing fold %d/%d", fold, self.folds - 1)

            dataset_csv, class_weights = self._load_fold_data(fold)
            train_loader = self._create_loader(dataset_csv, "train", shuffle=True)
            val_loader = self._create_loader(dataset_csv, "val", shuffle=False)
            test_loader = self._create_loader(dataset_csv, "test", shuffle=False)

            model = import_model(
                self.mil,
                self.feature_extractor,
                **self.params,
            ).to(device)

            model, train_metrics, checkpoint = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                results_dir=self.results_dir,
                learning_rate=self.learning_rate,
                fold=fold,
                epochs=self.epochs,
                class_weights=class_weights,
                model_name=self.mil,
                params=self.params,
                transfer_mode=self.transfer_mode,
                pretrained_checkpoint=self.pretrained_checkpoint,
            )

            test_metrics, y_scores, y_true = test(
                model,
                test_loader,
                class_weights=class_weights,
                model_name=self.mil,
            )

            train_row = dict(train_metrics)
            train_row["fold"] = fold
            train_row["feature_extractor"] = self.feature_extractor
            train_row["mil"] = self.mil
            train_row["transfer_mode"] = self.transfer_mode
            training_results.append(train_row)

            test_row = dict(test_metrics)
            test_row["fold"] = fold
            test_row["feature_extractor"] = self.feature_extractor
            test_row["mil"] = self.mil
            test_row["transfer_mode"] = self.transfer_mode
            test_results.append(test_row)

            self._save_predictions(fold, y_scores, y_true)

            best_model_file = os.path.join(
                self.results_dir,
                f"{fold}_best_model.pt",
            )
            shutil.copy2(checkpoint, best_model_file)

            self.logger.info(
                "Fold %d completed | val_auc=%.4f | test_auc=%.4f",
                fold,
                train_metrics.get("val_auc", float("nan")),
                test_metrics.get("test_auc", float("nan")),
            )

            del model, train_loader, val_loader, test_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        training_results_df = pd.DataFrame(training_results)
        test_results_df = pd.DataFrame(test_results)

        training_results_df.to_csv(
            os.path.join(
                self.results_dir,
                f"training_results_{self.feature_extractor}.{self.mil}.csv",
            ),
            index=False,
        )
        test_results_df.to_csv(
            os.path.join(
                self.results_dir,
                f"test_results_{self.feature_extractor}.{self.mil}.csv",
            ),
            index=False,
        )

        params_output = os.path.join(
            self.results_dir,
            f"best_params_{self.feature_extractor}.{self.mil}.json",
        )
        with open(params_output, "w", encoding="utf-8") as file:
            json.dump(self.params, file, indent=2)

        self.logger.info("=" * 60)
        self.logger.info("Fixed-parameter training completed")
        self.logger.info("=" * 60)

        return test_results_df