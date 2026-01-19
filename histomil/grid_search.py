"""Grid search for MIL models with cross-validation."""
import json
import os
from itertools import product

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from histomil import (
    H5Dataset,
    seed_torch,
    get_weights,
    import_model,
    variable_patches_collate_fn,
)
from histomil.train import train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GridSearch:
    """Grid search for MIL models with cross-validation."""
    
    SEED = 2
    BATCH_SIZE = 16
    
    def __init__(self, folds, features_path, splits_dir, csv_path, results_dir,
                 feature_extractor="uni_v2", epochs=10, learning_rate=4e-4,
                 mil="abmil", use_class_weights=True, grid_params_path="configs/abmil.json"):
        """
        Initialize GridSearch.
        
        Args:
            folds: Number of cross-validation folds
            features_path: Path to H5 feature files directory
            splits_dir: Directory containing split files
            csv_path: Path to dataset CSV
            results_dir: Output directory for results
            feature_extractor: Feature extractor name (default: "uni_v2")
            epochs: Number of training epochs (default: 10)
            learning_rate: Learning rate (default: 4e-4)
            mil: MIL architecture name (default: "abmil")
            use_class_weights: Whether to use class weights (default: True)
            grid_params_path: Path to JSON file with grid search parameters
        """
        self.folds = folds
        self.features_path = os.path.realpath(features_path)
        self.splits_dir = os.path.realpath(splits_dir)
        self.csv_path = os.path.realpath(csv_path)
        self.results_dir = os.path.realpath(results_dir)
        self.feature_extractor = feature_extractor
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.mil = mil
        self.use_class_weights = use_class_weights
        self.batch_size = 1 if mil == "clam" else self.BATCH_SIZE
        
        # Load grid parameters
        with open(grid_params_path, "r") as f:
            grid_params = json.load(f)
        
        self.param_combinations = [
            dict(zip(grid_params.keys(), combo))
            for combo in product(*grid_params.values())
        ]
        
        # Set seed
        seed_torch(self.SEED)
        
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _convert_value(self, v):
        """Convert float to int if whole number (pandas casts ints to floats)."""
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v
    
    def _load_fold_data(self, fold_idx):
        """Load splits, descriptors and class weights for a fold."""
        splits = pd.read_csv(f"{self.splits_dir}/splits_{fold_idx}_bool.csv")
        splits.columns = ["slide_id", "train", "val", "test"]
        descriptors = pd.read_csv(
            f"{self.splits_dir}/splits_{fold_idx}_descriptor.csv", index_col=0
        )
        class_weights = None
        if self.use_class_weights:
            class_weights = get_weights(descriptors.train)
        dataset_csv = pd.read_csv(self.csv_path).merge(splits, on="slide_id")
        return dataset_csv, class_weights
    
    def _create_loader(self, dataset_csv, split, shuffle):
        """Create DataLoader for a given split."""
        return DataLoader(
            H5Dataset(self.features_path, dataset_csv, split, variable_patches=True),
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=variable_patches_collate_fn,
            worker_init_fn=lambda _: np.random.seed(self.SEED),
        )
    
    def _get_best_params(self, grid_search_results, params_columns):
        """Find parameter combination with highest mean val_auc."""
        mean_grouped = (
            grid_search_results.groupby(params_columns)["val_auc"]
            .mean()
            .reset_index()
        )
        best_idx = mean_grouped["val_auc"].idxmax()
        return mean_grouped.loc[best_idx, params_columns].to_dict()
    
    def _filter_best_folds(self, grid_search_results, best_params_dict, params_columns):
        """Filter folds that use the best parameters."""
        mask = pd.Series([True] * len(grid_search_results))
        for col in params_columns:
            mask = mask & (grid_search_results[col] == best_params_dict[col])
        return grid_search_results[mask]
    
    def run(self):
        """Execute grid search."""
        print(f"Using: {self.feature_extractor} with {self.mil}")
        print(f"Generated {len(self.param_combinations)} parameter combinations")
        
        # Grid search: train all param/fold combinations
        grid_search_results = []
        for params in self.param_combinations:
            print(f"Params: {params}")
            for fold in range(self.folds):
                print(f"Fold: {fold}")
                dataset_csv, class_weights = self._load_fold_data(fold)
                train_loader = self._create_loader(dataset_csv, "train", shuffle=True)
                val_loader = self._create_loader(dataset_csv, "val", shuffle=True)
                
                mil = import_model(
                    self.mil, self.feature_extractor, **params
                ).to(device)
                _, train_metrics, checkpoint = train(
                    mil,
                    train_loader,
                    val_loader,
                    self.results_dir,
                    self.learning_rate,
                    fold,
                    self.epochs,
                    class_weights=class_weights,
                    model_name=self.mil,
                    params=params,
                )
                train_metrics.update({
                    "model_checkpoint": checkpoint,
                    "fold": fold,
                    "params": params,
                })
                grid_search_results.append(train_metrics)
        
        # Select best parameters by mean validation AUC
        grid_search_results = pd.json_normalize(grid_search_results)
        params_columns = [
            col for col in grid_search_results.columns if col.startswith("params.")
        ]
        best_params_dict = self._get_best_params(grid_search_results, params_columns)
        best_folds = self._filter_best_folds(
            grid_search_results, best_params_dict, params_columns
        )
        print(f"Best parameters: {best_params_dict}")
        print(f"Folds with best params: {len(best_folds)}")
        
        clean_params = {
            k.replace("params.", ""): self._convert_value(v)
            for k, v in best_params_dict.items()
        }
        
        # Test best model on each fold
        test_results = []
        for _, row in best_folds.iterrows():
            fold_idx = int(row["fold"])
            print(f"Testing fold: {fold_idx}")
            
            dataset_csv, class_weights = self._load_fold_data(fold_idx)
            test_loader = self._create_loader(dataset_csv, "test", shuffle=False)
            
            mil = import_model(
                self.mil, self.feature_extractor, **clean_params
            ).to(device)
            mil.load_state_dict(torch.load(row["model_checkpoint"]))
            
            test_metrics, y_preds, y_true = test(
                mil, test_loader, class_weights=class_weights, model_name=self.mil
            )
            test_metrics["fold"] = fold_idx
            test_results.append(test_metrics)
            print(f"Fold {fold_idx}: {test_metrics}")
            
            predictions = pd.DataFrame()
            predictions["y_pred"] = y_preds
            predictions["y_true"] = y_true
            
            predictions.to_csv(
                f"{self.results_dir}/predictions_{self.feature_extractor}.{self.mil}_{fold_idx}.csv",
                index=False
            )
        
        test_results_df = pd.DataFrame(test_results)
        test_results_df["feature_extractor"] = self.feature_extractor
        test_results_df["mil"] = self.mil
        print("Test results:\n", test_results_df)
        test_results_df.to_csv(
            f"{self.results_dir}/test_results_{self.feature_extractor}.{self.mil}.csv",
            index=False
        )
        
        json.dump(
            clean_params,
            open(f"{self.results_dir}/best_params_{self.feature_extractor}.{self.mil}.json", "w")
        )

