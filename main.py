"""
Main file
"""
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from histomil.datasets import H5Dataset
from histomil.utils import get_embed_dim, seed_torch

SEED = 2
BATCH_SIZE = 1
seed_torch(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIL Training Script")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./temp_dir/")
    parser.add_argument("--pretrained_model", type=str, default = "uni_v2")
    parser.add_argument("--mlflow_dir", type=str, default="~/mlflow")
    parser.add_argument("--mlflow_exp", type=str, default=None)
    parser.add_argument("--epochs", type=int, default = 20)
    parser.add_argument("--learning_rate", type=float, default = 4e-4)
    args = parser.parse_args()

    features_path = os.path.realpath(args.features_path)
    split_dir = os.path.realpath(args.split_dir)
    csv_path = os.path.realpath(args.csv_path)
    results_dir = os.path.realpath(args.results_dir)
    embed_dim = get_embed_dim(args.pretrained_model)
    print("Using:", args.pretrained_model, "with embedding size:", embed_dim)
    os.makedirs(results_dir, exist_ok=True)

    dataset_csv = pd.read_csv(csv_path)
    exit()
    splits = pd.read_csv(f"{split_dir}/splits_{args.fold}_bool.csv")
    splits.columns = ["slide_id", "train", "val", "test"]

    dataset_csv = dataset_csv.merge(splits, on="slide_id")
    print("Create datasets")
    train_loader = DataLoader(H5Dataset(features_path, dataset_csv, "train"),
                            batch_size=BATCH_SIZE, shuffle=True,
                            worker_init_fn=lambda _: np.random.seed(SEED))

    val_loader = DataLoader(H5Dataset(features_path, dataset_csv, "val"), 
                            batch_size=BATCH_SIZE, shuffle=True,
                            worker_init_fn=lambda _: np.random.seed(SEED))

    test_loader = DataLoader(H5Dataset(features_path, dataset_csv, "test"),
                            batch_size=BATCH_SIZE, shuffle=False,
                            worker_init_fn=lambda _: np.random.seed(SEED))
    print("Slides train:",len(train_loader))
    print("Slides val:", len(val_loader))
    print("Slides test:", len(test_loader))
