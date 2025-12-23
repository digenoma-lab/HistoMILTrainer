"""
Main file
"""
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from histomil import H5Dataset, seed_torch, get_embed_dim, get_weights, train, test, ABMIL

SEED = 2
BATCH_SIZE = 16
seed_torch(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRAB-MIL Training Script")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./temp_dir/")
    parser.add_argument("--pretrained_model", type=str, default = "uni_v2")
    parser.add_argument("--mlflow_dir", type=str, default="~/mlflow")
    parser.add_argument("--mlflow_exp", type=str, default=None)
    parser.add_argument("--epochs", type=int, default = 3)
    parser.add_argument("--learning_rate", type=float, default = 4e-4)
    parser.add_argument("--model", type=str, default="CLAM")
    parser.add_argument("--use_class_weights", type=bool, default=True)
    args = parser.parse_args()
    

    features_path = os.path.realpath(args.features_path)
    split_dir = os.path.realpath(args.splits_dir)
    csv_path = os.path.realpath(args.csv_path)
    results_dir = os.path.realpath(args.results_dir)
    embed_dim = get_embed_dim(args.pretrained_model)
    print("Using:", args.pretrained_model, "with embedding size:", embed_dim)
    os.makedirs(results_dir, exist_ok=True)

    dataset_csv = pd.read_csv(csv_path)
    splits = pd.read_csv(f"{split_dir}/splits_{args.fold}_bool.csv")
    splits.columns = ["slide_id", "train", "val", "test"]

    descriptors = pd.read_csv(f"{split_dir}/splits_{args.fold}_descriptor.csv", index_col=0)
    print(descriptors)
    if args.use_class_weights:
        class_weights = get_weights(descriptors.train)
        print("Using class_weights:", class_weights)
    else:
        class_weights = None
        
    dataset_csv = dataset_csv.merge(splits, on="slide_id")
    print(dataset_csv)
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
    print("Importing model")

    model = ABMIL(input_feature_dim = embed_dim).to(device)
    
    print(model)
    print("Training")
    trained_model, train_metrics = train(model, train_loader,
                                         val_loader, results_dir,
                                         args.learning_rate,
                                         class_weights = class_weights)
    print("Testing")
    test_metrics = test(trained_model, test_loader)
    print("Exporting metrics")
    train_metrics = pd.json_normalize(train_metrics)
    test_metrics = pd.json_normalize(test_metrics)
    metrics = pd.concat([train_metrics, test_metrics], axis = 1)

    results_path = f"{results_dir}/{args.fold}.csv"
    metrics.to_csv(results_path, index=False)
    params = {
        "dataset": os.path.basename(args.csv_path),
        "fold": args.fold,
        "pretrained_model": args.pretrained_model,
        "embed_dim": embed_dim
    }