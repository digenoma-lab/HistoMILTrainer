"""
Utils functions
"""

import os
import numpy as np
import torch
import random
import mlflow
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_epoch = -1 

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.best_epoch = epoch  
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if epoch >= self.stop_epoch and self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0
            self.best_epoch = epoch 

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def use_mlflow(args, params, metrics):
    """Use mlflow to save results and compare"""
    home_path = os.path.expanduser(args.mlflow_dir)
    print(f"Using mlflow to save results: {args.mlflow_dir}")
    mlflow.set_tracking_uri(f"file:{home_path}")
    mlflow.set_experiment(args.mlflow_exp)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("train AUC", metrics["train_auc"].values[0])
        mlflow.log_metric("train ACC", metrics["train_acc"].values[0])
        mlflow.log_metric("val AUC", metrics["val_auc"].values[0])
        mlflow.log_metric("val ACC", metrics["val_acc"].values[0])
        mlflow.log_metric("test AUC", metrics["test_auc"].values[0])
        mlflow.log_metric("test ACC", metrics["test_acc"].values[0])
        model_path = f"{args.results_dir}/checkpoint.pt"
        mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Saved to {args.mlflow_dir}, {args.mlflow_exp}")

def get_embed_dim(patch_encoder):
    embed_dims = {
        "uni": 1024,
        "uni_v2": 1536,
        "conch_v1": 512,
        "conch_v15": 768,
        "virchow": 2560,
        "virchow2": 2560,
        "phikon": 768,
        "phikon_v2": 1024,
        "ctranspath": 768,
        "resnet50": 1024,
    }

    if patch_encoder not in embed_dims:
        raise ValueError(
            f"Patch encoder '{patch_encoder}' no reconocido. Opciones válidas:\n{list(embed_dims.keys())}"
        )

    return embed_dims[patch_encoder]

def get_weights(serie):
    labels = serie.index
    counts = serie.values
    n_elements = counts.sum()
    weights = n_elements / (len(labels.unique()) * counts)
    return weights