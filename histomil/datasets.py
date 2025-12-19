import os
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path

device = torch.device("cuda")
class H5Dataset(Dataset):
    def __init__(self, feats_path, df,
            split = None,
            num_features = 1536,
            label_col = "label"):
        self.df = df
        self.feats_path = feats_path
        self.num_features = num_features
        self.label_col = label_col

    @staticmethod
    def drop_extention(filepath):
        filename = Path(filepath)
        return filename.stem

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = H5Dataset.drop_extention(row["slide_id"])

        with h5py.File(os.path.join(self.feats_path, slide_id + '.h5'), "r") as f:
            features = torch.from_numpy(f["features"][:])
        label = torch.tensor(row[self.label_col], dtype=torch.float32)
        return features, label
