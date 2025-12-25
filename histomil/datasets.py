import os
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
SEED = 2
device = torch.device("cuda")

def variable_patches_collate_fn(batch):
    """
    Custom collate function to handle batches with variable number of patches per slide.
    Returns a list of tuples (features, label) where each slide can have different number of patches.
    """
    # Simply return the batch as a list (no padding/stacking)
    # Each element is (features, label) where features shape is (num_patches, feature_dim)
    return batch

class H5Dataset(Dataset):
    def __init__(self, feats_path, df,
            split = None,
            num_features = 1536,  # Kept for backward compatibility but not used for variable patches
            label_col = "label",
            variable_patches = True):
        if split != None:
            self.df = df[df[split]]
        else:
            self.df = df
        self.split = split
        self.feats_path = feats_path
        self.num_features = num_features
        self.label_col = label_col
        self.variable_patches = variable_patches

    @staticmethod
    def drop_extention(filepath):
        filename = Path(filepath)
        return filename.stem

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with h5py.File(os.path.join(self.feats_path, self.drop_extention(row['slide_id']) + '.h5'), "r") as f:
            features = torch.from_numpy(f["features"][:])

        if self.variable_patches:
            # Return all patches without sampling - allows variable number of patches per slide
            # features shape: (num_patches, feature_dim)
            pass  # Already have all features, no sampling needed
        else:
            # Legacy behavior: sample to fixed size
            num_available = features.shape[0] #Number of instances
            if num_available >= self.num_features:
                if self.split == 'train':
                    # Random sampling for training
                    indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
                else:
                    # Deterministic sampling for validation/test (take first num_features)
                    indices = torch.arange(self.num_features, dtype=torch.long)
            else:
                # Oversampling if not enough instances
                if self.split == 'train':
                    indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED), dtype=torch.long)
                else:
                    # Deterministic oversampling for validation/test
                    indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED + idx), dtype=torch.long)
            features = features[indices]

        label = torch.tensor(row[self.label_col], dtype=torch.float32)
        return features, label