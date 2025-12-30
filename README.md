# HistoMIL

A library for training Multi-Instance Learning (MIL) architectures from [MIL-Lab](https://github.com/AMLab-Amsterdam/MIL) on histology datasets. HistoMIL provides a unified interface to train and evaluate various state-of-the-art MIL models for whole slide image (WSI) analysis.

## Overview

HistoMIL offers a streamlined framework to train MIL architectures on histology data. It supports multiple architectures from MIL-Lab, including:

- **ABMIL** - Attention-based Multiple Instance Learning
- **CLAM** - Clustering-constrained Attention Multiple instance learning
- **DSMIL** - Dual-stream Multiple Instance Learning
- **DFTD** - Deep Feature-based Top-Down attention
- **ILRA** - Instance-Level Representation Aggregation
- **RRT** - Residual Regression Transformer
- **Transformer** - Transformer-based MIL
- **TransMIL** - Transductive Multiple Instance Learning
- **WIKG** - Weighted Instance Knowledge Graph

## Features

- **Unified Training Interface**: Train any supported MIL architecture with consistent parameters
- **Flexible Data Loading**: Support for variable number of patches per slide
- **Cross-Validation**: Built-in support for k-fold cross-validation
- **Feature Extraction Integration**: Works seamlessly with pre-extracted patch features (e.g., from TRIDENT)
- **Class Weighting**: Automatic class weight calculation for imbalanced datasets
- **Early Stopping**: Prevent overfitting with configurable early stopping
- **MLflow Integration**: Optional experiment tracking with MLflow

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd HistoMIL

# Install dependencies
pip install -r requirements.txt
```

**Note**: HistoMIL requires MIL-Lab to be installed and accessible. Make sure MIL-Lab is properly configured in your environment.

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:
```
features/
  ├── slide1.h5
  ├── slide2.h5
  └── ...
```

Each H5 file should contain:
- `features`: Array of shape `(num_patches, feature_dim)`
- Optionally: `coords`: Array of patch coordinates

### 2. Create Dataset CSV

Create a CSV file with columns:
- `slide_id`: Unique identifier for each slide
- `label`: Target label for classification

### 3. Generate Splits

```bash
python make_splits.py \
  --csv_path ./data/dataset.csv \
  --splits_dir ./splits/ \
  --output_name my_task \
  --folds 10 \
  --test_frac 0.2
```

### 4. Train a Model

```bash
python main.py \
  --fold 0 \
  --features_path ./features/ \
  --splits_dir ./splits/my_task/ \
  --csv_path ./data/dataset.csv \
  --model abmil \
  --pretrained_model uni_v2 \
  --results_dir ./results/abmil/ \
  --epochs 20 \
  --learning_rate 4e-4
```

### 5. Run Multiple Folds (SLURM)

Use the provided shell scripts for running multiple folds:

```bash
sbatch run_abmil.sh
sbatch run_clam.sh
# ... etc
```

## Usage

### Command Line Arguments

```bash
python main.py \
  --fold <fold_number>              # Cross-validation fold (0-9)
  --features_path <path>            # Path to H5 feature files
  --splits_dir <path>               # Directory containing split files
  --csv_path <path>                 # Path to dataset CSV
  --model <model_name>              # Model: abmil, clam, dsmil, dftd, etc.
  --pretrained_model <name>         # Feature extractor: uni_v2, etc.
  --results_dir <path>              # Output directory for results
  --epochs <int>                    # Number of training epochs (default: 3)
  --learning_rate <float>           # Learning rate (default: 4e-4)
  --use_class_weights <bool>       # Use class weights (default: True)
  --mlflow_dir <path>               # MLflow tracking directory (optional)
  --mlflow_exp <name>               # MLflow experiment name (optional)
```

### Supported Models

- `abmil` - Attention-based MIL
- `clam` - CLAM architecture
- `dsmil` - Dual-stream MIL
- `dftd` - Deep Feature Top-Down
- `ilra` - Instance-Level Representation Aggregation
- `rrt` - Residual Regression Transformer
- `transformer` - Transformer-based MIL
- `transmil` - Transductive MIL
- `wikg` - Weighted Instance Knowledge Graph

### Python API

```python
from histomil import import_model, H5Dataset, train, test
from histomil.utils import get_embed_dim, seed_torch
import torch

# Set up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(2)
embed_dim = get_embed_dim("uni_v2")

# Load model
model = import_model("abmil", "uni_v2").to(device)

# Create dataset
dataset = H5Dataset(
    features_path="./features/",
    df=dataset_df,
    split="train",
    variable_patches=True
)

# Train and evaluate
trained_model, metrics = train(model, train_loader, val_loader, ...)
test_metrics = test(trained_model, test_loader, ...)
```

## Project Structure

```
HistoMIL/
├── histomil/           # Main library package
│   ├── models.py       # Model import functions
│   ├── datasets.py     # Dataset classes
│   ├── train.py        # Training and evaluation functions
│   ├── splits.py       # Split management
│   └── utils.py        # Utility functions
├── main.py             # Main training script
├── make_splits.py      # Split generation script
├── run_*.sh            # SLURM job scripts for different models
└── splits/             # Generated split files
```

## Output

Training produces:
- `{fold}-checkpoint.pt`: Best model checkpoint
- `{fold}.csv`: Training and test metrics (AUC, accuracy, F1, etc.)

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- tqdm
- h5py
- MIL-Lab (for model architectures)

## Citation

If you use HistoMIL in your research, please cite the original MIL-Lab paper and the specific architecture papers you use.

## License

See LICENSE file for details.
