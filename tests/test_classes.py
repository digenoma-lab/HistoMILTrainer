"""Test that GridSearch and SplitManager classes can be instantiated."""
import os
import tempfile
import json
import pandas as pd

from histomil import GridSearch, SplitManager


def test_split_manager_instantiation():
    """Test that SplitManager can be instantiated with required parameters."""
    # Create a temporary CSV file for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_dataset.csv")
        
        # Create a minimal valid CSV
        data = pd.DataFrame({
            "case_id": ["case1", "case1", "case2", "case2"],
            "slide_id": ["slide1", "slide2", "slide3", "slide4"],
            "target": [0, 0, 1, 1]
        })
        data.to_csv(csv_path, index=False)
        
        # Test instantiation with required parameters
        split_manager = SplitManager(
            csv_path=csv_path,
            output_name="test_output"
        )
        
        assert split_manager.csv_path == csv_path
        assert split_manager.output_name == "test_output"
        assert split_manager.folds == 10  # default
        assert split_manager.splits_dir == "./splits"  # default
        assert split_manager.test_frac == 0.2  # default
        assert split_manager.target == "target"  # default


def test_split_manager_instantiation_with_all_params():
    """Test that SplitManager can be instantiated with all parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_dataset.csv")
        
        # Create a minimal valid CSV
        data = pd.DataFrame({
            "case_id": ["case1", "case1", "case2", "case2"],
            "slide_id": ["slide1", "slide2", "slide3", "slide4"],
            "label": [0, 0, 1, 1]  # Using 'label' instead of 'target'
        })
        data.to_csv(csv_path, index=False)
        
        # Test instantiation with all parameters
        split_manager = SplitManager(
            csv_path=csv_path,
            output_name="test_output",
            folds=5,
            splits_dir=tmpdir,
            test_frac=0.3,
            target="label"
        )
        
        assert split_manager.csv_path == csv_path
        assert split_manager.output_name == "test_output"
        assert split_manager.folds == 5
        assert split_manager.splits_dir == tmpdir
        assert split_manager.test_frac == 0.3
        assert split_manager.target == "label"


def test_grid_search_instantiation():
    """Test that GridSearch can be instantiated with required parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = os.path.join(tmpdir, "features")
        splits_dir = os.path.join(tmpdir, "splits")
        csv_path = os.path.join(tmpdir, "dataset.csv")
        results_dir = os.path.join(tmpdir, "results")
        grid_params_path = os.path.join(tmpdir, "grid_params.json")
        
        # Create directories
        os.makedirs(features_path, exist_ok=True)
        os.makedirs(splits_dir, exist_ok=True)
        
        # Create a minimal CSV
        data = pd.DataFrame({
            "case_id": ["case1", "case2"],
            "slide_id": ["slide1", "slide2"],
            "target": [0, 1]
        })
        data.to_csv(csv_path, index=False)
        
        # Create a minimal grid params JSON
        grid_params = {
            "n_classes": [2],
            "dropout": [0.25]
        }
        with open(grid_params_path, "w") as f:
            json.dump(grid_params, f)
        
        # Test instantiation with required parameters
        grid_search = GridSearch(
            folds=5,
            features_path=features_path,
            splits_dir=splits_dir,
            csv_path=csv_path,
            results_dir=results_dir,
            grid_params_path=grid_params_path
        )
        
        assert grid_search.folds == 5
        assert os.path.exists(grid_search.features_path)
        assert os.path.exists(grid_search.splits_dir)
        assert os.path.exists(grid_search.csv_path)
        assert os.path.exists(grid_search.results_dir)
        assert grid_search.feature_extractor == "uni_v2"  # default
        assert grid_search.epochs == 10  # default
        assert grid_search.learning_rate == 4e-4  # default
        assert grid_search.mil == "abmil"  # default
        assert grid_search.use_class_weights is True  # default
        assert len(grid_search.param_combinations) == 1  # 1 combination from grid_params


def test_grid_search_instantiation_with_all_params():
    """Test that GridSearch can be instantiated with all parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = os.path.join(tmpdir, "features")
        splits_dir = os.path.join(tmpdir, "splits")
        csv_path = os.path.join(tmpdir, "dataset.csv")
        results_dir = os.path.join(tmpdir, "results")
        grid_params_path = os.path.join(tmpdir, "grid_params.json")
        
        # Create directories
        os.makedirs(features_path, exist_ok=True)
        os.makedirs(splits_dir, exist_ok=True)
        
        # Create a minimal CSV
        data = pd.DataFrame({
            "case_id": ["case1", "case2"],
            "slide_id": ["slide1", "slide2"],
            "target": [0, 1]
        })
        data.to_csv(csv_path, index=False)
        
        # Create grid params with multiple values
        grid_params = {
            "n_classes": [2, 3],
            "dropout": [0.25, 0.5]
        }
        with open(grid_params_path, "w") as f:
            json.dump(grid_params, f)
        
        # Test instantiation with all parameters
        grid_search = GridSearch(
            folds=3,
            features_path=features_path,
            splits_dir=splits_dir,
            csv_path=csv_path,
            results_dir=results_dir,
            feature_extractor="custom_extractor",
            epochs=20,
            learning_rate=1e-3,
            mil="clam",
            use_class_weights=False,
            grid_params_path=grid_params_path
        )
        
        assert grid_search.folds == 3
        assert grid_search.feature_extractor == "custom_extractor"
        assert grid_search.epochs == 20
        assert grid_search.learning_rate == 1e-3
        assert grid_search.mil == "clam"
        assert grid_search.use_class_weights is False
        assert grid_search.batch_size == 1  # CLAM uses batch_size=1
        assert len(grid_search.param_combinations) == 4  # 2 * 2 = 4 combinations

