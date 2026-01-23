"""Test that GridSearch, SplitManager, Predictor and HeatmapVisualizer classes can be instantiated."""
import os
import tempfile
import json
import pandas as pd

from histomil import GridSearch, SplitManager, Predictor, HeatmapVisualizer


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


def test_predictor_instantiation():
    """Test that Predictor can be instantiated with required parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_dataset.csv")
        weights_path = os.path.join(tmpdir, "weights.pt")
        features_folder = os.path.join(tmpdir, "features")
        results_dir = os.path.join(tmpdir, "results")
        params_path = os.path.join(tmpdir, "params.json")
        
        # Create directories
        os.makedirs(features_folder, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a minimal CSV
        data = pd.DataFrame({
            "slide_id": ["slide1", "slide2"],
            "target": [0, 1]
        })
        data.to_csv(csv_path, index=False)
        
        # Create a minimal params JSON
        params = {
            "n_classes": 2,
            "dropout": 0.25
        }
        with open(params_path, "w") as f:
            json.dump(params, f)
        
        # Create a dummy weights file (empty, just for testing instantiation)
        # Note: This won't work for actual model loading, but tests instantiation
        import torch
        dummy_state = {"layer.weight": torch.randn(1, 1)}
        torch.save(dummy_state, weights_path)
        
        # Test instantiation with required parameters
        predictor = Predictor(
            csv_path=csv_path,
            weights_path=weights_path,
            features_folder=features_folder,
            feature_extractor="uni_v2",
            results_dir=results_dir,
            mil="abmil",
            params_path=params_path
        )
        
        assert predictor.csv_path == csv_path
        assert predictor.weights_path == weights_path
        assert predictor.features_folder == features_folder
        assert predictor.feature_extractor == "uni_v2"
        assert predictor.results_dir == results_dir
        assert predictor.mil == "abmil"
        assert predictor.params_path == params_path
        assert predictor.batch_size == 16  # default for abmil
        assert predictor.SEED == 2
        assert predictor.BATCH_SIZE == 16


def test_predictor_instantiation_with_clam():
    """Test that Predictor can be instantiated with CLAM (batch_size=1)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_dataset.csv")
        weights_path = os.path.join(tmpdir, "weights.pt")
        features_folder = os.path.join(tmpdir, "features")
        results_dir = os.path.join(tmpdir, "results")
        params_path = os.path.join(tmpdir, "params.json")
        
        # Create directories
        os.makedirs(features_folder, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a minimal CSV
        data = pd.DataFrame({
            "slide_id": ["slide1", "slide2"],
            "target": [0, 1]
        })
        data.to_csv(csv_path, index=False)
        
        # Create a minimal params JSON
        params = {
            "n_classes": 2,
            "dropout": 0.25
        }
        with open(params_path, "w") as f:
            json.dump(params, f)
        
        # Create a dummy weights file
        import torch
        dummy_state = {"layer.weight": torch.randn(1, 1)}
        torch.save(dummy_state, weights_path)
        
        # Test instantiation with CLAM (should use batch_size=1)
        predictor = Predictor(
            csv_path=csv_path,
            weights_path=weights_path,
            features_folder=features_folder,
            feature_extractor="uni_v2",
            results_dir=results_dir,
            mil="clam",
            params_path=params_path
        )
        
        assert predictor.mil == "clam"
        assert predictor.batch_size == 1  # CLAM uses batch_size=1


def test_predictor_drop_extension():
    """Test Predictor.drop_extension static method."""
    # Test with full path
    assert Predictor.drop_extension("/path/to/file.h5") == "file"
    # Test with just filename
    assert Predictor.drop_extension("slide1.h5") == "slide1"
    # Test with no extension
    assert Predictor.drop_extension("slide1") == "slide1"
    # Test with multiple dots
    assert Predictor.drop_extension("slide.1.2.h5") == "slide.1.2"


def test_predictor_load_data():
    """Test Predictor._load_data method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_dataset.csv")
        weights_path = os.path.join(tmpdir, "weights.pt")
        features_folder = os.path.join(tmpdir, "features")
        results_dir = os.path.join(tmpdir, "results")
        params_path = os.path.join(tmpdir, "params.json")
        
        # Create directories
        os.makedirs(features_folder, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a minimal CSV
        data = pd.DataFrame({
            "slide_id": ["slide1", "slide2", "slide3"],
            "target": [0, 1, 0]
        })
        data.to_csv(csv_path, index=False)
        
        # Create a minimal params JSON
        params = {"n_classes": 2}
        with open(params_path, "w") as f:
            json.dump(params, f)
        
        # Create a dummy weights file
        import torch
        dummy_state = {"layer.weight": torch.randn(1, 1)}
        torch.save(dummy_state, weights_path)
        
        predictor = Predictor(
            csv_path=csv_path,
            weights_path=weights_path,
            features_folder=features_folder,
            feature_extractor="uni_v2",
            results_dir=results_dir,
            mil="abmil",
            params_path=params_path
        )
        
        # Test _load_data method
        loaded_data = predictor._load_data(csv_path)
        assert len(loaded_data) == 3
        assert list(loaded_data.columns) == ["slide_id", "target"]
        assert list(loaded_data["slide_id"]) == ["slide1", "slide2", "slide3"]


def test_heatmap_visualizer_instantiation():
    """Test that HeatmapVisualizer can be instantiated with required parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        slide_id = "test_slide.svs"
        slide_folder = os.path.join(tmpdir, "slides")
        features_folder = os.path.join(tmpdir, "features")
        attn_scores_folder = os.path.join(tmpdir, "attention_scores")
        results_dir = os.path.join(tmpdir, "results")
        
        # Create directories
        os.makedirs(slide_folder, exist_ok=True)
        os.makedirs(features_folder, exist_ok=True)
        os.makedirs(attn_scores_folder, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Test instantiation with required parameters
        visualizer = HeatmapVisualizer(
            slide_id=slide_id,
            slide_folder=slide_folder,
            features_folder=features_folder,
            attn_scores_folder=attn_scores_folder,
            results_dir=results_dir
        )
        
        assert visualizer.slide_id == slide_id
        assert visualizer.slide_folder == slide_folder
        assert visualizer.features_folder == features_folder
        assert visualizer.attn_scores_folder == attn_scores_folder
        assert visualizer.results_dir == results_dir


def test_heatmap_visualizer_drop_extension():
    """Test HeatmapVisualizer.drop_extension static method."""
    # Test with full path
    assert HeatmapVisualizer.drop_extension("/path/to/file.svs") == "file"
    # Test with just filename
    assert HeatmapVisualizer.drop_extension("slide1.svs") == "slide1"
    # Test with no extension
    assert HeatmapVisualizer.drop_extension("slide1") == "slide1"
    # Test with multiple dots
    assert HeatmapVisualizer.drop_extension("slide.1.2.svs") == "slide.1.2"
    # Test with .h5 extension
    assert HeatmapVisualizer.drop_extension("slide1.h5") == "slide1"

