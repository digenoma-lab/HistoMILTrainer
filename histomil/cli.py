"""Command-line interface for HistoMILTrainer."""
import argparse

from histomil import SplitManager, GridSearch


def make_splits():
    """CLI entry point for making splits."""
    parser = argparse.ArgumentParser(description="HistoMIL Make Splits Script")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, default="./splits")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--output_name", type=str, required=True)
    args = parser.parse_args()

    split_manager = SplitManager(
        csv_path=args.csv_path,
        output_name=args.output_name,
        folds=args.folds,
        splits_dir=args.splits_dir,
        test_frac=args.test_frac,
        target=args.target,
    )
    split_manager.create_splits()


def grid_search():
    """CLI entry point for grid search."""
    parser = argparse.ArgumentParser(description="MIL Grid Search")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./temp_dir/")
    parser.add_argument("--feature_extractor", type=str, default="uni_v2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--mil", type=str, default="abmil")
    parser.add_argument("--use_class_weights", type=bool, default=True)
    parser.add_argument("--grid_params", type=str, default="configs/abmil.json")
    args = parser.parse_args()

    # Create GridSearch instance and run
    grid_search = GridSearch(
        folds=args.folds,
        features_path=args.features_path,
        splits_dir=args.splits_dir,
        csv_path=args.csv_path,
        results_dir=args.results_dir,
        feature_extractor=args.feature_extractor,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mil=args.mil,
        use_class_weights=args.use_class_weights,
        grid_params_path=args.grid_params,
    )
    grid_search.run()

