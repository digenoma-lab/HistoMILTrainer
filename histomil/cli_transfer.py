"""Command-line interface for MIL transfer training."""

import argparse
import logging

from histomil.cli import setup_logging


def grid_search_transfer():
    """Run fixed-parameter MIL transfer training."""
    from histomil.grid_search_transfer import GridSearchTransfer

    parser = argparse.ArgumentParser(
        description="MIL transfer training with fixed model parameters"
    )

    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--feature_extractor", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--mil", type=str, required=True)
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=True,)
    parser.add_argument("--model_params_path", type=str, required=True,help="JSON con los parámetros fijos del modelo fuente.",)
    parser.add_argument("--transfer_mode", type=str, required=True, choices=["scratch", "head_only", "partial", "full"], help="Estrategia de transferencia: scratch, head_only, partial o full.",)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None, help="Checkpoint fuente requerido para transfer_mode=head_only, transfer_mode=partial o transfer_mode=full.",)
    parser.add_argument("--partial_unfreeze_modules", type=int, default=2, help="Número de módulos a descongelar en el modo parcial.",)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],)

    args = parser.parse_args()

    setup_logging(
        level=getattr(logging, args.log_level.upper())
    )

    transfer_training = GridSearchTransfer(
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
        model_params_path=args.model_params_path,
        transfer_mode=args.transfer_mode,
        pretrained_checkpoint=args.pretrained_checkpoint,
        partial_unfreeze_modules=args.partial_unfreeze_modules,
    )

    transfer_training.run()
