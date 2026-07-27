"""Tests for fixed-parameter MIL training."""

import json
import os
import tempfile

import pandas as pd
import pytest

from histomil import FixedParameterTrainer


def create_test_files(tmpdir, params):
    features_path = os.path.join(tmpdir, "features")
    splits_dir = os.path.join(tmpdir, "splits")
    results_dir = os.path.join(tmpdir, "results")
    csv_path = os.path.join(tmpdir, "dataset.csv")
    params_path = os.path.join(tmpdir, "params.json")

    os.makedirs(features_path)
    os.makedirs(splits_dir)

    pd.DataFrame({
        "slide_id": ["slide1", "slide2"],
        "label": [0, 1],
    }).to_csv(csv_path, index=False)

    pd.DataFrame({
        "slide_id": ["slide1", "slide2"],
        "train": [True, False],
        "val": [False, True],
        "test": [False, False],
    }).to_csv(
        os.path.join(splits_dir, "splits_0_bool.csv"),
        index=False,
    )

    pd.DataFrame({
        "": [0, 1],
        "train": [1, 1],
        "val": [1, 1],
        "test": [1, 1],
    }).to_csv(
        os.path.join(splits_dir, "splits_0_descriptor.csv"),
        index=False,
    )

    with open(params_path, "w", encoding="utf-8") as file:
        json.dump(params, file)

    return features_path, splits_dir, csv_path, results_dir, params_path


def test_fixed_parameter_trainer_instantiation():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = create_test_files(
            tmpdir,
            {"embed_dim": 512, "dropout": 0.25},
        )

        trainer = FixedParameterTrainer(
            folds=1,
            features_path=paths[0],
            splits_dir=paths[1],
            csv_path=paths[2],
            results_dir=paths[3],
            params_path=paths[4],
            feature_extractor="virchow2",
            mil="abmil",
            transfer_mode="scratch",
        )

        assert trainer.folds == 1
        assert trainer.transfer_mode == "scratch"
        assert trainer.pretrained_checkpoint is None
        assert trainer.params == {
            "embed_dim": 512,
            "dropout": 0.25,
        }
        assert trainer.batch_size == 16


def test_fixed_parameter_trainer_rejects_parameter_lists():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = create_test_files(
            tmpdir,
            {"embed_dim": [256, 512], "dropout": 0.25},
        )

        with pytest.raises(ValueError, match="List-valued parameters"):
            FixedParameterTrainer(
                folds=1,
                features_path=paths[0],
                splits_dir=paths[1],
                csv_path=paths[2],
                results_dir=paths[3],
                params_path=paths[4],
            )


def test_head_only_requires_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = create_test_files(
            tmpdir,
            {"embed_dim": 512, "dropout": 0.25},
        )

        with pytest.raises(ValueError, match="requires pretrained_checkpoint"):
            FixedParameterTrainer(
                folds=1,
                features_path=paths[0],
                splits_dir=paths[1],
                csv_path=paths[2],
                results_dir=paths[3],
                params_path=paths[4],
                transfer_mode="head_only",
            )
