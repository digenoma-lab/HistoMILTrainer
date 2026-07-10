"""Tests for transfer-learning utilities."""

import importlib.util
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_PATH = PROJECT_ROOT / "histomil" / "transfer.py"

spec = importlib.util.spec_from_file_location("transfer_utils", TRANSFER_PATH)
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )
        self.norm = nn.LayerNorm(4)
        self.classifier = nn.Linear(4, 2)


def assert_raises(exception_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception_type:
        return
    raise AssertionError(f"Se esperaba {exception_type.__name__}")


def test_freeze_and_unfreeze():
    model = DummyModel()
    transfer.freeze_all_parameters(model)
    assert all(not p.requires_grad for p in model.parameters())

    transfer.unfreeze_all_parameters(model)
    assert all(p.requires_grad for p in model.parameters())


def test_parameter_counts_and_patterns():
    model = DummyModel()
    total = sum(p.numel() for p in model.parameters())

    transfer.freeze_all_parameters(model)
    matched = transfer.unfreeze_parameters_by_patterns(model, ["classifier"])

    assert matched == ["classifier.weight", "classifier.bias"]
    counts = transfer.count_parameters(model)
    assert counts["total"] == total
    assert counts["trainable"] == 10
    assert counts["frozen"] == total - 10

    names = transfer.get_trainable_parameter_names(model)
    assert names == ["classifier.weight", "classifier.bias"]

    parameters = transfer.get_trainable_parameters(model)
    assert len(parameters) == 2


def test_pattern_validation():
    model = DummyModel()
    transfer.freeze_all_parameters(model)

    assert_raises(RuntimeError, transfer.unfreeze_parameters_by_patterns, model, ["unknown_module"])
    assert_raises(ValueError, transfer.unfreeze_parameters_by_patterns, model, [])


def test_find_classification_heads():
    model = DummyModel()
    heads = transfer.find_classification_heads(model, num_classes=2)

    assert len(heads) == 1
    assert heads[0][0] == "classifier"
    assert heads[0][1].out_features == 2


def test_configure_head_only():
    model = DummyModel()
    trainable_modules = transfer.configure_head_only(model, num_classes=2)
    names = transfer.get_trainable_parameter_names(model)

    assert trainable_modules == ["classifier"]
    assert names == ["classifier.weight", "classifier.bias"]


def test_configure_partial():
    model = DummyModel()
    trainable_modules = transfer.configure_partial(model, num_classes=2, unfreeze_modules=6)
    names = transfer.get_trainable_parameter_names(model)

    assert "classifier" in trainable_modules
    assert "classifier.weight" in names
    assert "classifier.bias" in names
    assert len(names) >= 2


def test_config_loading():
    config = {"example_model": {"head": ["classifier"], "partial": ["encoder"]}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as file:
        json.dump(config, file)
        config_path = file.name

    try:
        loaded = transfer.load_transfer_layers_config(config_path)
        assert loaded == config
    finally:
        Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_freeze_and_unfreeze()
    test_parameter_counts_and_patterns()
    test_pattern_validation()
    test_find_classification_heads()
    test_configure_head_only()
    test_configure_partial()
    test_config_loading()
    print("OK: todas las utilidades de transferencia funcionan correctamente")