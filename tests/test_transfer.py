"""Tests for generic transfer-learning utilities without PyTorch."""

import importlib.util
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_PATH = PROJECT_ROOT / "histomil" / "transfer.py"

spec = importlib.util.spec_from_file_location("transfer_utils", TRANSFER_PATH)
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class DummyParameter:
    def __init__(self, size):
        self.size = size
        self.requires_grad = True

    def numel(self):
        return self.size


class DummyModel:
    def __init__(self):
        self._parameters = [
            ("encoder.weight", DummyParameter(20)),
            ("encoder.bias", DummyParameter(5)),
            ("classifier.weight", DummyParameter(10)),
            ("classifier.bias", DummyParameter(2)),
        ]

    def parameters(self):
        return [parameter for _, parameter in self._parameters]

    def named_parameters(self):
        return iter(self._parameters)


def assert_raises(exception_type, function, *args):
    try:
        function(*args)
    except exception_type:
        return
    raise AssertionError(f"Se esperaba {exception_type.__name__}")


def test_freeze_and_unfreeze():
    model = DummyModel()
    transfer.freeze_all_parameters(model)
    assert all(not parameter.requires_grad for parameter in model.parameters())

    transfer.unfreeze_all_parameters(model)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_parameter_counts():
    model = DummyModel()
    transfer.freeze_all_parameters(model)
    transfer.unfreeze_parameters_by_patterns(model, ["classifier"])

    counts = transfer.count_parameters(model)
    assert counts == {"total": 37, "trainable": 12, "frozen": 25}

    names = transfer.get_trainable_parameter_names(model)
    assert names == ["classifier.weight", "classifier.bias"]

    parameters = transfer.get_trainable_parameters(model)
    assert len(parameters) == 2


def test_pattern_validation():
    model = DummyModel()
    transfer.freeze_all_parameters(model)

    assert_raises(
        RuntimeError,
        transfer.unfreeze_parameters_by_patterns,
        model,
        ["unknown_module"],
    )

    assert_raises(
        ValueError,
        transfer.unfreeze_parameters_by_patterns,
        model,
        [],
    )


def test_config_loading():
    config = {
        "example_model": {
            "head": ["classifier"],
            "partial": ["encoder"],
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as file:
        json.dump(config, file)
        config_path = file.name

    try:
        loaded = transfer.load_transfer_layers_config(config_path)
        assert loaded == config
    finally:
        Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_freeze_and_unfreeze()
    test_parameter_counts()
    test_pattern_validation()
    test_config_loading()
    print("OK: todas las utilidades de transferencia funcionan correctamente")
