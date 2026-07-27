"""Tests for transfer-learning utilities."""

import pytest
from torch import nn

from histomil import transfer


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


def test_freeze_and_unfreeze():
    model = DummyModel()

    transfer.freeze_all_parameters(model)
    assert all(not parameter.requires_grad for parameter in model.parameters())

    transfer.unfreeze_all_parameters(model)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_parameter_counts_and_names():
    model = DummyModel()
    transfer.freeze_all_parameters(model)

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    counts = transfer.count_parameters(model)
    names = transfer.get_trainable_parameter_names(model)
    parameters = transfer.get_trainable_parameters(model)

    assert counts["total"] == 74
    assert counts["trainable"] == 10
    assert counts["frozen"] == 64
    assert names == ["classifier.weight", "classifier.bias"]
    assert len(parameters) == 2


def test_get_trainable_parameters_rejects_fully_frozen_model():
    model = DummyModel()
    transfer.freeze_all_parameters(model)

    with pytest.raises(RuntimeError, match="does not contain trainable"):
        transfer.get_trainable_parameters(model)


def test_find_classification_heads():
    model = DummyModel()
    heads = transfer.find_classification_heads(model, num_classes=2)

    assert len(heads) == 1
    assert heads[0][0] == "classifier"
    assert heads[0][1] is model.classifier


def test_configure_head_only():
    model = DummyModel()
    trainable_modules = transfer.configure_head_only(model, num_classes=2)
    names = transfer.get_trainable_parameter_names(model)

    assert trainable_modules == ["classifier"]
    assert names == ["classifier.weight", "classifier.bias"]


def test_apply_req_grid():
    model = DummyModel()
    req_grid = {
        "full_finetune": 0,
        "strict": 1,
        "groups": {
            "final_encoder": {
                "trainable": 1,
                "layers": ["encoder.2"],
            },
            "classifier": {
                "trainable": 1,
                "layers": ["classifier"],
            },
        },
    }

    names = transfer.apply_req_grid(model, req_grid)

    assert names == [
        "encoder.2.weight",
        "encoder.2.bias",
        "classifier.weight",
        "classifier.bias",
    ]


def test_apply_req_grid_full_finetune():
    model = DummyModel()
    req_grid = {
        "full_finetune": 1,
        "strict": 1,
        "groups": {},
    }

    names = transfer.apply_req_grid(model, req_grid)

    assert names == [
        name for name, _ in model.named_parameters()
    ]
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_apply_req_grid_strict_rejects_missing_layer():
    model = DummyModel()
    req_grid = {
        "full_finetune": 0,
        "strict": 1,
        "groups": {
            "missing": {
                "trainable": 1,
                "layers": ["unknown.layer"],
            },
        },
    }

    with pytest.raises(RuntimeError, match="missing_layers"):
        transfer.apply_req_grid(model, req_grid)


def test_configure_partial_uses_req_grid(monkeypatch):
    model = DummyModel()
    req_grid = {
        "full_finetune": 0,
        "strict": 1,
        "groups": {
            "final_encoder": {
                "trainable": 1,
                "layers": ["encoder.2"],
            },
            "classifier": {
                "trainable": 1,
                "layers": ["classifier"],
            },
        },
    }

    monkeypatch.setattr(
        transfer,
        "load_req_grid",
        lambda mil: req_grid,
    )

    modules = transfer.configure_partial(
        model,
        num_classes=2,
        mil="dummy",
    )
    names = transfer.get_trainable_parameter_names(model)

    assert modules == ["classifier", "encoder.2"]
    assert names == [
        "encoder.2.weight",
        "encoder.2.bias",
        "classifier.weight",
        "classifier.bias",
    ]


@pytest.mark.parametrize(
    "mil",
    [
        "abmil",
        "clam",
        "dftd",
        "dsmil",
        "ilra",
        "rrt",
        "transformer",
        "transmil",
        "wikg",
    ],
)
def test_architecture_req_grid_can_be_loaded(mil):
    req_grid = transfer.load_req_grid(mil)

    assert isinstance(req_grid, dict)
    assert "full_finetune" in req_grid
    assert "strict" in req_grid
    assert isinstance(req_grid["groups"], dict)
    assert req_grid["groups"]
