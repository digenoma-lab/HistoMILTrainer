"""Utilities for configuring trainable parameters in transfer learning."""

import json
from pathlib import Path


def freeze_all_parameters(model):
    """Freeze every parameter in the model."""
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_all_parameters(model):
    """Make every parameter in the model trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def get_trainable_parameters(model):
    """Return parameters currently marked as trainable."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("The model does not contain trainable parameters.")
    return parameters


def count_parameters(model):
    """Count total, trainable and frozen parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def get_trainable_parameter_names(model):
    """Return names of all trainable parameters."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def get_frozen_parameter_names(model):
    """Return names of all frozen parameters."""
    return [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]

def find_classification_heads(model, num_classes=2):
    """Find classification heads producing class logits."""
    from torch import nn

    heads = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            heads.append((name, module))
        elif isinstance(module, nn.Conv1d) and module.out_channels == num_classes:
            heads.append((name, module))

    if not heads:
        raise RuntimeError(
            f"No se encontraron cabezales Linear/Conv1d con salida={num_classes}."
        )

    return heads


def reset_linear_layer(module):
    """Reset parameters of a linear layer."""
    module.reset_parameters()


def configure_head_only(model, num_classes=2):
    """Freeze the model, reset heads and enable only classification heads."""
    freeze_all_parameters(model)
    heads = find_classification_heads(model, num_classes)

    trainable_modules = []
    for name, module in heads:
        reset_linear_layer(module)
        for parameter in module.parameters():
            parameter.requires_grad = True
        trainable_modules.append(name)

    return trainable_modules

def load_transfer_layers_config(config_path):
    """Load external layer-selection rules."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No existe la configuración de capas: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict) or not config:
        raise ValueError("La configuración de capas debe ser un objeto JSON no vacío.")

    return config


def unfreeze_parameters_by_patterns(model, patterns):
    """Unfreeze parameters matching module names or prefixes."""
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("La lista de patrones no puede estar vacía.")

    matched = []
    for name, parameter in model.named_parameters():
        if any(name == pattern or name.startswith(f"{pattern}.") for pattern in patterns):
            parameter.requires_grad = True
            matched.append(name)

    if not matched:
        raise RuntimeError(f"No se encontraron parámetros para los patrones: {patterns}")

    return matched

def get_parameterized_leaf_modules(model):
    """Return leaf modules that own parameters directly."""
    modules = []

    for name, module in model.named_modules():
        if name == "":
            continue

        has_children = any(True for _ in module.children())
        has_parameters = any(
            parameter.numel() > 0
            for parameter in module.parameters(recurse=False)
        )

        if not has_children and has_parameters:
            modules.append((name, module))

    if not modules:
        raise RuntimeError("No se encontraron módulos hoja con parámetros.")

    return modules


def get_partial_unfreeze_patterns(mil, unfreeze_modules=2):
    """Return architecture-aware parameter prefixes for partial fine-tuning."""
    mil = mil.lower()
    rules = {
        "abmil": ["model.global_attn", "model.classifier"],
        "clam": ["model.global_attn", "model.classifier", "model.instance_classifiers"],
        "dsmil": ["model.i_classifier", "model.b_classifier", "model.classifier"],
        "dftd": ["model.patch_embed.resBlocks", "model.classifier", "model.attCls"],
        "ilra": ["model.pooling", "model.classifier"],
        "rrt": ["model.encoder", "model.classifier"],
        "transformer": ["model.encoder", "model.transformer", "model.blocks", "model.norm", "model.classifier"],
        "transmil": ["model.blocks", "model.pos_layer", "model.norm", "model.classifier"],
        "wikg": ["model.gate_U", "model.gate_V", "model.gate_W", "model.W_head", "model.W_tail", "model.linear1", "model.linear2", "model.readout", "model.norm", "model.classifier"],
    }
    return rules.get(mil, ["model.classifier", "model.instance_classifiers"])


def configure_partial(model, num_classes=2, unfreeze_modules=2, mil=None):
    """Freeze the model, reset heads and unfreeze architecture-aware modules."""
    if unfreeze_modules < 1:
        raise ValueError("unfreeze_modules debe ser mayor o igual a 1.")
    freeze_all_parameters(model)
    heads = find_classification_heads(model, num_classes)
    trainable_modules = []
    if mil is not None:
        patterns = get_partial_unfreeze_patterns(mil, unfreeze_modules=unfreeze_modules)
        matched = []
        for name, parameter in model.named_parameters():
            if any(name == pattern or name.startswith(f"{pattern}.") for pattern in patterns):
                parameter.requires_grad = True
                matched.append(name)
        if matched:
            trainable_modules.extend(sorted(set(name.rsplit(".", 1)[0] for name in matched)))
        else:
            leaf_modules = get_parameterized_leaf_modules(model)
            selected_modules = leaf_modules[-unfreeze_modules:]
            for name, module in selected_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
                trainable_modules.append(name)
    else:
        leaf_modules = get_parameterized_leaf_modules(model)
        selected_modules = leaf_modules[-unfreeze_modules:]
        for name, module in selected_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
            trainable_modules.append(name)
    for name, module in heads:
        reset_linear_layer(module)
        for parameter in module.parameters():
            parameter.requires_grad = True
        trainable_modules.append(name)
    return list(dict.fromkeys(trainable_modules))
