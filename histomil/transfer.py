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



def load_req_grid(mil):
    """Load functional transfer rules for partial mode."""
    if mil is None:
        raise ValueError("mil es requerido para cargar req_grid.")
    req_grid_path = Path(__file__).resolve().parent / "configs" / "req_grid" / f"{mil.lower()}.json"
    if not req_grid_path.is_file():
        raise FileNotFoundError(f"No existe req_grid para partial: {req_grid_path}")
    with req_grid_path.open("r", encoding="utf-8") as file:
        req_grid = json.load(file)
    if not isinstance(req_grid, dict):
        raise ValueError(f"req_grid debe ser un objeto JSON: {req_grid_path}")
    if "groups" not in req_grid or not isinstance(req_grid["groups"], dict):
        raise ValueError(f"req_grid debe contener un objeto 'groups': {req_grid_path}")
    return req_grid


def find_matching_layers(parameter_name, layers):
    return [layer_name for layer_name in layers if parameter_name == layer_name or parameter_name.startswith(f"{layer_name}.")]


def find_block_indices(model, block_prefix):
    indices = set()
    prefix = f"{block_prefix}."
    for parameter_name, _ in model.named_parameters():
        if not parameter_name.startswith(prefix):
            continue
        first = parameter_name[len(prefix):].split(".", 1)[0]
        if first.isdigit():
            indices.add(int(first))
    return sorted(indices)



def parse_binary(value, field_name):
    """Convert a JSON 0/1 value into bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{field_name} debe ser 0 o 1; recibido: {value!r}")


def normalize_req_grid(req_grid):
    """Convert functional groups into layer and repeated-block rules."""
    layer_rules = {}
    block_rules = {}

    for group_name, group in req_grid["groups"].items():
        if not isinstance(group, dict):
            raise ValueError(f"Grupo inválido en req_grid: {group_name}")

        trainable = parse_binary(
            group.get("trainable", 0),
            f"groups.{group_name}.trainable",
        )

        layers = group.get("layers", [])
        blocks = group.get("blocks", [])

        if not isinstance(layers, list):
            raise ValueError(f"groups.{group_name}.layers debe ser una lista.")
        if not isinstance(blocks, list):
            raise ValueError(f"groups.{group_name}.blocks debe ser una lista.")

        for layer_name in layers:
            if not isinstance(layer_name, str) or not layer_name:
                raise ValueError(f"Ruta de capa inválida en grupo {group_name}.")
            layer_rules[layer_name] = trainable

        for block_prefix in blocks:
            if not isinstance(block_prefix, str) or not block_prefix:
                raise ValueError(f"Ruta de bloques inválida en grupo {group_name}.")
            block_rules[block_prefix] = trainable

    return layer_rules, block_rules


def apply_layer_rules(model, layer_rules):
    matched_layers = {layer_name: 0 for layer_name in layer_rules}

    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad = False
        matches = find_matching_layers(parameter_name, layer_rules)

        if matches:
            for layer_name in matches:
                matched_layers[layer_name] += 1

            selected_layer = max(matches, key=len)
            parameter.requires_grad = layer_rules[selected_layer]

    return matched_layers


def apply_block_rules(model, block_rules):
    matched_blocks = {}

    for block_prefix, trainable in block_rules.items():
        block_indices = find_block_indices(model, block_prefix)
        matched_blocks[block_prefix] = len(block_indices)

        if not block_indices:
            continue

        prefix = f"{block_prefix}."

        for parameter_name, parameter in model.named_parameters():
            if parameter_name.startswith(prefix):
                parameter.requires_grad = trainable

    return matched_blocks


def apply_req_grid(model, req_grid):
    full_finetune = parse_binary(
        req_grid.get("full_finetune", 0),
        "full_finetune",
    )
    strict = parse_binary(
        req_grid.get("strict", 1),
        "strict",
    )

    if full_finetune:
        unfreeze_all_parameters(model)
        trainable_names = get_trainable_parameter_names(model)

        if not trainable_names:
            raise RuntimeError("El modelo no contiene parámetros entrenables.")

        return trainable_names

    layer_rules, block_rules = normalize_req_grid(req_grid)
    matched_layers = apply_layer_rules(model, layer_rules)
    matched_blocks = apply_block_rules(model, block_rules)

    missing_layers = [
        layer_name
        for layer_name, count in matched_layers.items()
        if count == 0
    ]
    missing_blocks = [
        block_prefix
        for block_prefix, count in matched_blocks.items()
        if count == 0
    ]

    if strict and (missing_layers or missing_blocks):
        raise RuntimeError(
            "req_grid apunta a capas o bloques inexistentes. "
            f"missing_layers={missing_layers}; "
            f"missing_blocks={missing_blocks}"
        )

    trainable_names = get_trainable_parameter_names(model)

    if not trainable_names:
        raise RuntimeError(
            "req_grid dejó el modelo sin parámetros entrenables."
        )

    return trainable_names

def configure_partial(model, num_classes=2, unfreeze_modules=2, mil=None):
    """Configure partial transfer using functional req_grid groups."""
    freeze_all_parameters(model)
    req_grid = load_req_grid(mil)
    trainable_names = apply_req_grid(model, req_grid)

    for head_name, head_module in find_classification_heads(model, num_classes):
        if any(parameter.requires_grad for parameter in head_module.parameters()):
            reset_linear_layer(head_module)

    trainable_modules = sorted(set(name.rsplit(".", 1)[0] for name in trainable_names))
    return trainable_modules
