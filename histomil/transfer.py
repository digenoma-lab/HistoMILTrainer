def freeze_all_parameters(model):
    """Freeze every parameter in the model."""
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_all_parameters(model):
    """Make every parameter in the model trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def get_trainable_parameters(model):
    """Return the parameters currently marked as trainable."""
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "The model does not contain trainable parameters."
        )

    return trainable_parameters


def count_parameters(model):
    """Count total, trainable and frozen model parameters."""
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def get_trainable_parameter_names(model):
    """Return the names of all trainable parameters."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def get_frozen_parameter_names(model):
    """Return the names of all frozen parameters."""
    return [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]