"""Inspect modules and parameters of one MIL model."""

import argparse
import json
from pathlib import Path

from histomil.models import import_model


def load_params(params_path):
    path = Path(params_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No existe el JSON de parámetros: {path}")

    with path.open("r", encoding="utf-8") as file:
        params = json.load(file)

    if not isinstance(params, dict) or not params:
        raise ValueError("El JSON de parámetros debe contener un objeto no vacío.")

    return path, params


def build_report(model, architecture, feature_extractor, params_path, params):
    modules = [
        {"name": name or "<root>", "type": module.__class__.__name__}
        for name, module in model.named_modules()
    ]

    parameters = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "requires_grad": parameter.requires_grad,
        }
        for name, parameter in model.named_parameters()
    ]

    return {
        "architecture": architecture,
        "feature_extractor": feature_extractor,
        "model_params_path": str(params_path),
        "model_params": params,
        "total_parameters": sum(parameter["numel"] for parameter in parameters),
        "modules": modules,
        "parameters": parameters,
    }


def print_report(report):
    print("\n=== MODELO ===")
    print(f"Arquitectura: {report['architecture']}")
    print(f"Extractor: {report['feature_extractor']}")
    print(f"Parámetros totales: {report['total_parameters']}")

    print("\n=== MÓDULOS ===")
    for module in report["modules"]:
        print(f"{module['name']:<80} {module['type']}")

    print("\n=== PARÁMETROS ===")
    for parameter in report["parameters"]:
        print(
            f"{parameter['name']:<90} "
            f"shape={str(parameter['shape']):<24} "
            f"numel={parameter['numel']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Inspect one MIL-Lab model.")
    parser.add_argument("--architecture",required=True, help="Arquitectura MIL aceptada por MIL-Lab.",)
    parser.add_argument("--feature_extractor", required=True, help="Extractor de características aceptado por MIL-Lab.",)
    parser.add_argument("--model_params_path", required=True, help="JSON con los parámetros utilizados para construir el modelo.",)
    parser.add_argument("--output", default=None, help="Ruta opcional del reporte JSON.",)
    args = parser.parse_args()

    params_path, params = load_params(args.model_params_path)
    model = import_model(args.architecture, args.feature_extractor, **params)

    report = build_report(
        model=model,
        architecture=args.architecture,
        feature_extractor=args.feature_extractor,
        params_path=params_path,
        params=params,
    )

    print_report(report)

    output_path = Path(
        args.output
        or f"reports/model_structure/{args.feature_extractor}_{args.architecture}.json"
    ).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nReporte guardado en: {output_path}")


if __name__ == "__main__":
    main()