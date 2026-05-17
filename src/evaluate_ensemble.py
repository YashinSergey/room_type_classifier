from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.convnext_tiny.model import build_convnext_tiny
from models.densenet121.densenet121 import build_densenet121
from models.resnet18.resnet18 import build_resnet18
from src.dataloaders import create_dataloaders
from src.device import get_default_device
from src.labels import load_label_mapping
from src.metrics import calculate_accuracy, calculate_macro_f1, calculate_per_class_f1
from src.mlflow_utils import (
    end_mlflow_run,
    log_mlflow_artifacts,
    log_mlflow_metrics,
    log_mlflow_params,
    start_mlflow_run,
)
from src.training_helpers import load_json, save_json, to_project_relative_path

DEFAULT_CHECKPOINTS = [
    ROOT_DIR / "outputs" / "models" / "convnext_nano" / "convnext_nano_best.pt",
    ROOT_DIR / "outputs" / "models" / "resnet50" / "resnet50_best.pt",
    ROOT_DIR / "outputs" / "models" / "resnet18" / "resnet18_best.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка ансамбля на валидации", add_help=False)
    parser._optionals.title = "Аргументы"
    parser.add_argument("-h", "--help", action="help", help="Показать справку и выйти.")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=[path for path in DEFAULT_CHECKPOINTS if path.exists()],
        help="Пути к checkpoint-файлам моделей.",
    )
    parser.add_argument("--val-csv", type=Path, default=ROOT_DIR / "data" / "processed" / "val_df.csv")
    parser.add_argument("--val-images", type=Path, default=ROOT_DIR / "data" / "raw" / "val_images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--weighting",
        choices=["uniform", "val_f1"],
        default="uniform",
        help="uniform: равные веса; val_f1: веса по macro-F1 из checkpoint.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Свои веса моделей в том же порядке, что и --checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "reports" / "metrics" / "ensemble")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-predictions", action="store_true", help="Сохранить предсказания на валидации.")
    parser.add_argument("--log-mlflow", action="store_true", help="Залогировать результат в MLflow.")
    parser.add_argument(
        "--search-subsets",
        action="store_true",
        help="Перебрать все подмножества моделей и найти лучший ансамбль.",
    )
    parser.add_argument(
        "--mlflow-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="При --log-mlflow писать в локальный mlflow.db/mlruns.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_convnext_nano(num_classes: int) -> nn.Module:
    import timm

    return timm.create_model(
        "convnext_nano",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.5,
        drop_path_rate=0.3,
    )


def build_efficientnet(variant: str, num_classes: int) -> nn.Module:
    if variant == "b0":
        model = efficientnet_b0(weights=None)
    elif variant == "b1":
        model = efficientnet_b1(weights=None)
    else:
        raise ValueError(f"Неизвестный EfficientNet: {variant}")

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_model_name(path: Path, checkpoint: dict) -> str:
    model_name = checkpoint.get("model_name")
    if model_name == "efficientnet":
        return f"efficientnet_{checkpoint.get('variant')}"
    if model_name:
        return str(model_name)

    name = path.name.lower()
    if "convnext_nano" in name:
        return "convnext_nano"
    if "convnext_tiny" in name:
        return "convnext_tiny"
    if "resnet50" in name:
        return "resnet50"
    if "resnet18" in name:
        return "resnet18"
    if "densenet121" in name:
        return "densenet121"
    if "efficientnet_b0" in name:
        return "efficientnet_b0"
    if "efficientnet_b1" in name:
        return "efficientnet_b1"

    raise ValueError(f"Не удалось понять тип модели: {path}")


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "convnext_nano":
        return build_convnext_nano(num_classes)
    if model_name == "convnext_tiny":
        return build_convnext_tiny(num_classes=num_classes, pretrained=False)
    if model_name == "resnet50":
        return build_resnet50(num_classes)
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=False)
    if model_name == "densenet121":
        return build_densenet121(num_classes=num_classes, pretrained=False)
    if model_name.startswith("efficientnet_"):
        return build_efficientnet(model_name.replace("efficientnet_", ""), num_classes)

    raise ValueError(f"Модель не поддерживается: {model_name}")


def make_val_loader(args: argparse.Namespace, image_size: int):
    _, val_loader = create_dataloaders(
        val_csv_path=args.val_csv,
        val_image_root=args.val_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        use_weighted_sampling=False,
    )
    return val_loader


@torch.inference_mode()
def predict_probs(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs = logits.softmax(dim=1)

        all_probs.append(probs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy().tolist())

    return np.concatenate(all_probs), np.array(all_targets)


def normalize_weights(weights: list[float]) -> list[float]:
    weights_array = np.array(weights, dtype=float)
    if (weights_array < 0).any():
        raise ValueError("Веса не должны быть отрицательными")

    total = weights_array.sum()
    if total <= 0:
        raise ValueError("Сумма весов должна быть больше нуля")

    return (weights_array / total).tolist()


def calculate_metrics(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> dict:
    y_pred = probs.argmax(axis=1)
    label_mapping = load_label_mapping()
    per_class_metrics = calculate_per_class_f1(y_true, y_pred, num_classes)

    eps = 1e-12
    sample_loss = -np.log(np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0))

    per_class_metrics = [
        {
            **row,
            "label": label_mapping.get(int(row["class_id"]), str(row["class_id"])),
            "accuracy": float((y_pred[y_true == int(row["class_id"])] == int(row["class_id"])).mean())
            if (y_true == int(row["class_id"])).any()
            else 0.0,
            "loss": float(sample_loss[y_true == int(row["class_id"])].mean())
            if (y_true == int(row["class_id"])).any()
            else 0.0,
        }
        for row in per_class_metrics
    ]

    return {
        "val_loss": float(sample_loss.mean()),
        "accuracy": float(calculate_accuracy(y_true, y_pred)),
        "macro_f1": float(calculate_macro_f1(y_true, y_pred)),
        "per_class_metrics": per_class_metrics,
    }


def save_predictions(path: Path, y_true: np.ndarray, probs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["row_idx", "target", "pred", "confidence"])
        writer.writeheader()
        for row_idx, (target, pred, conf) in enumerate(zip(y_true, preds, confidence, strict=True)):
            writer.writerow(
                {
                    "row_idx": row_idx,
                    "target": int(target),
                    "pred": int(pred),
                    "confidence": float(conf),
                }
            )


def get_ensemble_weights(model_results: list[dict], args: argparse.Namespace, indexes: tuple[int, ...]) -> tuple[str, list[float]]:
    if args.weights is not None:
        selected_weights = [args.weights[i] for i in indexes]
        return "custom", normalize_weights(selected_weights)

    if args.weighting == "val_f1":
        selected_weights = [
            float(model_results[i]["checkpoint_macro_f1"] or model_results[i]["macro_f1"])
            for i in indexes
        ]
        return "val_f1", normalize_weights(selected_weights)

    return "uniform", normalize_weights([1.0] * len(indexes))


def make_ensemble_probs(all_model_probs: list[np.ndarray], weights: list[float], indexes: tuple[int, ...]) -> np.ndarray:
    ensemble_probs = np.zeros_like(all_model_probs[indexes[0]], dtype=float)
    for weight, model_index in zip(weights, indexes, strict=True):
        ensemble_probs += weight * all_model_probs[model_index]
    return ensemble_probs


def search_subsets(
    model_results: list[dict],
    all_model_probs: list[np.ndarray],
    y_true: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    for size in range(1, len(model_results) + 1):
        for indexes in combinations(range(len(model_results)), size):
            weighting, weights = get_ensemble_weights(model_results, args, indexes)
            probs = make_ensemble_probs(all_model_probs, weights, indexes)
            metrics = calculate_metrics(y_true, probs, num_classes)
            rows.append(
                {
                    "size": size,
                    "models": [model_results[i]["name"] for i in indexes],
                    "weighting": weighting,
                    "weights": weights,
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "val_loss": metrics["val_loss"],
                }
            )

    rows.sort(key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)
    return rows


def log_to_mlflow(args: argparse.Namespace, metrics: dict, artifacts: list[Path]) -> None:
    if not args.log_mlflow:
        return

    if args.mlflow_local:
        os.environ.setdefault("RTC_MLFLOW_LOCAL", "1")

    start_mlflow_run(
        "ensemble",
        metrics["run_name"],
        {
            "model": "ensemble",
            "metric_name": "macro_f1",
            "checkpoint": metrics["checkpoint"],
            "weighting": metrics["hyperparameters"]["weighting"],
            "weights": metrics["hyperparameters"]["weights"],
            "checkpoints": metrics["hyperparameters"]["checkpoints"],
            "batch_size": metrics["hyperparameters"]["batch_size"],
            "num_workers": metrics["hyperparameters"]["num_workers"],
            "num_classes": metrics["hyperparameters"]["num_classes"],
            "num_models": metrics["hyperparameters"]["num_models"],
            "val_csv": metrics["hyperparameters"]["val_csv"],
            "val_images": metrics["hyperparameters"]["val_images"],
        },
    )
    log_mlflow_metrics(
        {
            "best_metric": metrics["best_macro_f1"],
            "macro_f1": metrics["ensemble"]["macro_f1"],
            "accuracy": metrics["ensemble"]["accuracy"],
            "val_loss": metrics["ensemble"]["val_loss"],
            "best_macro_f1": metrics["best_macro_f1"],
            "best_accuracy": metrics["best_accuracy"],
            "best_val_loss": metrics["best_val_loss"],
            "best_epoch": metrics["best_epoch"],
            "num_models": len(metrics["models"]),
        }
    )
    log_mlflow_params({"metrics_json": to_project_relative_path(artifacts[0])})
    log_mlflow_artifacts(artifacts)
    end_mlflow_run()


def save_experiment(report: dict, metrics_path: Path, experiments_path: Path) -> None:
    experiments = load_json(experiments_path, default=[])
    experiments.append(
        {
            "run_id": report["run_id"],
            "model": report["model"],
            "best_epoch": report["best_epoch"],
            "best_macro_f1": report["best_macro_f1"],
            "best_accuracy": report["best_accuracy"],
            "best_val_loss": report["best_val_loss"],
            "stop_reason": report["stop_reason"],
            "checkpoint": report["checkpoint"],
            "weighting": report["hyperparameters"]["weighting"],
            "weights": report["hyperparameters"]["weights"],
            "batch_size": report["hyperparameters"]["batch_size"],
            "num_workers": report["hyperparameters"]["num_workers"],
            "num_classes": report["hyperparameters"]["num_classes"],
            "num_models": report["hyperparameters"]["num_models"],
            "metrics_json": to_project_relative_path(metrics_path),
        }
    )
    save_json(experiments, experiments_path)


def main() -> None:
    args = parse_args()
    if not args.checkpoints:
        raise ValueError("Не найдено ни одного checkpoint-файла")
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        raise ValueError("Количество --weights должно совпадать с количеством --checkpoints")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or f"ensemble_val_{run_id}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    print(f"Устройство: {device}")

    model_results = []
    all_model_probs = []
    y_true_base = None
    num_classes_base = None

    for checkpoint_path in args.checkpoints:
        checkpoint = load_checkpoint(checkpoint_path)
        model_name = get_model_name(checkpoint_path, checkpoint)
        num_classes = int(checkpoint["num_classes"])
        image_size = int(checkpoint.get("image_size", 224))

        if num_classes_base is None:
            num_classes_base = num_classes
        elif num_classes != num_classes_base:
            raise ValueError("У моделей разное количество классов")

        print(
            f"Модель: {model_name}, "
            f"checkpoint-файл: {to_project_relative_path(checkpoint_path)}, "
            f"image_size={image_size}"
        )

        model = build_model(model_name, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)

        loader = make_val_loader(args, image_size)
        probs, y_true = predict_probs(model, loader, device)

        if y_true_base is None:
            y_true_base = y_true
        elif not np.array_equal(y_true_base, y_true):
            raise ValueError("Порядок объектов в val изменился между моделями")

        metrics = calculate_metrics(y_true, probs, num_classes)
        checkpoint_macro_f1 = checkpoint.get("best_macro_f1", checkpoint.get("macro_f1"))

        model_results.append(
            {
                "name": model_name,
                "checkpoint": to_project_relative_path(checkpoint_path),
                "image_size": image_size,
                "checkpoint_macro_f1": None if checkpoint_macro_f1 is None else float(checkpoint_macro_f1),
                **metrics,
            }
        )
        all_model_probs.append(probs)

        print(f"  macro_f1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_indexes = tuple(range(len(model_results)))
    weighting, weights = get_ensemble_weights(model_results, args, all_indexes)
    ensemble_probs = make_ensemble_probs(all_model_probs, weights, all_indexes)

    ensemble_metrics = calculate_metrics(y_true_base, ensemble_probs, num_classes_base)
    subset_results = None
    if args.search_subsets:
        subset_results = search_subsets(model_results, all_model_probs, y_true_base, num_classes_base, args)

    report = {
        "run_id": run_id,
        "model": "ensemble",
        "run_name": run_name,
        "hyperparameters": {
            "weighting": weighting,
            "weights": weights,
            "checkpoints": [to_project_relative_path(path) for path in args.checkpoints],
            "val_csv": to_project_relative_path(args.val_csv),
            "val_images": to_project_relative_path(args.val_images),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_classes": num_classes_base,
            "num_rows": int(len(y_true_base)),
            "num_models": len(model_results),
        },
        "best_epoch": 0,
        "best_macro_f1": ensemble_metrics["macro_f1"],
        "best_accuracy": ensemble_metrics["accuracy"],
        "best_train_loss": None,
        "best_val_loss": ensemble_metrics["val_loss"],
        "best_epoch_metrics": {
            "epoch": 0,
            "train_loss": None,
            "val_loss": ensemble_metrics["val_loss"],
            "accuracy": ensemble_metrics["accuracy"],
            "macro_f1": ensemble_metrics["macro_f1"],
            "per_class_metrics": ensemble_metrics["per_class_metrics"],
        },
        "checkpoint": [to_project_relative_path(path) for path in args.checkpoints],
        "stop_reason": "validation_evaluation",
        "models": model_results,
        "ensemble": ensemble_metrics,
    }
    if subset_results is not None:
        report["subset_search"] = subset_results

    metrics_path = args.output_dir / "ensemble_metrics.json"
    experiments_path = args.output_dir / "ensemble_experiments.json"
    save_json(report, metrics_path)
    save_experiment(report, metrics_path, experiments_path)
    artifacts = [metrics_path, experiments_path]

    if args.save_predictions:
        predictions_path = args.output_dir / f"{run_name}_predictions.csv"
        save_predictions(predictions_path, y_true_base, ensemble_probs)
        artifacts.append(predictions_path)

    log_to_mlflow(args, report, artifacts)

    print("\nРезультаты моделей:")
    for row, weight in zip(model_results, weights, strict=True):
        print(
            f"  {row['name']:<16} вес={weight:.4f} "
            f"macro_f1={row['macro_f1']:.4f} accuracy={row['accuracy']:.4f}"
        )

    print(
        f"\nАнсамбль ({weighting}): "
        f"macro_f1={ensemble_metrics['macro_f1']:.4f}, "
        f"accuracy={ensemble_metrics['accuracy']:.4f}"
    )
    if subset_results is not None:
        print("\nЛучшие подмножества:")
        for row in subset_results[:10]:
            print(
                f"  моделей={row['size']} "
                f"macro_f1={row['macro_f1']:.4f} "
                f"accuracy={row['accuracy']:.4f} "
                f"состав={', '.join(row['models'])}"
            )
    print(f"Отчёт сохранён: {to_project_relative_path(metrics_path)}")


if __name__ == "__main__":
    main()
