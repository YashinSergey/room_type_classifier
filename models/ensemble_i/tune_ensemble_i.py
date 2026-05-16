"""
Обучение/оценка ensemble_i.
- stacking: meta на train, лучший чекпо val F1 (как базовые модели)
- per_class_soft_vote: legacy
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.ensemble_i.config import DEFAULT_CONFIG, load_config, resolve_path, save_config
from models.ensemble_i.members import load_members
from models.ensemble_i.meta_learner import save_meta
from models.ensemble_i.train_meta import train_stacking_meta
from models.ensemble_i.weights import weights_from_config, weights_to_config_dict
from src.dataloaders import create_dataloaders
from src.device import get_default_device
from src.labels import load_label_mapping
from src.mlflow_utils import end_mlflow_run, log_mlflow_artifacts, log_mlflow_metrics, log_mlflow_params, start_mlflow_run
from src.training_helpers import set_seed, to_project_relative_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение stacking / оценка ensemble_i")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--refresh-weights-from-metrics",
        action="store_true",
        help="(soft_vote) пересчитать per_class_weights из metrics_json",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def append_experiment(metrics_dir: Path, model_name: str, record: dict[str, Any]) -> Path:
    exp_path = metrics_dir / f"{model_name}_experiments.json"
    history: list[dict[str, Any]] = []
    if exp_path.is_file():
        history = json.loads(exp_path.read_text(encoding="utf-8"))
    history.append(record)
    exp_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return exp_path


def _loader_pair(train_csv: Path, val_csv: Path, train_images: Path, val_images: Path, **kwargs):
    return create_dataloaders(
        train_csv_path=str(train_csv),
        val_csv_path=str(val_csv),
        train_image_root=str(train_images),
        val_image_root=str(val_images),
        **kwargs,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    model_name = str(cfg.get("model_name", "ensemble_i"))
    num_classes = int(cfg.get("num_classes", 19))
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 2))
    combination = str(cfg.get("combination", "stacking"))
    weight_power = float(cfg.get("weight_power", 2.0))
    meta_cfg = cfg.get("meta_learner") or {}

    data = cfg.get("data") or {}
    train_csv = resolve_path(ROOT_DIR, data.get("train_csv", "data/processed/train_df.csv"))
    train_images = resolve_path(ROOT_DIR, data.get("train_images", "data/raw/train_images"))
    val_csv = resolve_path(ROOT_DIR, data.get("val_csv", "data/processed/val_df.csv"))
    val_images = resolve_path(ROOT_DIR, data.get("val_images", "data/raw/val_images"))
    metrics_dir = resolve_path(ROOT_DIR, cfg.get("metrics_dir", "reports/metrics/ensemble_i"))
    output_dir = resolve_path(ROOT_DIR, cfg.get("output_dir", "outputs/models/ensemble_i"))
    assert train_csv and train_images and val_csv and val_images and metrics_dir and output_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    print(f"device={device} combination={combination}", flush=True)

    members = load_members(cfg["members"], ROOT_DIR, device)
    keys = [m.key for m in members]
    print("members:", keys, flush=True)

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        use_weighted_sampling=False,
        seed=args.seed,
    )
    train_loader, val_loader = _loader_pair(train_csv, val_csv, train_images, val_images, **loader_kw)
    print(f"train n={len(train_loader.dataset)} val n={len(val_loader.dataset)}", flush=True)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_mlflow_run(
        model_name,
        run_name=f"{model_name}_{run_id}",
        params={
            "config": to_project_relative_path(args.config.resolve()),
            "combination": combination,
            "members": keys,
            "num_classes": num_classes,
        },
    )

    meta_path = output_dir / "stacking_meta.pt"
    macro_f1 = 0.0
    accuracy = 0.0
    best_epoch = 0
    per_class: list[dict[str, Any]] = []

    if combination == "stacking":
        print("stacking: backprop on train, best checkpoint by val F1 (like base models)", flush=True)
        meta_learner, meta_info = train_stacking_meta(
            members,
            train_loader,
            val_loader,
            device,
            epochs=int(meta_cfg.get("epochs", 200)),
            lr=float(meta_cfg.get("lr", 0.05)),
            weight_decay=float(meta_cfg.get("weight_decay", 1e-3)),
            patience=int(meta_cfg.get("patience", 25)),
            train_batch_size=int(meta_cfg.get("train_batch_size", 64)),
        )
        save_meta(
            {
                "model_state_dict": meta_learner.state_dict(),
                "n_members": len(members),
                "n_classes": num_classes,
                "member_keys": keys,
                **meta_info,
            },
            meta_path,
        )
        cfg.setdefault("meta_learner", {})["checkpoint"] = to_project_relative_path(meta_path)
        save_config(cfg, args.config.resolve())

        label_map = load_label_mapping()
        per_class = list(meta_info.get("per_class_metrics") or [])
        for item in per_class:
            item["label"] = label_map.get(item["class_id"], str(item["class_id"]))

        macro_f1 = float(meta_info["best_macro_f1"])
        accuracy = float(meta_info["best_accuracy"])
        best_epoch = int(meta_info["best_epoch"])

        metrics = {
            "run_id": run_id,
            "model": model_name,
            "split": "val",
            "hyperparameters": {
                "config": str(args.config.resolve()),
                "combination": combination,
                "members": keys,
                "member_checkpoints": [to_project_relative_path(m.checkpoint) for m in members],
                "meta_learner": meta_cfg,
                "meta_checkpoint": to_project_relative_path(meta_path),
                "protocol": "train_backprop_val_checkpoint",
            },
            "best_epoch": best_epoch,
            "best_macro_f1": macro_f1,
            "best_accuracy": accuracy,
            "best_epoch_metrics": {
                "epoch": best_epoch,
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "per_class_metrics": per_class,
                "num_samples": meta_info.get("val_samples"),
            },
            "checkpoint": to_project_relative_path(meta_path),
        }
    else:
        from models.ensemble_i.predict import predict_batch
        from src.metrics import calculate_accuracy, calculate_macro_f1, calculate_per_class_f1
        from tqdm import tqdm

        per_class_weights = None if args.refresh_weights_from_metrics else cfg.get("per_class_weights")
        _, w_np = weights_from_config(
            cfg["members"],
            num_classes=num_classes,
            weight_power=weight_power,
            per_class_weights=per_class_weights,
            root=ROOT_DIR,
        )
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)
        w_dict = weights_to_config_dict(keys, w_np)
        cfg["per_class_weights"] = w_dict
        weights_path = output_dir / f"{model_name}_weights.json"
        weights_path.write_text(json.dumps(w_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        save_config(cfg, args.config.resolve())

        amb = cfg.get("ambiguous") or {}
        y_true: list[int] = []
        y_pred: list[int] = []
        ambiguous_count = 0
        for images, targets in tqdm(val_loader, desc="ensemble val"):
            images = images.to(device)
            batch_rows = predict_batch(
                images,
                members,
                weights,
                num_classes=num_classes,
                combination=combination,
                ambiguous_enabled=bool(amb.get("enabled", True)),
                ambiguous_class_id=int(amb.get("class_id", 18)),
                ambiguous_std_threshold=float(amb.get("std_threshold", 0.03)),
            )
            for j, row in enumerate(batch_rows):
                y_true.append(int(targets[j].item()))
                y_pred.append(int(row["pred"]))
                if row["pred_source"] == "ambiguous_std":
                    ambiguous_count += 1

        macro_f1 = float(calculate_macro_f1(y_true, y_pred))
        accuracy = float(calculate_accuracy(y_true, y_pred))
        per_class = calculate_per_class_f1(y_true, y_pred, num_classes)
        label_map = load_label_mapping()
        for item in per_class:
            item["label"] = label_map.get(item["class_id"], str(item["class_id"]))

        metrics = {
            "run_id": run_id,
            "model": model_name,
            "split": "val",
            "hyperparameters": {
                "config": str(args.config.resolve()),
                "combination": combination,
                "members": keys,
                "per_class_weights": w_dict,
            },
            "best_macro_f1": macro_f1,
            "best_accuracy": accuracy,
            "best_epoch_metrics": {
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "per_class_metrics": per_class,
                "ambiguous_assigned_count": ambiguous_count,
                "num_samples": len(y_true),
            },
            "checkpoint": to_project_relative_path(weights_path),
        }

    metrics_path = metrics_dir / f"{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    exp_path = append_experiment(metrics_dir, model_name, metrics)

    mlflow_metrics: dict[str, float] = {
        "best_macro_f1": macro_f1,
        "macro_f1": macro_f1,
        "val_macro_f1": macro_f1,
        "best_accuracy": accuracy,
        "accuracy": accuracy,
        "val_accuracy": accuracy,
    }
    if combination == "stacking":
        mlflow_metrics["best_epoch"] = float(best_epoch)
    log_mlflow_metrics(mlflow_metrics)
    artifacts = [metrics_path, exp_path, args.config.resolve()]
    if meta_path.is_file():
        artifacts.append(meta_path)
    log_mlflow_artifacts(artifacts)
    if combination == "stacking":
        log_mlflow_params({"best_epoch": str(best_epoch)})
    end_mlflow_run()

    print(f"best_epoch={best_epoch} val macro_f1={macro_f1:.4f} accuracy={accuracy:.4f}", flush=True)
    print(f"metrics -> {metrics_path}", flush=True)
    print(f"stacking meta -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
