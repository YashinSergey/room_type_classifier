"""
Инференс ensemble_i на test (submission) или val (предсказания + accuracy / macro F1).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.ensemble_i.config import DEFAULT_CONFIG, load_config, resolve_path
from models.ensemble_i.members import load_members
from models.ensemble_i.predict import load_meta_from_config, predict_batch
from models.ensemble_i.weights import weights_from_config
from src.dataloaders import create_dataloaders, create_test_dataloader
from src.device import get_default_device
from src.labels import load_label_mapping
from src.metrics import calculate_accuracy, calculate_macro_f1, calculate_per_class_f1
from src.mlflow_utils import end_mlflow_run, log_mlflow_artifacts, log_mlflow_metrics, log_mlflow_params, start_mlflow_run
from src.training_helpers import set_seed, to_project_relative_path


def _take_batch_field(batch, j):
    if isinstance(batch, torch.Tensor):
        return batch[j].item()
    if isinstance(batch, (list, tuple)):
        return batch[j]
    return batch[j]


def _append_experiment(metrics_dir: Path, model_name: str, record: dict[str, Any]) -> Path:
    exp_path = metrics_dir / f"{model_name}_experiments.json"
    history: list[dict[str, Any]] = []
    if exp_path.is_file():
        history = json.loads(exp_path.read_text(encoding="utf-8"))
    history.append(record)
    exp_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return exp_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Инференс ensemble_i")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="test: submission; val: полный CSV с метками если есть",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Полный predictions CSV (по умолчанию outputs/ensemble_i_<split>_predictions.csv)",
    )
    p.add_argument(
        "--submission-output",
        type=Path,
        default=None,
        help="Файл submission: image_id_ext,Predicted (по умолчанию outputs/ensemble_i_submission.csv для test)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    model_name = str(cfg.get("model_name", "ensemble_i"))
    num_classes = int(cfg.get("num_classes", 19))
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 2))
    weight_power = float(cfg.get("weight_power", 2.0))

    data = cfg.get("data") or {}
    metrics_dir = resolve_path(ROOT_DIR, cfg.get("metrics_dir", "reports/metrics/ensemble_i"))
    output_dir = resolve_path(ROOT_DIR, cfg.get("output_dir", "outputs/models/ensemble_i"))
    assert metrics_dir and output_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "test":
        csv_path = resolve_path(ROOT_DIR, data.get("test_csv", "data/processed/test_df.csv"))
        img_root = resolve_path(ROOT_DIR, data.get("test_images", "data/raw/test_images"))
    else:
        csv_path = resolve_path(ROOT_DIR, data.get("val_csv", "data/processed/val_df.csv"))
        img_root = resolve_path(ROOT_DIR, data.get("val_images", "data/raw/val_images"))
    assert csv_path and img_root

    default_out = ROOT_DIR / "outputs" / f"{model_name}_{args.split}_predictions.csv"
    out_path = args.output or default_out
    sub_path = args.submission_output
    if sub_path is None and args.split == "test":
        sub_path = ROOT_DIR / "outputs" / f"{model_name}_submission.csv"

    device = get_default_device()
    combination = str(cfg.get("combination", "stacking"))
    print(f"device={device} split={args.split} combination={combination}", flush=True)

    members = load_members(cfg["members"], ROOT_DIR, device)
    keys = [m.key for m in members]
    meta_learner = load_meta_from_config(cfg, ROOT_DIR, device)
    weights = None
    if combination != "stacking":
        _, w_np = weights_from_config(
            cfg["members"],
            num_classes=num_classes,
            weight_power=weight_power,
            per_class_weights=cfg.get("per_class_weights"),
            root=ROOT_DIR,
        )
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)

    amb = cfg.get("ambiguous") or {}
    amb_enabled = bool(amb.get("enabled", True))
    amb_class = int(amb.get("class_id", 18))
    amb_thr = float(amb.get("std_threshold", 0.03))

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_mlflow_run(
        model_name,
        run_name=f"{model_name}_infer_{args.split}_{run_id}",
        params={
            "split": args.split,
            "config": to_project_relative_path(args.config.resolve()),
            "members": keys,
        },
    )

    label_map = load_label_mapping()
    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    ambiguous_count = 0

    if args.split == "test":
        loader = create_test_dataloader(
            test_csv_path=str(csv_path),
            test_image_root=str(img_root),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
        )
        for images, image_ids, item_ids in tqdm(loader, desc="ensemble test"):
            images = images.to(device)
            batch_rows = predict_batch(
                images,
                members,
                weights,
                num_classes=num_classes,
                combination=combination,
                meta_learner=meta_learner,
                ambiguous_enabled=amb_enabled,
                ambiguous_class_id=amb_class,
                ambiguous_std_threshold=amb_thr,
            )
            for j, br in enumerate(batch_rows):
                pred = int(br["pred"])
                rows.append(
                    {
                        "image_id_ext": _take_batch_field(image_ids, j),
                        "item_id": _take_batch_field(item_ids, j),
                        "Predicted": pred,
                        "pred": pred,
                        "pred_source": br["pred_source"],
                        "prob_std": br["prob_std"],
                        "confidence": br["confidence"],
                        "label": label_map.get(pred, str(pred)),
                        **{k: v for k, v in br.items() if k.startswith(("convnext", "densenet", "efficientnet", "resnet"))},
                    }
                )
    else:
        _, val_loader = create_dataloaders(
            train_csv_path=str(csv_path),
            val_csv_path=str(csv_path),
            train_image_root=str(img_root),
            val_image_root=str(img_root),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            use_weighted_sampling=False,
            seed=args.seed,
        )
        for images, targets in tqdm(val_loader, desc="ensemble val infer"):
            images = images.to(device)
            batch_rows = predict_batch(
                images,
                members,
                weights,
                num_classes=num_classes,
                combination=combination,
                meta_learner=meta_learner,
                ambiguous_enabled=amb_enabled,
                ambiguous_class_id=amb_class,
                ambiguous_std_threshold=amb_thr,
            )
            for j, br in enumerate(batch_rows):
                pred = int(br["pred"])
                true_id = int(targets[j].item())
                y_true.append(true_id)
                y_pred.append(pred)
                if br["pred_source"] == "ambiguous_std":
                    ambiguous_count += 1
                rows.append(
                    {
                        "pred": pred,
                        "Predicted": pred,
                        "true": true_id,
                        "true_label": label_map.get(true_id, str(true_id)),
                        "pred_source": br["pred_source"],
                        "prob_std": br["prob_std"],
                        "confidence": br["confidence"],
                        "label": label_map.get(pred, str(pred)),
                        **{k: v for k, v in br.items() if k.startswith(("convnext", "densenet", "efficientnet", "resnet"))},
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    artifacts = [out_path]
    if sub_path is not None:
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        sub_df = df[["image_id_ext", "Predicted"]] if "image_id_ext" in df.columns else df
        if "image_id_ext" in df.columns:
            sub_df.to_csv(sub_path, index=False)
            artifacts.append(sub_path)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "model_name": model_name,
        "split": args.split,
        "predictions_csv": str(out_path.resolve()),
        "submission_csv": str(sub_path.resolve()) if sub_path else None,
        "num_rows": len(rows),
        "members": keys,
        "ambiguous_enabled": amb_enabled,
    }
    mlflow_metrics: dict[str, float] = {"num_predictions": float(len(rows))}

    if args.split == "val" and y_true:
        macro_f1 = calculate_macro_f1(y_true, y_pred)
        accuracy = calculate_accuracy(y_true, y_pred)
        per_class = calculate_per_class_f1(y_true, y_pred, num_classes)
        for item in per_class:
            item["label"] = label_map.get(item["class_id"], str(item["class_id"]))

        metrics_record: dict[str, Any] = {
            "run_id": run_id,
            "model": model_name,
            "split": "val",
            "source": "infer",
            "hyperparameters": {
                "config": str(args.config.resolve()),
                "combination": cfg.get("combination"),
                "weight_power": weight_power,
                "members": keys,
                "member_checkpoints": [to_project_relative_path(m.checkpoint) for m in members],
                "per_class_weights": cfg.get("per_class_weights"),
            },
            "best_macro_f1": float(macro_f1),
            "best_accuracy": float(accuracy),
            "best_epoch_metrics": {
                "macro_f1": float(macro_f1),
                "accuracy": float(accuracy),
                "per_class_metrics": per_class,
                "ambiguous_assigned_count": ambiguous_count,
                "num_samples": len(y_true),
            },
            "predictions_csv": to_project_relative_path(out_path),
        }
        metrics_path = metrics_dir / f"{model_name}_metrics.json"
        metrics_path.write_text(json.dumps(metrics_record, indent=2, ensure_ascii=False), encoding="utf-8")
        exp_path = _append_experiment(metrics_dir, model_name, metrics_record)
        artifacts.extend([metrics_path, exp_path])

        summary["macro_f1"] = float(macro_f1)
        summary["accuracy"] = float(accuracy)
        summary["ambiguous_assigned_count"] = ambiguous_count
        summary["metrics_json"] = str(metrics_path.resolve())

        macro_f1_f = float(macro_f1)
        accuracy_f = float(accuracy)
        mlflow_metrics.update(
            {
                "best_macro_f1": macro_f1_f,
                "macro_f1": macro_f1_f,
                "val_macro_f1": macro_f1_f,
                "best_accuracy": accuracy_f,
                "accuracy": accuracy_f,
                "val_accuracy": accuracy_f,
                "ambiguous_count": float(ambiguous_count),
            }
        )
        log_mlflow_params(
            {
                "metric_name": "macro_f1",
                "best_epoch": 0,
                "metrics_json": to_project_relative_path(metrics_path),
                "predictions_csv": to_project_relative_path(out_path),
            }
        )

        print(f"val macro_f1={macro_f1:.4f} accuracy={accuracy:.4f}", flush=True)
        print(f"ambiguous_std rows: {ambiguous_count}/{len(y_true)}", flush=True)
        print(f"metrics -> {metrics_path}", flush=True)

    summary_path = metrics_dir / f"{model_name}_inference_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    artifacts.append(summary_path)

    log_mlflow_metrics(mlflow_metrics)
    log_mlflow_artifacts(artifacts)
    end_mlflow_run()

    print(f"Saved {len(rows)} rows -> {out_path}", flush=True)
    if sub_path:
        print(f"Submission -> {sub_path}", flush=True)
    print(f"Report -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
