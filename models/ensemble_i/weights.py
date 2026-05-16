"""Веса ансамбля: per-class из val F1 или из JSON-конфига."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def per_class_f1_from_metrics(metrics_path: Path, num_classes: int) -> list[float]:
    """Извлечь F1 по классам из metrics JSON проекта."""
    data = _load_json(metrics_path)
    pcm = None
    bem = data.get("best_epoch_metrics")
    if isinstance(bem, dict):
        pcm = bem.get("per_class_metrics")
    if pcm is None and isinstance(data.get("best_per_class_f1"), list):
        pcm = data["best_per_class_f1"]
    if not pcm:
        macro = data.get("best_macro_f1") or data.get("macro_f1")
        if macro is not None:
            return [float(macro)] * num_classes
        raise ValueError(f"В {metrics_path} нет per_class_metrics / best_per_class_f1 / macro_f1")

    f1 = [0.0] * num_classes
    for row in pcm:
        cid = int(row["class_id"])
        if 0 <= cid < num_classes:
            f1[cid] = float(row.get("f1", 0.0))
    return f1


def build_per_class_weights(
    member_keys: list[str],
    member_f1: dict[str, list[float]],
    *,
    num_classes: int,
    power: float = 2.0,
    min_weight: float = 1e-6,
) -> np.ndarray:
    """
    Матрица [n_members, num_classes], по каждому классу сумма весов = 1.
    w[m,c] ∝ F1(m,c)^power
    """
    n = len(member_keys)
    w = np.zeros((n, num_classes), dtype=np.float64)
    for c in range(num_classes):
        col = np.array([max(member_f1[k][c], min_weight) for k in member_keys], dtype=np.float64)
        col = np.power(col, power)
        s = col.sum()
        if s <= 0:
            w[:, c] = 1.0 / n
        else:
            w[:, c] = col / s
    return w


def weights_from_config(
    members: list[dict[str, Any]],
    *,
    num_classes: int,
    weight_power: float,
    per_class_weights: dict[str, list[float]] | None,
    root: Path,
) -> tuple[list[str], np.ndarray]:
    keys = [str(m["key"]) for m in members]
    if per_class_weights:
        rows = []
        for k in keys:
            if k not in per_class_weights:
                raise KeyError(f"per_class_weights: нет ключа {k}")
            row = per_class_weights[k]
            if len(row) != num_classes:
                raise ValueError(f"per_class_weights[{k}]: длина {len(row)}, ожидалось {num_classes}")
            rows.append([float(x) for x in row])
        w = np.array(rows, dtype=np.float64)
        w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
        return keys, w

    f1_map: dict[str, list[float]] = {}
    for m in members:
        key = str(m["key"])
        mj = m.get("metrics_json")
        if not mj:
            raise ValueError(f"member {key}: укажите metrics_json или per_class_weights в конфиге")
        path = Path(mj)
        if not path.is_absolute():
            path = (root / path).resolve()
        f1_map[key] = per_class_f1_from_metrics(path, num_classes)

    w = build_per_class_weights(keys, f1_map, num_classes=num_classes, power=weight_power)
    return keys, w


def weights_to_config_dict(keys: list[str], weights: np.ndarray) -> dict[str, list[float]]:
    return {k: [float(x) for x in weights[i].tolist()] for i, k in enumerate(keys)}
