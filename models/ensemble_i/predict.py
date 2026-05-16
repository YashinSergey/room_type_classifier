"""Инференс ансамбля: soft vote или stacking meta-learner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.ensemble_i.ensemble import combine_member_probs, softmax_std
from models.ensemble_i.members import EnsembleMember
from models.ensemble_i.meta_learner import StackingMetaLearner, load_meta


def model_index_to_original(pred_idx: int, excluded_original_class_id: int | None) -> int:
    if excluded_original_class_id is None:
        return pred_idx
    return pred_idx if pred_idx < excluded_original_class_id else pred_idx + 1


@torch.inference_mode()
def forward_members(
    images: torch.Tensor,
    members: list[EnsembleMember],
) -> list[torch.Tensor]:
    probs: list[torch.Tensor] = []
    for m in members:
        logits = m.model(images)
        probs.append(logits.softmax(dim=1))
    return probs


@torch.inference_mode()
def predict_batch(
    images: torch.Tensor,
    members: list[EnsembleMember],
    weights: torch.Tensor | None,
    *,
    num_classes: int,
    combination: str = "per_class_soft_vote",
    meta_learner: StackingMetaLearner | None = None,
    ambiguous_enabled: bool,
    ambiguous_class_id: int,
    ambiguous_std_threshold: float,
    excluded_original_class_id: int | None = None,
) -> list[dict[str, Any]]:
    member_probs = forward_members(images, members)

    if combination == "stacking":
        if meta_learner is None:
            raise ValueError("combination=stacking требует загруженный meta_learner")
        ens = meta_learner.predict_probs(member_probs)
        pred_source_default = "stacking"
    else:
        if weights is None:
            raise ValueError("per_class_soft_vote требует weights")
        ens = combine_member_probs(member_probs, weights)
        pred_source_default = "soft_vote"

    rows: list[dict[str, Any]] = []
    for j in range(ens.shape[0]):
        prob = ens[j]
        std_p = softmax_std(prob)
        idx0 = int(prob.argmax().item())
        pred_argmax = model_index_to_original(idx0, excluded_original_class_id)
        if ambiguous_enabled and std_p < ambiguous_std_threshold:
            pred = ambiguous_class_id
            pred_source = "ambiguous_std"
        else:
            pred = pred_argmax
            pred_source = pred_source_default

        row: dict[str, Any] = {
            "pred": pred,
            "pred_argmax_original": pred_argmax,
            "pred_source": pred_source,
            "prob_std": std_p,
            "ensemble_class_index": idx0,
            "confidence": float(prob[idx0].item()),
        }
        for mi, m in enumerate(members):
            p_m = member_probs[mi][j]
            row[f"{m.key}_pred"] = int(p_m.argmax().item())
            row[f"{m.key}_conf"] = float(p_m.max().item())
        rows.append(row)
    return rows


def load_meta_from_config(
    cfg: dict[str, Any],
    root: Path,
    device: torch.device,
) -> StackingMetaLearner | None:
    if cfg.get("combination") != "stacking":
        return None
    meta_cfg = cfg.get("meta_learner") or {}
    rel = meta_cfg.get("checkpoint", "outputs/models/ensemble_i/stacking_meta.pt")
    path = Path(rel)
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет stacking meta-learner: {path}. Сначала: just tune-ensemble-i"
        )
    model, _ = load_meta(path, device)
    return model
