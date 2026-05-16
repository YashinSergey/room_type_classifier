"""Комбинирование вероятностей членов ансамбля."""

from __future__ import annotations

import torch


def combine_member_probs(
    member_probs: list[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    member_probs: список [B, C] softmax по каждой модели (одинаковый C).
    weights: [M, C] — веса по классам, сумма по M для каждого c равна 1.
    Возвращает [B, C] — взвешенная сумма вероятностей.
    """
    if not member_probs:
        raise ValueError("member_probs пуст")
    b, c = member_probs[0].shape
    m = len(member_probs)
    if weights.shape != (m, c):
        raise ValueError(f"weights shape {tuple(weights.shape)}, ожидалось ({m}, {c})")

    stacked = torch.stack(member_probs, dim=0)
    w = weights.unsqueeze(1)
    return (stacked * w).sum(dim=0)


def softmax_std(prob_row: torch.Tensor) -> float:
    return float(torch.std(prob_row, unbiased=False).item())
