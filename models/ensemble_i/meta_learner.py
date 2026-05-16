"""Мета-классификатор (stacking): concat(probs) → Linear → logits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class StackingMetaLearner(nn.Module):
    """
    Вход: конкатенация softmax [M моделей × C классов].
    Один полносвязный слой: (M·C) → C.
    """

    def __init__(self, n_members: int, n_classes: int) -> None:
        super().__init__()
        self.n_members = n_members
        self.n_classes = n_classes
        self.fc = nn.Linear(n_members * n_classes, n_classes)

    def forward(self, member_probs: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(member_probs, dim=1)
        return self.fc(x)

    def predict_probs(self, member_probs: list[torch.Tensor]) -> torch.Tensor:
        return self.forward(member_probs).softmax(dim=1)


def save_meta(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_meta(path: Path, device: torch.device) -> tuple[StackingMetaLearner, dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False)
    n_members = int(state["n_members"])
    n_classes = int(state["n_classes"])
    model = StackingMetaLearner(n_members, n_classes)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state
