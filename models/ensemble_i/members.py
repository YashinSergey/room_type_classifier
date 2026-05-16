"""Загрузка базовых моделей ансамбля из чекпоинтов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

from models.convnext_tiny.model import build_convnext_tiny
from models.densenet121.densenet121 import build_densenet121
from models.efficientNet.train_efficientnet import build_model as build_efficientnet
from models.resnet18.resnet18 import build_resnet18
from models.resnet50.resnet50 import build_model as build_resnet50


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class EnsembleMember:
    key: str
    model: nn.Module
    num_classes: int
    image_size: int
    checkpoint: Path
    excluded_original_class_id: int | None = None
    extra: dict[str, Any] | None = None


def _build_convnext_nano(num_classes: int, pretrained: bool = False) -> nn.Module:
    return timm.create_model(
        "convnext_nano",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.5,
        drop_path_rate=0.3,
    )


def _build_from_key(key: str, num_classes: int, ckpt: dict[str, Any]) -> nn.Module:
    if key == "convnext_tiny":
        return build_convnext_tiny(num_classes=num_classes, pretrained=False)
    if key == "convnext_nano":
        return _build_convnext_nano(num_classes=num_classes, pretrained=False)
    if key == "resnet50":
        return build_resnet50(num_classes=num_classes)
    if key == "densenet121":
        return build_densenet121(num_classes=num_classes, pretrained=False)
    if key == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=False)
    if key.startswith("efficientnet"):
        variant = str(ckpt.get("variant") or key.replace("efficientnet_", "") or "b1")
        return build_efficientnet(variant=variant, num_classes=num_classes)
    raise ValueError(f"Неизвестный member key: {key}")


def load_member(key: str, checkpoint: Path, device: torch.device) -> EnsembleMember:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"[{key}] checkpoint не найден: {checkpoint}")

    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"[{key}] в checkpoint нет model_state_dict: {checkpoint}")

    num_classes = int(ckpt.get("num_classes") or ckpt.get("extra", {}).get("num_classes") or 19)
    image_size = int(ckpt.get("image_size") or ckpt.get("extra", {}).get("image_size") or 224)
    excl = ckpt.get("excluded_original_class_id")
    excluded = int(excl) if excl is not None else None

    model = _build_from_key(key, num_classes, ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return EnsembleMember(
        key=key,
        model=model,
        num_classes=num_classes,
        image_size=image_size,
        checkpoint=checkpoint.resolve(),
        excluded_original_class_id=excluded,
        extra={k: v for k, v in ckpt.items() if k != "model_state_dict"},
    )


def load_members(specs: list[dict[str, Any]], root: Path, device: torch.device) -> list[EnsembleMember]:
    out: list[EnsembleMember] = []
    for spec in specs:
        key = str(spec["key"])
        ckpt = spec.get("checkpoint")
        if not ckpt:
            raise ValueError(f"member {key}: нет checkpoint")
        path = Path(ckpt)
        if not path.is_absolute():
            path = (root / path).resolve()
        out.append(load_member(key, path, device))
    return out
