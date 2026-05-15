from __future__ import annotations

import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def build_convnext_tiny(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


__all__ = ["build_convnext_tiny"]
