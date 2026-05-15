from __future__ import annotations

from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


def build_densenet121(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


__all__ = ["build_densenet121"]
