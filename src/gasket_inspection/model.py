from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    convnext_small,
    convnext_tiny,
)


def _build_convnext(architecture: str, pretrained: bool):
    builders = {
        "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
        "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
        "convnext_base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT),
    }
    if architecture not in builders:
        raise ValueError(f"지원하지 않는 ConvNeXt 모델입니다: {architecture}")
    builder, default_weights = builders[architecture]
    return builder(weights=default_weights if pretrained else None)


class SingleImageConvNeXt(nn.Module):
    """상단면 또는 측면 사진 한 장을 독립적으로 분류하는 ConvNeXt 모델."""

    def __init__(self, num_classes: int, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        base = _build_convnext(
            architecture=model_cfg.get("architecture", "convnext_tiny"),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
        feature_dim = int(base.classifier[-1].in_features)
        self.backbone = base.features
        self.pool = base.avgpool
        # 사전학습 classifier의 normalization까지 feature encoder에 포함합니다.
        self.feature_norm = base.classifier[0]
        dropout = float(model_cfg.get("dropout", 0.2))
        self.dropout = nn.Dropout(dropout)
        self.class_head = nn.Linear(feature_dim, num_classes)

        self.severity_head = nn.Linear(feature_dim, 1)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(image)
        feature = self.pool(feature)
        feature = self.feature_norm(feature)
        return torch.flatten(feature, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.dropout(self.encode(image))
        return {
            "class_logits": self.class_head(feature),
            "severity": torch.sigmoid(self.severity_head(feature).squeeze(1)),
        }

    def set_backbone_trainable(self, trainable: bool) -> None:
        for module in (self.backbone, self.feature_norm):
            for parameter in module.parameters():
                parameter.requires_grad = trainable


def build_model(num_classes: int, model_cfg: dict[str, Any]) -> SingleImageConvNeXt:
    return SingleImageConvNeXt(num_classes=num_classes, model_cfg=model_cfg)
