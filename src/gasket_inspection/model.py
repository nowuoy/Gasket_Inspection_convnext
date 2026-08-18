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


class DualViewConvNeXt(nn.Module):
    """하나의 ConvNeXt encoder를 전면/측면에 공유하는 late-fusion 모델."""

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
        hidden_dim = int(model_cfg.get("fusion_hidden_dim", 512))
        dropout = float(model_cfg.get("dropout", 0.2))
        self.fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(hidden_dim, num_classes)

        self.quality_head_enabled = bool(model_cfg.get("quality_head_enabled", False))
        self.severity_head_enabled = bool(model_cfg.get("severity_head_enabled", False))
        self.quality_head = nn.Linear(hidden_dim, 1) if self.quality_head_enabled else None
        self.severity_head = nn.Linear(hidden_dim, 1) if self.severity_head_enabled else None

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(image)
        feature = self.pool(feature)
        feature = self.feature_norm(feature)
        return torch.flatten(feature, 1)

    def forward(self, front: torch.Tensor, side: torch.Tensor) -> dict[str, torch.Tensor]:
        front_feature = self.encode(front)
        side_feature = self.encode(side)
        fused = self.fusion(torch.cat((front_feature, side_feature), dim=1))
        output = {"class_logits": self.class_head(fused)}
        if self.quality_head is not None:
            output["quality_logit"] = self.quality_head(fused).squeeze(1)
        if self.severity_head is not None:
            output["severity"] = torch.sigmoid(self.severity_head(fused).squeeze(1))
        return output

    def set_backbone_trainable(self, trainable: bool) -> None:
        for module in (self.backbone, self.feature_norm):
            for parameter in module.parameters():
                parameter.requires_grad = trainable


def build_model(num_classes: int, model_cfg: dict[str, Any]) -> DualViewConvNeXt:
    return DualViewConvNeXt(num_classes=num_classes, model_cfg=model_cfg)

