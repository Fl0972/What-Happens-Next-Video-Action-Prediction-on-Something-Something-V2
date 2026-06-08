"""
R(2+1)D-18: a 3D CNN whose 3x3x3 spatio-temporal convs are factored into a
2D spatial conv (1, k, k) followed by a 1D temporal conv (k, 1, 1). The extra
nonlinearity between the two halves makes the model more expressive than a
plain 3D conv with the same parameter count and easier to optimize from
scratch on small video datasets — a good fit for the closed track.

Reference: Tran et al., "A Closer Look at Spatiotemporal Convolutions for
           Action Recognition", CVPR 2018 (https://arxiv.org/abs/1711.11248).

Forward:
    Input:  (B, T, C, H, W)        — same convention as the rest of the codebase
    Permute -> (B, C, T, H, W)     — what torchvision's video models expect
    Backbone: r2plus1d_18 (random init for the closed track)
    Pool: global average -> (B, 512)
    Optional dropout, then linear -> (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18


class R2Plus1D(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # The closed-track rule forbids pretrained *weights*, not the architecture
        # itself. ``pretrained=False`` -> torchvision instantiates with random init.
        weights = None
        if pretrained:
            from torchvision.models.video import R2Plus1D_18_Weights
            weights = R2Plus1D_18_Weights.KINETICS400_V1
        net = r2plus1d_18(weights=weights)

        feature_dim = net.fc.in_features  # 512
        net.fc = nn.Identity()

        self.backbone = net
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        # torchvision video models want channel-first time-second: (B, C, T, H, W)
        video = video_batch.permute(0, 2, 1, 3, 4).contiguous()
        features = self.backbone(video)
        features = self.dropout(features)
        return self.classifier(features)
