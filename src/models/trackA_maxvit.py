"""
MaxViT-T per-frame backbone for video classification.

All models follow the same (B, T, C, H, W) → (B, num_classes) interface.

Backbone: torchvision maxvit_t, optionally ImageNet-pretrained.
  - Stem + 4 MaxVit stages → (B, 512, 7, 7) spatial features
  - Classifier head up to Tanh gives a 512-dim representation per frame.
The backbone is exposed as self.backbone so LLRD (backbone_lr_scale) works.

Temporal aggregation: mean pool over the T-frame sequence, then dropout +
linear classifier.

When combined with frame interpolation (dataset.interpolate_frames=true),
4 real frames are expanded to 8 by inserting linear blends between adjacent
pairs. This doubles TSM-style temporal positions at zero disk cost.

Input images must be 224×224 (MaxViT window partitioning is hardcoded to
that resolution in torchvision's maxvit_t).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MaxVit_T_Weights, maxvit_t

MAXVIT_DIM = 512  # representation_size for maxvit_t


class _MaxViTBackbone(nn.Module):
    """MaxViT-T encoder without the final classification linear.

    Returns a 512-dim vector per image (after AdaptiveAvgPool → Flatten →
    LayerNorm → Linear(512,512) → Tanh).
    """

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None
        m = maxvit_t(weights=weights)
        self.stem = m.stem
        self.blocks = m.blocks          # ModuleList of 4 MaxVitBlock stages
        self.pool = m.classifier[0]     # AdaptiveAvgPool2d(1)
        self.flat = m.classifier[1]     # Flatten
        self.norm = m.classifier[2]     # LayerNorm(512)
        self.proj = m.classifier[3]     # Linear(512, 512)
        self.act  = m.classifier[4]     # Tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)                # hierarchical window+grid attention
        x = self.flat(self.pool(x))     # (B, 512)
        return self.act(self.proj(self.norm(x)))  # (B, 512)


class MaxViTVideo(nn.Module):
    """MaxViT-T per-frame features + temporal mean pooling classifier.

    Forward:
        (B, T, C, H, W) → MaxViT-T per frame → (B, T, 512)
                        → mean over T → Dropout → Linear → (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = True,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = _MaxViTBackbone(pretrained=pretrained)
        self.num_frames = num_frames
        self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(MAXVIT_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feats = self.backbone(x.view(B * T, C, H, W))   # (B*T, 512)
        feats = feats.view(B, T, MAXVIT_DIM).mean(dim=1) # (B, 512)
        return self.classifier(self.head_drop(feats))
