"""
EfficientFormer per-frame backbone for video classification.

Paper: "EfficientFormer: Vision Transformers at MobileNet Speed"
       Yanyu Li, Geng Yuan, Yang Wen, Evan Shelhamer et al. — NeurIPS 2022
       https://arxiv.org/abs/2206.01191

Architecture: latency-driven pure transformer — 4-stage MetaFormer with
dimension-consistent design. Unlike ViT, it keeps spatial dimensions intact
through most stages (like a CNN) to exploit hardware-friendly pooling, then
applies self-attention only in the final stage. Achieves sub-10 ms latency on
mobile phones while matching EfficientNet accuracy.

This wrapper applies the timm backbone per-frame (B*T images), mean-pools over T,
then classifies. The backbone is exposed as self.backbone so backbone_lr_scale LLRD
works transparently from train.py.

Recommended timm variant: efficientformer_l3.snap_dist_in1k
  ~31 M params, official Snap Research weights, 82.4% IN-1K top-1.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class EfficientFormerVideo(nn.Module):
    """EfficientFormer (timm) per-frame features + temporal mean-pool classifier.

    Forward:
        (B, T, C, H, W) → EfficientFormer backbone per frame → (B, T, feat_dim)
                        → mean over T → Dropout → Linear → (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = True,
        variant: str = "efficientformer_l3.snap_dist_in1k",
        head_dropout: float = 0.3,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.backbone = timm.create_model(
            variant, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
        )
        feat_dim: int = self.backbone.num_features
        self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feats = self.backbone(x.view(B * T, C, H, W))   # (B*T, feat_dim)
        feats = feats.view(B, T, -1).mean(dim=1)         # (B, feat_dim)
        return self.classifier(self.head_drop(feats))
