"""
CoAtNet per-frame backbone for video classification.

Paper: "CoAtNet: Marrying Convolution and Attention for All Data Sizes"
       Zihang Dai, Hanxiao Liu, Quoc V. Le, Mingxing Tan — NeurIPS 2021
       https://arxiv.org/abs/2106.04803

Architecture: C-C-T-T layout — two MBConv stages followed by two Transformer stages.
Depthwise convolution and relative self-attention are unified through a shared
pre-normalization formulation, giving both the generalization of CNNs and the
large-capacity of Transformers.

This wrapper applies the timm backbone per-frame (B*T images), mean-pools over T,
then classifies. The backbone is exposed as self.backbone so backbone_lr_scale LLRD
works transparently from train.py.

Recommended timm variant: coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k
  ~42 M params, IN-12K pre-trained then fine-tuned on IN-1K (~83-84% top-1).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CoAtNetVideo(nn.Module):
    """CoAtNet (timm) per-frame features + temporal mean-pool classifier.

    Forward:
        (B, T, C, H, W) → CoAtNet backbone per frame → (B, T, feat_dim)
                        → mean over T → Dropout → Linear → (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = True,
        variant: str = "coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k",
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.backbone = timm.create_model(variant, pretrained=pretrained, num_classes=0)
        feat_dim: int = self.backbone.num_features
        self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feats = self.backbone(x.view(B * T, C, H, W))   # (B*T, feat_dim)
        feats = feats.view(B, T, -1).mean(dim=1)         # (B, feat_dim)
        return self.classifier(self.head_drop(feats))
