"""
TSM-ResNet50 + Bidirectional GRU head.

TSM provides local temporal mixing inside every residual block (zero extra params).
BiGRU provides global ordered temporal reasoning over the frame sequence.

Motivation for BiGRU vs mean-pool:
  SSv2 hardest confusions are *temporal direction* (folding ↔ unfolding, real ↔ pretend).
  BiGRU reads the 4-frame sequence in both directions, directly targeting these errors.
  LSTM adds a cell state gate that is redundant for only 4 frames and adds overfit risk.

Forward:
    Input:       (B, T, C, H, W)
    Backbone:    ResNet50 with TSM → per-frame features (B, T, 2048)
    FeatDrop:    applied per feature vector before GRU
    BiGRU:       (B, T, 2048) → (B, T, 2*hidden)
    Pool:        mean over T  → (B, 2*hidden)
    HeadDrop:    applied before classifier
    Classifier:  (B, 2*hidden) → (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.tsm_resnet import _inject_tsm


class TSMBiGRU(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = False,
        backbone: str = "resnet50",
        fold_div: int = 8,
        drop_path_rate: float = 0.2,
        gru_hidden: int = 256,
        gru_layers: int = 1,
        feat_dropout: float = 0.3,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet50(weights=weights)
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}")

        feature_dim = net.fc.in_features
        net.fc = nn.Identity()

        self.backbone = _inject_tsm(
            net,
            num_frames=num_frames,
            fold_div=fold_div,
            drop_path_rate=float(drop_path_rate),
        )
        self.num_frames = num_frames

        self.feat_drop = nn.Dropout(p=float(feat_dropout)) if feat_dropout > 0 else nn.Identity()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.head_drop = nn.Dropout(p=float(head_dropout)) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(2 * gru_hidden, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        B, T, C, H, W = video_batch.shape
        assert T == self.num_frames

        frames = video_batch.reshape(B * T, C, H, W)

        # (B*T, feature_dim) — TSM shifts happen inside backbone blocks
        features = self.backbone(frames)
        features = torch.flatten(features, start_dim=1)

        # (B, T, feature_dim)
        features = self.feat_drop(features)
        sequence = features.view(B, T, -1)

        # (B, T, 2*gru_hidden)
        gru_out, _ = self.gru(sequence)

        # Mean over time → (B, 2*gru_hidden)
        pooled = gru_out.mean(dim=1)
        pooled = self.head_drop(pooled)

        return self.classifier(pooled)
