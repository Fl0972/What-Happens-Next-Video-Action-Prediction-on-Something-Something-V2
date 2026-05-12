"""
Temporal Shift Module (TSM) on top of ResNet.

TSM shifts a fraction of channels along the time axis before each residual block,
letting frames share information across time with zero extra parameters or FLOPs.

Reference: Lin et al., "TSM: Temporal Shift Module for Efficient Video Understanding"
           https://arxiv.org/abs/1811.08383

Forward:
    Input:  (B, T, C, H, W)
    Reshape: (B*T, C, H, W)   <- TemporalShift layers reshape internally to (B,T,C,H,W)
    Backbone: ResNet with TSM (and optional stochastic depth) injected in every block
    Pool: global average pool -> (B*T, feature_dim)
    Flatten + reshape: (B, T, feature_dim) -> mean over T -> (B, feature_dim)
    Optional dropout, then linear: (B, num_classes)

Closed-track (from-scratch) extras:
    * Stochastic depth on the residual branch (linearly scaled per block, a la Huang
      et al., "Deep Networks with Stochastic Depth", ECCV 2016).
    * Dropout right before the classifier head.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision import models


class TemporalShift(nn.Module):
    """
    Wraps a residual block and shifts channels along the time dimension before
    forwarding through the block.  No learnable parameters are added.

    fold_div=8  ->  1/8 of channels shift backward (see past frame),
                    1/8 of channels shift forward  (see future frame),
                    remaining 6/8 channels are unchanged.
    """

    def __init__(self, block: nn.Module, num_frames: int, fold_div: int = 8) -> None:
        super().__init__()
        self.block = block
        self.num_frames = num_frames
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._shift(x)
        return self.block(x)

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        BT, C, H, W = x.shape
        T = self.num_frames
        B = BT // T
        fold = C // self.fold_div

        x = x.view(B, T, C, H, W)
        out = x.clone()

        # Shift [0:fold] one step into the future: frame t reads from frame t-1
        out[:, 1:, :fold] = x[:, :-1, :fold]
        out[:, 0, :fold] = 0

        # Shift [fold:2*fold] one step into the past: frame t reads from frame t+1
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]
        out[:, -1, fold : 2 * fold] = 0

        # Remaining channels [2*fold:] are unchanged (already copied by clone)
        return out.view(BT, C, H, W)


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample DropPath: zero a fraction of samples and rescale survivors."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask / keep_prob


class _ResBlockWithDropPath(nn.Module):
    """
    Wraps a torchvision BasicBlock or Bottleneck and applies stochastic depth
    on the residual branch only — the identity shortcut is always preserved.
    """

    def __init__(self, block: nn.Module, drop_prob: float) -> None:
        super().__init__()
        self.block = block
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        identity = x
        if hasattr(b, "conv3"):  # Bottleneck (ResNet50+)
            out = b.relu(b.bn1(b.conv1(x)))
            out = b.relu(b.bn2(b.conv2(out)))
            out = b.bn3(b.conv3(out))
        else:  # BasicBlock (ResNet18/34)
            out = b.relu(b.bn1(b.conv1(x)))
            out = b.bn2(b.conv2(out))
        out = _drop_path(out, self.drop_prob, self.training)
        if b.downsample is not None:
            identity = b.downsample(x)
        return b.relu(out + identity)


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    if num_blocks <= 1 or max_rate <= 0.0:
        return [0.0] * num_blocks
    return [max_rate * i / (num_blocks - 1) for i in range(num_blocks)]


def _inject_tsm(
    backbone: nn.Module,
    num_frames: int,
    fold_div: int,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Wrap every residual block in the four ResNet stages with TemporalShift,
    optionally adding stochastic depth on the residual branch (rate scales
    linearly from 0 at the first block to ``drop_path_rate`` at the last)."""
    stage_names = ("layer1", "layer2", "layer3", "layer4")
    total_blocks = sum(len(getattr(backbone, s)) for s in stage_names)
    rates = _linear_drop_path_rates(total_blocks, drop_path_rate)
    rate_iter = iter(rates)

    for stage_name in stage_names:
        stage = getattr(backbone, stage_name)
        for i, block in enumerate(stage):
            r = next(rate_iter)
            inner = block if r == 0.0 else _ResBlockWithDropPath(block, r)
            stage[i] = TemporalShift(inner, num_frames=num_frames, fold_div=fold_div)
    return backbone


class TSMResNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = False,
        backbone: str = "resnet50",
        fold_div: int = 8,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet50(weights=weights)
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}. Choose 'resnet50' or 'resnet18'.")

        feature_dim = net.fc.in_features
        net.fc = nn.Identity()

        self.backbone = _inject_tsm(
            net,
            num_frames=num_frames,
            fold_div=fold_div,
            drop_path_rate=float(drop_path_rate),
        )
        self.num_frames = num_frames
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        B, T, C, H, W = video_batch.shape
        assert T == self.num_frames, (
            f"Expected {self.num_frames} frames per clip, got {T}. "
            "TSM shift indices are fixed at construction time."
        )

        # (B*T, C, H, W) — TemporalShift blocks reshape internally for the shift
        frames = video_batch.reshape(B * T, C, H, W)

        # (B*T, feature_dim)
        features = self.backbone(frames)
        features = torch.flatten(features, start_dim=1)

        # (B, T, feature_dim) -> mean over T -> (B, feature_dim)
        pooled = features.view(B, T, -1).mean(dim=1)
        pooled = self.dropout(pooled)

        return self.classifier(pooled)
