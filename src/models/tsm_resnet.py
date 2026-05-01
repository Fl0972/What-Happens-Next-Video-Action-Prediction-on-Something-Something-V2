"""
Temporal Shift Module (TSM) on top of ResNet.

TSM shifts a fraction of channels along the time axis before each residual block,
letting frames share information across time with zero extra parameters or FLOPs.

Reference: Lin et al., "TSM: Temporal Shift Module for Efficient Video Understanding"
           https://arxiv.org/abs/1811.08383

Forward:
    Input:  (B, T, C, H, W)
    Reshape: (B*T, C, H, W)   <- TemporalShift layers reshape internally to (B,T,C,H,W)
    Backbone: ResNet50 with TSM injected in every residual block
    Pool: global average pool -> (B*T, 2048)
    Flatten + reshape: (B, T, 2048) -> mean over T -> (B, 2048)
    Linear: (B, num_classes)
"""

from __future__ import annotations

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


def _inject_tsm(backbone: nn.Module, num_frames: int, fold_div: int) -> nn.Module:
    """Wrap every residual block in the four ResNet stages with TemporalShift."""
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(backbone, stage_name)
        for i, block in enumerate(stage):
            stage[i] = TemporalShift(block, num_frames=num_frames, fold_div=fold_div)
    return backbone


class TSMResNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = False,
        backbone: str = "resnet50",
        fold_div: int = 8,
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

        self.backbone = _inject_tsm(net, num_frames=num_frames, fold_div=fold_div)
        self.num_frames = num_frames
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

        return self.classifier(pooled)
