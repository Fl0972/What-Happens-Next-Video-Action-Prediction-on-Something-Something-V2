"""
VideoFormer-Lite: a Frame-CNN + temporal Transformer for short-clip action recognition.

Designed for the closed-track challenge (4 frames per clip, 33 classes, trained
from scratch). The pipeline keeps spatial reasoning in a lightweight 2D CNN —
the natural inductive bias for images — and offloads temporal reasoning to a
small Transformer that attends across the 4 frame tokens.

Pipeline:
    Input (B, T, C, H, W)
      -> per-frame ResNet18 (random init), drop the final fc
      -> (B*T, 512, 7, 7) -> global avg pool -> (B*T, 512)
      -> reshape (B, T, 512)
      -> prepend learnable [CLS] token + add learnable temporal positional embedding
         (B, T+1, 512)
      -> L pre-LN Transformer encoder blocks (MHSA + MLP, DropPath on residuals)
      -> take [CLS] -> LayerNorm -> Dropout -> Linear(d_model, num_classes)

Lecture references:
    * Self-attention / positional encoding / [CLS] token / pre-LN Transformer
      encoder block (Lecture 6 — Vision Transformer slides).
    * Hybrid "CNN + self-attention" composition (Lecture 6, slide 27).
    * Bias-variance reasoning behind the small parameter count and heavy
      regularization on a modest dataset (Lecture 7).

Closed-track regularization knobs exposed by this module:
    * drop_path_rate: stochastic depth on Transformer residual branches.
    * attn_dropout / proj_dropout: dropout inside MHSA and MLP.
    * dropout: dropout right before the classifier head.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision import models


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth (Huang et al., 2016)."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask / keep_prob


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class _MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _TransformerBlock(nn.Module):
    """Pre-LN Transformer encoder block with DropPath on both residual branches."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_dropout: float,
        proj_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_path1 = _DropPath(drop_path)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model, d_ff, proj_dropout)
        self.drop_path2 = _DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path1(h)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """0 at the first block, max_rate at the last (Huang et al., 2016)."""
    if num_blocks <= 1 or max_rate <= 0.0:
        return [max_rate if num_blocks == 1 else 0.0] * num_blocks
    return [max_rate * i / (num_blocks - 1) for i in range(num_blocks)]


class VideoFormerLite(nn.Module):
    """
    Frame-CNN + temporal Transformer.

    Args:
        num_classes: number of output classes.
        num_frames:  T, fixed at construction (used to size the temporal pos. embed.).
        pretrained:  unused in the closed track (kept for interface parity).
        d_model:     transformer width — must equal the CNN feature dim (512 for ResNet18).
        n_heads:     number of MHSA heads.
        n_layers:    number of Transformer encoder blocks.
        mlp_ratio:   d_ff = round(d_model * mlp_ratio).
        attn_dropout:  dropout inside MHSA attention weights.
        proj_dropout:  dropout inside the MLP.
        drop_path_rate: max stochastic depth (linearly scaled across blocks).
        dropout:       dropout applied to the pooled [CLS] vector before the classifier.
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = False,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features  # 512
        if d_model != feature_dim:
            raise ValueError(
                f"d_model ({d_model}) must equal the ResNet18 feature dim ({feature_dim})."
            )
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.num_frames = int(num_frames)

        # [CLS] token + learnable temporal positional embedding (ViT-style).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_frames + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        d_ff = int(round(d_model * mlp_ratio))
        rates = _linear_drop_path_rates(n_layers, drop_path_rate)
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                    drop_path=rates[i],
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        B, T, C, H, W = video_batch.shape
        if T != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames per clip, got {T}. "
                "The positional embedding is sized at construction time."
            )

        frames = video_batch.reshape(B * T, C, H, W)
        feats = self.backbone(frames)              # (B*T, 512)
        feats = feats.view(B, T, -1)               # (B, T, 512)

        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, 512)
        tokens = torch.cat([cls, feats], dim=1)    # (B, T+1, 512)
        tokens = tokens + self.pos_embed           # (B, T+1, 512)

        for blk in self.blocks:
            tokens = blk(tokens)

        cls_out = self.norm(tokens[:, 0])          # (B, 512)
        cls_out = self.head_dropout(cls_out)
        return self.classifier(cls_out)            # (B, num_classes)
