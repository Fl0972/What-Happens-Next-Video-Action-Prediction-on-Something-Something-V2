"""
ViT backbone variants with temporal aggregation heads for video classification.

All models share the same (B, T, C, H, W) → (B, num_classes) interface.

Backbones:
  _ViTBackbone      — ViT-B/16 (768-dim, 86M params)
  _ViTSBackbone     — ViT-S/16 (384-dim, 22M params) — ~4× faster from scratch

Temporal heads:
  ViTBiGRU          — [CLS] per frame → BiGRU → mean pool → classifier
  ViTBiGRUAttn      — [CLS] per frame → BiGRU → learned temporal attention → classifier
  ViTPerceiver      — all patch tokens → N learned queries cross-attend → mean → classifier
  ViTSBiGRUAttn     — ViT-S/16 + BiGRU + temporal attention (scratch-optimised)

Diversity rationale (why ViT + RNN alongside TSM-ResNet):
  TSM uses CNN local receptive fields with temporal channel-shift; ViT uses global
  spatial self-attention. These two inductive biases are orthogonal, making a ViT-based
  model a strong ensemble partner for the closed-track leaderboard.

Input requirement: images must be 224×224 (positional embeddings fixed at construction
time for (224/16)^2 = 196 patches + 1 CLS = 197 tokens per frame).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

VIT_DIM = 768    # hidden_dim for vit_b_16
VIT_S_DIM = 384  # hidden_dim for ViT-S/16


class _ViTBackbone(nn.Module):
    """Torchvision ViT-B/16 encoder without the classification head.

    Returns either the [CLS] token (return_all_tokens=False) or the full
    sequence of N+1 tokens including patches (return_all_tokens=True).
    Mirrors the _process_input + encoder logic from torchvision's VisionTransformer.
    """

    def __init__(self, pretrained: bool, return_all_tokens: bool = False) -> None:
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit = vit_b_16(weights=weights)
        self.patch_size: int = vit.patch_size
        self.hidden_dim: int = vit.hidden_dim
        self.class_token = vit.class_token   # nn.Parameter (1, 1, D)
        self.conv_proj = vit.conv_proj       # Conv2d(3, D, patch_size, patch_size)
        self.encoder = vit.encoder           # includes pos_embedding + transformer layers
        self.return_all_tokens = return_all_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — must be 224×224
        n = x.shape[0]
        x = self.conv_proj(x)                          # (B, D, n_h, n_w)
        x = x.reshape(n, self.hidden_dim, -1)          # (B, D, N)
        x = x.permute(0, 2, 1)                        # (B, N, D)
        cls = self.class_token.expand(n, -1, -1)       # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                 # (B, N+1, D)
        x = self.encoder(x)                            # (B, N+1, D)
        if self.return_all_tokens:
            return x                                   # (B, N+1, D)
        return x[:, 0]                                 # (B, D)


# ---------------------------------------------------------------------------
# Option A: ViT-B/16 + BiGRU + mean pool
# ---------------------------------------------------------------------------

class ViTBiGRU(nn.Module):
    """ViT-B/16 per-frame [CLS] token + Bidirectional GRU with mean pooling.

    Forward:
        (B, T, C, H, W) → vit_b_16 [CLS] per frame → (B, T, 768)
                        → BiGRU → (B, T, 2*gru_hidden)
                        → mean over T → Linear → (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = True,
        gru_hidden: int = 256,
        gru_layers: int = 1,
        feat_dropout: float = 0.0,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = _ViTBackbone(pretrained=pretrained, return_all_tokens=False)
        self.num_frames = num_frames

        self.feat_drop = nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity()
        self.gru = nn.GRU(
            VIT_DIM, gru_hidden, gru_layers,
            batch_first=True, bidirectional=True,
        )
        self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(2 * gru_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feats = self.backbone(x.view(B * T, C, H, W))     # (B*T, 768)
        feats = self.feat_drop(feats).view(B, T, VIT_DIM)  # (B, T, 768)
        out, _ = self.gru(feats)                           # (B, T, 2*hidden)
        pooled = out.mean(dim=1)                           # (B, 2*hidden)
        return self.classifier(self.head_drop(pooled))


# ---------------------------------------------------------------------------
# Option B: ViT-B/16 + BiGRU + temporal attention pooling
# ---------------------------------------------------------------------------

class ViTBiGRUAttn(nn.Module):
    """ViT-B/16 per-frame [CLS] token + BiGRU + learned temporal attention pooling.

    A single Linear(2*gru_hidden, 1) scores each time step; softmax over T gives
    a per-frame importance weight. The weighted sum focuses on the most informative
    frame, which helps for actions with a clear decisive moment.

    Forward:
        (B, T, C, H, W) → vit_b_16 [CLS] per frame → (B, T, 768)
                        → BiGRU → (B, T, 2*gru_hidden)
                        → attn scores → softmax → weighted sum → Linear → (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = True,
        gru_hidden: int = 256,
        gru_layers: int = 1,
        feat_dropout: float = 0.0,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = _ViTBackbone(pretrained=pretrained, return_all_tokens=False)
        self.num_frames = num_frames

        self.feat_drop = nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity()
        self.gru = nn.GRU(
            VIT_DIM, gru_hidden, gru_layers,
            batch_first=True, bidirectional=True,
        )
        self.attn_proj = nn.Linear(2 * gru_hidden, 1, bias=False)
        self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(2 * gru_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feats = self.backbone(x.view(B * T, C, H, W))      # (B*T, 768)
        feats = self.feat_drop(feats).view(B, T, VIT_DIM)  # (B, T, 768)
        out, _ = self.gru(feats)                            # (B, T, 2*hidden)
        scores = self.attn_proj(out).squeeze(-1)            # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (out * weights).sum(dim=1)                 # (B, 2*hidden)
        return self.classifier(self.head_drop(pooled))


# ---------------------------------------------------------------------------
# Option C: ViT-B/16 + Perceiver cross-attention head
# ---------------------------------------------------------------------------

class ViTPerceiver(nn.Module):
    """ViT-B/16 all patch tokens + Perceiver cross-attention head.

    Inspired by G19's exp4a (LB 0.4971). N learned queries cross-attend the
    full set of T × (N_patches + 1) frame tokens simultaneously, fusing spatial
    and temporal context without imposing a sequential ordering.

    Architecture:
        per-frame ViT-B/16 → (B*T, 197, 768) → reshape (B, T*197, 768)
        N learned queries cross-attend with pre-LN cross-attention (L layers)
        mean over queries → LayerNorm → Dropout → Linear → (B, num_classes)

    Forward:
        (B, T, C, H, W) → all patch tokens → (B, T*197, 768)
                        → Perceiver cross-attn (queries: (B, num_queries, 768))
                        → mean over queries → Linear → (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = True,
        num_queries: int = 16,
        perceiver_heads: int = 8,
        perceiver_layers: int = 2,
        feat_dropout: float = 0.0,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = _ViTBackbone(pretrained=pretrained, return_all_tokens=True)
        self.num_frames = num_frames

        self.queries = nn.Parameter(torch.empty(1, num_queries, VIT_DIM))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Pre-LN cross-attention layers: queries attend to frame tokens (keys/values)
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(VIT_DIM, perceiver_heads, batch_first=True, dropout=0.0)
            for _ in range(perceiver_layers)
        ])
        self.norms_q = nn.ModuleList([nn.LayerNorm(VIT_DIM) for _ in range(perceiver_layers)])
        self.norms_kv = nn.ModuleList([nn.LayerNorm(VIT_DIM) for _ in range(perceiver_layers)])

        self.feat_drop = nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity()
        self.final_norm = nn.LayerNorm(VIT_DIM)
        self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(VIT_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        tokens = self.backbone(x.view(B * T, C, H, W))   # (B*T, N+1, 768)
        Np1 = tokens.shape[1]
        tokens = tokens.view(B, T * Np1, VIT_DIM)        # (B, T*(N+1), 768)
        tokens = self.feat_drop(tokens)

        q = self.queries.expand(B, -1, -1)               # (B, num_queries, 768)
        for cross, norm_q, norm_kv in zip(self.cross_attn, self.norms_q, self.norms_kv):
            kv = norm_kv(tokens)
            h, _ = cross(norm_q(q), kv, kv, need_weights=False)
            q = q + h                                    # residual connection

        pooled = self.final_norm(q.mean(dim=1))          # (B, 768)
        return self.classifier(self.head_drop(pooled))


# ---------------------------------------------------------------------------
# ViT-S/16 backbone (384-dim, 22M params — ~4× faster than ViT-B/16 from scratch)
# ---------------------------------------------------------------------------

class _ViTSBackbone(nn.Module):
    """ViT-S/16 encoder without the classification head.

    Implemented by building a ViT-B/16 and replacing the encoder with a
    smaller configuration: depth=12, heads=6, mlp_dim=1536, hidden=384.
    This matches the DeiT-Small / ViT-Small architecture exactly.

    Returns the [CLS] token: shape (B, 384).
    """

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("ViT-S/16 pretrained weights are not available in torchvision; use pretrained=False.")
        import math
        from torchvision.models.vision_transformer import Encoder, EncoderBlock

        hidden_dim = VIT_S_DIM
        patch_size = 16
        image_size = 224
        seq_length = (image_size // patch_size) ** 2 + 1  # 197

        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.conv_proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # ViT-Small: 12 layers, 6 heads, mlp_dim=1536 (4× hidden)
        self.encoder = Encoder(
            seq_length=seq_length,
            num_layers=12,
            num_heads=6,
            hidden_dim=hidden_dim,
            mlp_dim=1536,
            dropout=0.0,
            attention_dropout=0.0,
            norm_layer=nn.LayerNorm,
        )

        # Weight init following DeiT
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(2.0 / (3 * patch_size * patch_size)))
        nn.init.zeros_(self.conv_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x = self.conv_proj(x)                          # (B, 384, n_h, n_w)
        x = x.reshape(n, self.hidden_dim, -1)          # (B, 384, N)
        x = x.permute(0, 2, 1)                        # (B, N, 384)
        cls = self.class_token.expand(n, -1, -1)       # (B, 1, 384)
        x = torch.cat([cls, x], dim=1)                 # (B, N+1, 384)
        x = self.encoder(x)                            # (B, N+1, 384)
        return x[:, 0]                                 # (B, 384)


# ---------------------------------------------------------------------------
# ViT-S/16 + BiGRU + temporal attention — scratch-optimised
# ---------------------------------------------------------------------------

class ViTSBiGRUAttn(nn.Module):
    """ViT-S/16 per-frame [CLS] token + BiGRU + learned temporal attention pooling.

    Same head design as ViTBiGRUAttn but 4× cheaper backbone (22M vs 86M params).
    Intended for from-scratch training on small datasets.

    Forward:
        (B, T, C, H, W) → vit_s_16 [CLS] per frame → (B, T, 384)
                        → BiGRU → (B, T, 2*gru_hidden)
                        → attn scores → softmax → weighted sum → Linear → (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = False,
        gru_hidden: int = 192,
        gru_layers: int = 1,
        feat_dropout: float = 0.0,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = _ViTSBackbone(pretrained=pretrained)
        self.num_frames = num_frames

        self.feat_drop = nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity()
        self.gru = nn.GRU(
            VIT_S_DIM, gru_hidden, gru_layers,
            batch_first=True, bidirectional=True,
        )
        self.attn_proj = nn.Linear(2 * gru_hidden, 1, bias=False)
        self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(2 * gru_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feats = self.backbone(x.view(B * T, C, H, W))       # (B*T, 384)
        feats = self.feat_drop(feats).view(B, T, VIT_S_DIM)  # (B, T, 384)
        out, _ = self.gru(feats)                             # (B, T, 2*gru_hidden)
        scores = self.attn_proj(out).squeeze(-1)             # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1) # (B, T, 1)
        pooled = (out * weights).sum(dim=1)                  # (B, 2*gru_hidden)
        return self.classifier(self.head_drop(pooled))
