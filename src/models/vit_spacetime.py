"""
Stage-1 MAE pretraining + Stage-2 ViT with SpaceTimeBlocks + Perceiver head.

G19 exp4a architecture:
  Stage 1 (MAEViT):
    Single-frame MAE — 75% spatial masking, ViT-B/16 encoder, lightweight
    decoder (4 blocks / 512 dim), MSE norm-pix loss. NO horizontal flip.

  Stage 2 (ViTSpaceTimePerceiver):
    Input (B, T, 3, 224, 224)
    → per-frame PatchEmbed + sincos spatial pos + CLS
    → temporal positional embedding added
    → first (depth - spacetime_depth) ViTBlocks  [spatial attention only]
    → last spacetime_depth SpaceTimeBlocks        [spatial + temporal attention]
    → return all tokens (B, T, 197, D)
    → 16 learned Perceiver queries cross-attend → mean → Linear → 33 classes

MAE pretrained encoder weights load into ViTSpaceTimePerceiver via strict=False:
  spatial parts of SpaceTimeBlock share parameter names with ViTBlock
  (norm1, attn, norm2, mlp) → exact key match for blocks.6-11.
  Temporal parts (temporal_norm, temporal_attn) are new and zero-init at proj.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants (ViT-B/16 defaults)
# ---------------------------------------------------------------------------

VIT_DIM = 768
VIT_HEADS = 12
VIT_DEPTH = 12
VIT_MLP_RATIO = 4.0
PATCH_SIZE = 16
IMG_SIZE = 224
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 196
NUM_TOKENS = NUM_PATCHES + 1                  # 197 (CLS + patches)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2D image → flat patch sequence, (B, C, H, W) → (B, N, D)."""

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        patch_size: int = PATCH_SIZE,
        in_chans: int = 3,
        embed_dim: int = VIT_DIM,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


# ---------------------------------------------------------------------------
# Sincos 2D positional embedding (fixed, no parameters)
# ---------------------------------------------------------------------------

def sincos_pos_embed_2d(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Returns (grid_size^2, embed_dim) sincos 2D positional embedding.

    Each patch at (row, col) gets [sin(row*ω), cos(row*ω), sin(col*ω), cos(col*ω)]
    where ω spans embed_dim//4 log-spaced frequencies.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    quarter = embed_dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float32) / quarter))

    h = w = grid_size
    rows = torch.arange(h, dtype=torch.float32)  # (h,)
    cols = torch.arange(w, dtype=torch.float32)  # (w,)

    # (h, w) grids of row/col indices
    grid_rows = rows.unsqueeze(1).expand(h, w).reshape(-1)  # (h*w,)
    grid_cols = cols.unsqueeze(0).expand(h, w).reshape(-1)  # (h*w,)

    # (h*w, quarter)
    emb_sin_h = torch.sin(grid_rows.unsqueeze(1) * omega.unsqueeze(0))
    emb_cos_h = torch.cos(grid_rows.unsqueeze(1) * omega.unsqueeze(0))
    emb_sin_w = torch.sin(grid_cols.unsqueeze(1) * omega.unsqueeze(0))
    emb_cos_w = torch.cos(grid_cols.unsqueeze(1) * omega.unsqueeze(0))

    return torch.cat([emb_sin_h, emb_cos_h, emb_sin_w, emb_cos_w], dim=-1)  # (N, D)


def build_pos_embed(embed_dim: int, grid_size: int, with_cls: bool = True) -> torch.Tensor:
    """Build (1, N+1, D) positional embedding buffer (CLS pos = zeros)."""
    patch_pe = sincos_pos_embed_2d(embed_dim, grid_size)           # (N, D)
    if with_cls:
        cls_pe = torch.zeros(1, embed_dim)
        patch_pe = torch.cat([cls_pe, patch_pe], dim=0)            # (N+1, D)
    return patch_pe.unsqueeze(0)                                   # (1, N+1, D)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention with pre-projection QKV and output projection."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class ViTBlock(nn.Module):
    """Standard pre-LN ViT block (spatial attention only)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, T: int = 1) -> torch.Tensor:  # noqa: ARG002
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpaceTimeBlock(nn.Module):
    """ViT block augmented with factorized temporal attention (G19 SpaceTimeBlock).

    Spatial attention parts (norm1, attn, norm2, mlp) share names with ViTBlock
    so MAE pretrained encoder weights load via strict=False without key renaming.

    Temporal attention output projection is zero-initialized: at init, the block
    is identical to the corresponding pretrained ViTBlock.

    Forward (T > 1):
        1. Temporal self-attention across same-position tokens of all T frames
           (applied to all N tokens including CLS, factorized over spatial dim)
        2. Spatial self-attention within each frame (standard ViT)
        3. MLP
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Temporal attention (new; zero-init proj so pretrained behavior is preserved)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = Attention(dim, num_heads, attn_dropout, dropout)
        nn.init.zeros_(self.temporal_attn.proj.weight)
        nn.init.zeros_(self.temporal_attn.proj.bias)

        # Spatial attention + MLP — same names as ViTBlock for clean weight loading
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, T: int = 1) -> torch.Tensor:
        # x: (B*T, N, D)
        BT, N, D = x.shape
        B = BT // T

        if T > 1:
            # Temporal: reshape so each of the N token positions attends across T frames
            # (B*T, N, D) → (B, T, N, D) → (B, N, T, D) → (B*N, T, D)
            xt = x.view(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)
            xt = xt + self.temporal_attn(self.temporal_norm(xt))
            x = xt.reshape(B, N, T, D).permute(0, 2, 1, 3).reshape(B * T, N, D)

        # Spatial attention + MLP (standard ViT, per-frame)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Perceiver head (N learned queries cross-attend all video tokens)
# ---------------------------------------------------------------------------

class PerceiverHead(nn.Module):
    """N learned queries cross-attend all (B, S, D) context tokens.

    Pre-LN cross-attention with residual. Queries are updated in-place across
    L layers, then mean-pooled → LayerNorm → Dropout → Linear(D, num_classes).
    """

    def __init__(
        self,
        dim: int = VIT_DIM,
        num_queries: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        head_dropout: float = 0.3,
        num_classes: int = 33,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.0)
            for _ in range(num_layers)
        ])
        self.norms_q = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.norms_kv = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(dim)
        self.head_drop = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, S, D) — all video tokens
        B = context.shape[0]
        q = self.queries.expand(B, -1, -1)  # (B, Q, D)
        for cross, norm_q, norm_kv in zip(self.cross_attn, self.norms_q, self.norms_kv):
            kv = norm_kv(context)
            h, _ = cross(norm_q(q), kv, kv, need_weights=False)
            q = q + h
        pooled = self.final_norm(q.mean(dim=1))    # (B, D)
        return self.classifier(self.head_drop(pooled))


# ---------------------------------------------------------------------------
# Stage-2: ViTSpaceTimePerceiver
# ---------------------------------------------------------------------------

class ViTSpaceTimePerceiver(nn.Module):
    """ViT with SpaceTimeBlocks + Perceiver head (G19 exp4a downstream model).

    Interface: (B, T, C, H, W) → (B, num_classes)

    The .backbone attribute (patch_embed + blocks + norm) is exposed for
    build_optimizer's backbone_lr_scale split. For proper LLRD, call
    param_groups_llrd(lr, decay) to get per-layer param groups.

    Loading MAE pretrained weights:
        ckpt = torch.load(mae_path)
        model.load_state_dict(ckpt['encoder_state_dict'], strict=False)
    Keys patch_embed / cls_token / pos_embed / blocks.*.{norm1,attn,norm2,mlp}
    match the MAE encoder. The temporal and perceiver parts start freshly.
    """

    def __init__(
        self,
        num_classes: int = 33,
        num_frames: int = 4,
        img_size: int = IMG_SIZE,
        patch_size: int = PATCH_SIZE,
        embed_dim: int = VIT_DIM,
        depth: int = VIT_DEPTH,
        num_heads: int = VIT_HEADS,
        mlp_ratio: float = VIT_MLP_RATIO,
        spacetime_depth: int = 6,
        num_queries: int = 16,
        perceiver_heads: int = 8,
        perceiver_layers: int = 2,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        head_dropout: float = 0.3,
        mae_pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.depth = depth
        self.spacetime_depth = spacetime_depth

        # Patch embedding + sincos spatial positional embedding (frozen)
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            'pos_embed',
            build_pos_embed(embed_dim, img_size // patch_size, with_cls=True),
        )  # (1, 197, D) — shared across all frames

        # Learnable temporal positional embedding: (1, T, 1, D) broadcast to all tokens
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

        # Transformer blocks
        spatial_depth = depth - spacetime_depth
        self.blocks = nn.ModuleList(
            [ViTBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout) for _ in range(spatial_depth)]
            + [SpaceTimeBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout) for _ in range(spacetime_depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Perceiver classification head
        self.perceiver = PerceiverHead(
            dim=embed_dim,
            num_queries=num_queries,
            num_heads=perceiver_heads,
            num_layers=perceiver_layers,
            head_dropout=head_dropout,
            num_classes=num_classes,
        )

        # Expose backbone for build_optimizer backbone_lr_scale
        self.backbone = nn.ModuleList([self.patch_embed, *self.blocks, self.norm])

        self._init_weights()

        if mae_pretrained_path is not None:
            self._load_mae_pretrained(mae_pretrained_path)

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Re-zero temporal attention output projections AFTER general init so that,
        # at init, each SpaceTimeBlock behaves identically to its pretrained ViTBlock.
        for block in self.blocks:
            if isinstance(block, SpaceTimeBlock):
                nn.init.zeros_(block.temporal_attn.proj.weight)
                nn.init.zeros_(block.temporal_attn.proj.bias)

    def _load_mae_pretrained(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('encoder_state_dict', ckpt)
        missing, unexpected = self.load_state_dict(state, strict=False)
        # Expected missing: temporal_embed, temporal_norm/attn in SpaceTimeBlocks, perceiver, head_drop
        print(f"[MAE load] missing={len(missing)}, unexpected={len(unexpected)}")
        if unexpected:
            print(f"  unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    def param_groups_llrd(self, lr: float, decay: float = 0.75) -> List[dict]:
        """Layer-wise learning rate decay parameter groups.

        Block i (0-indexed from shallow to deep) gets lr * decay^(depth - i).
        The patch embed / CLS / temporal embed get lr * decay^(depth + 1).
        The norm and perceiver head get the full lr.
        """
        groups: List[dict] = []
        stem_params = list(self.patch_embed.parameters()) + [self.cls_token, self.temporal_embed]
        groups.append({'params': stem_params, 'lr': lr * (decay ** (self.depth + 1)), 'name': 'stem'})
        for i, block in enumerate(self.blocks):
            groups.append({
                'params': list(block.parameters()),
                'lr': lr * (decay ** (self.depth - i)),
                'name': f'block_{i}',
            })
        groups.append({'params': list(self.norm.parameters()), 'lr': lr, 'name': 'norm'})
        groups.append({'params': list(self.perceiver.parameters()), 'lr': lr, 'name': 'perceiver'})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Per-frame patch embedding + sincos spatial positional embedding
        frames = x.view(B * T, C, H, W)
        tokens = self.patch_embed(frames)          # (B*T, N, D)
        cls = self.cls_token.expand(B * T, -1, -1) # (B*T, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)   # (B*T, N+1, D)
        tokens = tokens + self.pos_embed           # broadcast (1, N+1, D)

        # Add temporal positional embedding
        # temporal_embed: (1, T, 1, D) → broadcast over patches dim → (B*T, N+1, D) needed
        # Reshape to (B, T, N+1, D), add temporal embed per frame, reshape back
        tokens = tokens.view(B, T, -1, self.embed_dim)       # (B, T, N+1, D)
        tokens = tokens + self.temporal_embed[:, :T, :, :]   # (B, T, N+1, D)
        tokens = tokens.view(B * T, -1, self.embed_dim)      # (B*T, N+1, D)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, T=T)            # (B*T, N+1, D)

        tokens = self.norm(tokens)                 # (B*T, N+1, D)

        # Reshape to (B, T, N+1, D) and flatten spatial-temporal → (B, T*(N+1), D)
        N1 = tokens.shape[1]
        context = tokens.view(B, T * N1, self.embed_dim)

        return self.perceiver(context)             # (B, num_classes)


# ---------------------------------------------------------------------------
# Stage-1: Masked Autoencoder (MAEViT)
# ---------------------------------------------------------------------------

class MAEViT(nn.Module):
    """Masked Autoencoder with ViT-B/16 encoder and lightweight decoder.

    Stage-1 self-supervised pretraining on single frames.

    forward() returns (loss, pred, mask):
      loss  — scalar MSE reconstruction loss (norm-pix on masked patches)
      pred  — (B, N, patch_size^2 * 3) reconstructed pixel values for ALL patches
      mask  — (B, N) bool tensor, True = masked (hidden from encoder)

    Save encoder weights for Stage-2 loading:
        torch.save({'encoder_state_dict': mae.encoder_state_dict()}, path)
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        patch_size: int = PATCH_SIZE,
        embed_dim: int = VIT_DIM,
        depth: int = VIT_DEPTH,
        num_heads: int = VIT_HEADS,
        mlp_ratio: float = VIT_MLP_RATIO,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 16,
        mask_ratio: float = 0.75,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # ---- Encoder ----
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches  # 196
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            'pos_embed',
            build_pos_embed(embed_dim, img_size // patch_size, with_cls=True),
        )  # (1, 197, D)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ---- Decoder ----
        # Linear projection from encoder dim to decoder dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        # Learnable mask token (replaces masked patches in decoder)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer(
            'decoder_pos_embed',
            build_pos_embed(decoder_embed_dim, img_size // patch_size, with_cls=True),
        )  # (1, 197, decoder_D)
        self.decoder_blocks = nn.ModuleList([
            ViTBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        # Predict raw pixels: patch_size^2 * 3 values per patch
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Masking utilities
    # ------------------------------------------------------------------

    def _random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask patch tokens; return (visible_tokens, ids_restore, mask).

        x:           (B, N, D) — all patch tokens (no CLS yet)
        mask_ratio:  fraction to mask
        Returns:
          x_masked:  (B, N_vis, D)  visible tokens only
          ids_restore: (B, N) long  indices that un-shuffle x_masked + mask_tokens
          mask:      (B, N) bool    True = masked
        """
        B, N, D = x.shape
        N_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :N_keep]
        x_masked = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)  # False = visible

        return x_masked, ids_restore, mask

    # ------------------------------------------------------------------
    # Encoder forward (used for pretraining and for extracting weights)
    # ------------------------------------------------------------------

    def _encode(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run masking + encoder. Returns (latent, ids_restore, mask)."""
        # x: (B, C, H, W)
        tokens = self.patch_embed(x)                      # (B, N, D)
        tokens = tokens + self.pos_embed[:, 1:, :]        # sincos pos (no CLS yet)

        tokens, ids_restore, mask = self._random_masking(tokens, mask_ratio)

        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)          # prepend CLS
        tokens = tokens + self.pos_embed[:, :1, :]        # add CLS pos (zeros)

        for block in self.blocks:
            tokens = block(tokens)

        latent = self.norm(tokens)                         # (B, N_vis+1, D)
        return latent, ids_restore, mask

    # ------------------------------------------------------------------
    # Decoder forward
    # ------------------------------------------------------------------

    def _decode(
        self, latent: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct all N patch tokens from encoder latent.

        latent:      (B, N_vis+1, D)  encoder output (CLS + visible patches)
        ids_restore: (B, N) long      indices to restore original patch order
        Returns: (B, N, patch_size^2 * 3)
        """
        B = latent.shape[0]
        N = ids_restore.shape[1]
        N_vis = latent.shape[1] - 1  # exclude CLS

        # Project encoder dim → decoder dim
        x = self.decoder_embed(latent)               # (B, N_vis+1, dec_D)

        # Fill in mask tokens in original order (excluding CLS at index 0)
        dec_D = x.shape[-1]
        mask_tokens = self.mask_token.expand(B, N - N_vis, dec_D)
        x_patches = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # (B, N, dec_D)
        x_patches = x_patches.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, dec_D)
        )  # restore original order

        # Re-attach CLS + add decoder pos embed
        x = torch.cat([x[:, :1, :], x_patches], dim=1)  # (B, N+1, dec_D)
        x = x + self.decoder_pos_embed                   # broadcast (1, N+1, dec_D)

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)                         # (B, N+1, patch_px)
        return x[:, 1:, :]                               # (B, N, patch_px) — drop CLS

    # ------------------------------------------------------------------
    # Patchify / unpatchify helpers
    # ------------------------------------------------------------------

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, N, patch_size^2 * 3) patch pixel values."""
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(imgs.shape[0], h * w, p * p * 3)
        return x

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(B, N, patch_size^2 * 3) → (B, 3, H, W)."""
        p = self.patch_size
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], 3, h * p, w * p)

    # ------------------------------------------------------------------
    # Loss: MSE on norm-pix for masked patches only
    # ------------------------------------------------------------------

    def _norm_pix_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss on normalised pixel values for masked patches.

        norm-pix normalises each patch by its own mean and variance,
        following the original MAE paper.
        """
        target = self.patchify(imgs)                # (B, N, p^2*3)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        loss = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
        loss = (loss * mask.float()).sum() / mask.float().sum()
        return loss

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(
        self, imgs: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        imgs: (B, C, H, W) single frames
        Returns: (loss, pred, mask)
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        latent, ids_restore, mask = self._encode(imgs, mask_ratio)
        pred = self._decode(latent, ids_restore)
        loss = self._norm_pix_loss(imgs, pred, mask)
        return loss, pred, mask

    # ------------------------------------------------------------------
    # Export encoder weights for Stage-2 loading
    # ------------------------------------------------------------------

    def encoder_state_dict(self) -> dict:
        """Return state dict with only encoder keys (patch_embed, cls_token,
        pos_embed, blocks, norm) — compatible with ViTSpaceTimePerceiver."""
        full = self.state_dict()
        encoder_keys = {
            'patch_embed', 'cls_token', 'pos_embed', 'blocks', 'norm'
        }
        return {
            k: v for k, v in full.items()
            if any(k.startswith(prefix) for prefix in encoder_keys)
        }
