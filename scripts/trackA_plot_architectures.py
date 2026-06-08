"""
Architecture diagrams for TSM_ultra (TSMResNet) and VideoFormer-Lite ultra (VFL_ultra).
Outputs two side-by-side figures saved to docs/figures/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ──────────────────────────────────────────────────────────
C_INPUT   = "#cfe2f3"   # light blue
C_CONV    = "#d9ead3"   # light green  (CNN / ResNet stages)
C_SHIFT   = "#fff2cc"   # yellow       (TemporalShift)
C_POOL    = "#fce5cd"   # light orange (pooling / reshape)
C_ATTN    = "#e8d5f5"   # lavender     (Transformer / attention)
C_HEAD    = "#f4cccc"   # light red    (classifier head)
C_EMBED   = "#d0e0e3"   # teal-ish     (embeddings)
ARROW     = "#555555"

def box(ax, x, y, w, h, label, sub="", color="#ffffff", fontsize=8.5, subsize=7):
    """Draw a rounded rectangle with a label and optional sub-label."""
    bp = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=0.8,
        edgecolor="#444444",
        facecolor=color,
    )
    ax.add_patch(bp)
    dy = 0.012 if sub else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", wrap=True)
    if sub:
        ax.text(x, y - 0.055, sub, ha="center", va="center",
                fontsize=subsize, color="#555555")

def arrow(ax, x, y0, y1, label=""):
    ax.annotate(
        "", xy=(x, y1 + 0.01), xytext=(x, y0 - 0.01),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=0.9),
    )
    if label:
        ax.text(x + 0.07, (y0 + y1)/2, label,
                fontsize=6.5, color="#666666", va="center")

def side_arrow(ax, x0, x1, y, label=""):
    ax.annotate(
        "", xy=(x1 - 0.01, y), xytext=(x0 + 0.01, y),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=0.9),
    )
    if label:
        ax.text((x0+x1)/2, y + 0.04, label, ha="center",
                fontsize=6.5, color="#666666")


# ════════════════════════════════════════════════════════════════════════════
#  Figure 1 – TSM_ultra  (TSMResNet with ResNet18 + TemporalShift)
# ════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(5.2, 12))
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis("off")
ax1.set_title("TSM_ultra Architecture\n(TSMResNet · ResNet18 · fold_div=4)",
              fontsize=11, fontweight="bold", pad=8)

X = 0.5
W, H = 0.68, 0.055

# layer positions (top → bottom)
rows = {
    "input":     0.945,
    "reshape1":  0.875,
    "stem":      0.805,
    "l1_ts":     0.718,
    "l1_blk":    0.655,
    "l2_ts":     0.568,
    "l2_blk":    0.505,
    "l3_ts":     0.418,
    "l3_blk":    0.355,
    "l4_ts":     0.268,
    "l4_blk":    0.205,
    "gap":       0.135,
    "reshape2":  0.072,
    "dropout":   0.018,
}

# ─ blocks ─
box(ax1, X, rows["input"],    W, H, "Video Clip  (B, T=4, C=3, H=224, W=224)", color=C_INPUT)
box(ax1, X, rows["reshape1"], W, H, "Reshape  →  (B·T, 3, 224, 224)",          color=C_POOL)
box(ax1, X, rows["stem"],     W, H, "ResNet18 Stem  (Conv 7×7, BN, ReLU, MaxPool)",
    sub="out: (B·T, 64, 56, 56)", color=C_CONV)

for stage, ts_y, blk_y, ch, sp in [
    (1, rows["l1_ts"], rows["l1_blk"], 64,  "56×56"),
    (2, rows["l2_ts"], rows["l2_blk"], 128, "28×28"),
    (3, rows["l3_ts"], rows["l3_blk"], 256, "14×14"),
    (4, rows["l4_ts"], rows["l4_blk"], 512, "7×7"),
]:
    box(ax1, X, ts_y,  W, H,
        f"TemporalShift  (fold_div=4)",
        sub=f"¼ ch → past · ¼ ch → future · ½ ch unchanged",
        color=C_SHIFT, subsize=6.5)
    box(ax1, X, blk_y, W, H,
        f"ResNet18 Layer {stage}  (2 × BasicBlock + StochDepth)",
        sub=f"out: (B·T, {ch}, {sp})   drop_path linearly ↑ to 0.10",
        color=C_CONV, subsize=6.5)

box(ax1, X, rows["gap"],     W, H, "Global Average Pool  +  Flatten",
    sub="(B·T, 512)", color=C_POOL)
box(ax1, X, rows["reshape2"], W, H,
    "Reshape  →  mean over T  →  (B, 512)",
    sub="temporal pooling by mean", color=C_POOL)
box(ax1, X, rows["dropout"],  W, H,
    "Dropout (p=0.3)  →  Linear(512, 33)  →  Logits",
    color=C_HEAD)

# ─ arrows ─
ys = list(rows.values())
for i in range(len(ys) - 1):
    arrow(ax1, X, ys[i], ys[i+1])

plt.tight_layout()
fig1.savefig("docs/figures/arch_tsm_ultra.png", dpi=150, bbox_inches="tight")
print("Saved docs/figures/arch_tsm_ultra.png")


# ════════════════════════════════════════════════════════════════════════════
#  Figure 2 – VFL_ultra  (VideoFormerLite)
# ════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(5.2, 12))
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis("off")
ax2.set_title("VFL_ultra Architecture\n(VideoFormer-Lite · ResNet18 + 2×Transformer)",
              fontsize=11, fontweight="bold", pad=8)

X = 0.5
rows2 = {
    "input":    0.948,
    "reshape1": 0.882,
    "cnn":      0.808,
    "gap":      0.735,
    "reshape2": 0.662,
    "cls_pos":  0.585,
    "tokens":   0.512,
    "tb1_ln1":  0.440,
    "tb1_attn": 0.378,
    "tb1_ln2":  0.315,
    "tb1_mlp":  0.252,
    "tb2":      0.175,
    "cls_ext":  0.108,
    "head":     0.038,
}

box(ax2, X, rows2["input"],   W, H, "Video Clip  (B, T=4, C=3, H=224, W=224)", color=C_INPUT)
box(ax2, X, rows2["reshape1"],W, H, "Reshape  →  (B·T, 3, 224, W)", color=C_POOL)
box(ax2, X, rows2["cnn"],     W, H, "ResNet18 Backbone  (conv stem + 4 stages)",
    sub="out: (B·T, 512, 7, 7)   [no TSM, no fc]", color=C_CONV)
box(ax2, X, rows2["gap"],     W, H, "Global Average Pool  +  Flatten",
    sub="(B·T, 512)", color=C_POOL)
box(ax2, X, rows2["reshape2"],W, H, "Reshape  →  (B, T=4, 512)  =  frame tokens",
    color=C_POOL)
box(ax2, X, rows2["cls_pos"], W, H,
    "Prepend [CLS] token  +  Temporal Pos. Embed.",
    sub="CLS ∈ ℝ^512 (learned) · pos ∈ ℝ^{T+1 × 512} (learned)",
    color=C_EMBED, subsize=6.5)
box(ax2, X, rows2["tokens"],  W, H, "Token sequence  (B, T+1=5, 512)",
    color=C_EMBED)

# Transformer block 1 (expanded)
box(ax2, X, rows2["tb1_ln1"], W, H,
    "Transformer Block 1  —  LayerNorm₁",
    sub="pre-LN (norm before attention)", color=C_ATTN)
box(ax2, X, rows2["tb1_attn"],W, H,
    "Multi-Head Self-Attention  (8 heads, d=512)",
    sub="attn_dropout=0.0  +  DropPath(0.00)", color=C_ATTN, subsize=6.5)
box(ax2, X, rows2["tb1_ln2"], W, H,
    "Residual add  →  LayerNorm₂",
    color=C_ATTN)
box(ax2, X, rows2["tb1_mlp"], W, H,
    "MLP  (512 → 2048 → 512, GELU)",
    sub="proj_dropout=0.0  +  DropPath(0.15)", color=C_ATTN, subsize=6.5)

box(ax2, X, rows2["tb2"],     W, H,
    "Transformer Block 2  (identical structure)",
    sub="DropPath rate linearly increases to 0.15", color=C_ATTN, subsize=6.5)

box(ax2, X, rows2["cls_ext"], W, H,
    "Extract [CLS] token  →  LayerNorm  →  (B, 512)",
    color=C_EMBED)
box(ax2, X, rows2["head"],    W, H,
    "Dropout (p=0.2)  →  Linear(512, 33)  →  Logits",
    color=C_HEAD)

# arrows
ys2 = list(rows2.values())
for i in range(len(ys2) - 1):
    arrow(ax2, X, ys2[i], ys2[i+1])

plt.tight_layout()
fig2.savefig("docs/figures/arch_vfl_ultra.png", dpi=150, bbox_inches="tight")
print("Saved docs/figures/arch_vfl_ultra.png")


# ════════════════════════════════════════════════════════════════════════════
#  Figure 3 – Side-by-side comparison overview
# ════════════════════════════════════════════════════════════════════════════

fig3, axes = plt.subplots(1, 2, figsize=(12, 9))
fig3.suptitle(
    "Architecture Comparison: TSM_ultra  vs  VFL_ultra",
    fontsize=13, fontweight="bold", y=0.98
)

for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ── Left: TSM_ultra summary ─────────────────────────────────────────────────
ax = axes[0]
ax.set_title("TSM_ultra (TSMResNet)", fontsize=11, fontweight="bold", pad=6)

bW, bH = 0.86, 0.07
X = 0.5

layers_tsm = [
    (0.92, "Input  (B, 4, 3, 224, 224)",              C_INPUT,  ""),
    (0.83, "Reshape  →  (B·T, 3, 224, 224)",          C_POOL,   ""),
    (0.73, "Stem: Conv7×7 · BN · ReLU · MaxPool",     C_CONV,   "(B·T, 64, 56, 56)"),
    (0.62, "Layer1 [TSM + 2×BasicBlock]",              C_SHIFT,  "fold_div=4 · 64 ch · 56×56"),
    (0.52, "Layer2 [TSM + 2×BasicBlock]",              C_SHIFT,  "fold_div=4 · 128 ch · 28×28"),
    (0.42, "Layer3 [TSM + 2×BasicBlock]",              C_SHIFT,  "fold_div=4 · 256 ch · 14×14"),
    (0.32, "Layer4 [TSM + 2×BasicBlock]",              C_SHIFT,  "fold_div=4 · 512 ch · 7×7"),
    (0.22, "Global Avg Pool + mean over T",            C_POOL,   "(B, 512)"),
    (0.12, "Dropout(0.3) + Linear(512→33)",            C_HEAD,   "Logits (B, 33)"),
]

for (y, lbl, col, sub) in layers_tsm:
    box(ax, X, y, bW, bH, lbl, sub=sub, color=col, fontsize=8, subsize=6.5)

for i in range(len(layers_tsm)-1):
    arrow(ax, X, layers_tsm[i][0], layers_tsm[i+1][0])

# param / design annotations
ax.text(0.5, 0.03,
        "~11.2M params · StochDepth max=0.10 · Focal(γ=2) · AdamW lr=1e-3",
        ha="center", fontsize=7, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eeeeee", edgecolor="#aaaaaa"))

# ── Right: VFL_ultra summary ────────────────────────────────────────────────
ax = axes[1]
ax.set_title("VFL_ultra (VideoFormer-Lite)", fontsize=11, fontweight="bold", pad=6)

layers_vfl = [
    (0.92, "Input  (B, 4, 3, 224, 224)",                C_INPUT,  ""),
    (0.83, "Reshape  →  (B·T, 3, 224, 224)",            C_POOL,   ""),
    (0.73, "ResNet18 Backbone (no TSM)",                 C_CONV,   "(B·T, 512) after GAP"),
    (0.63, "Reshape  →  Frame tokens  (B, 4, 512)",     C_POOL,   ""),
    (0.53, "Prepend [CLS] + Pos. Embed.",               C_EMBED,  "(B, 5, 512)  learned"),
    (0.42, "Transformer Block 1",                        C_ATTN,   "Pre-LN · 8-head MHSA · MLP(4×) · DropPath"),
    (0.32, "Transformer Block 2",                        C_ATTN,   "Pre-LN · 8-head MHSA · MLP(4×) · DropPath"),
    (0.22, "Extract [CLS] → LayerNorm",                 C_EMBED,  "(B, 512)"),
    (0.12, "Dropout(0.2) + Linear(512→33)",             C_HEAD,   "Logits (B, 33)"),
]

for (y, lbl, col, sub) in layers_vfl:
    box(ax, X, y, bW, bH, lbl, sub=sub, color=col, fontsize=8, subsize=6.5)

for i in range(len(layers_vfl)-1):
    arrow(ax, X, layers_vfl[i][0], layers_vfl[i+1][0])

ax.text(0.5, 0.03,
        "~12.0M params · StochDepth max=0.15 · CE + LabelSmooth · AdamW lr=5e-4",
        ha="center", fontsize=7, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eeeeee", edgecolor="#aaaaaa"))

# ── Legend ──────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT, edgecolor="#444", label="Input / Output"),
    mpatches.Patch(facecolor=C_CONV,  edgecolor="#444", label="CNN / ResNet"),
    mpatches.Patch(facecolor=C_SHIFT, edgecolor="#444", label="Temporal Shift (TSM only)"),
    mpatches.Patch(facecolor=C_POOL,  edgecolor="#444", label="Reshape / Pooling"),
    mpatches.Patch(facecolor=C_ATTN,  edgecolor="#444", label="Transformer (VFL only)"),
    mpatches.Patch(facecolor=C_EMBED, edgecolor="#444", label="Learnable Embeddings"),
    mpatches.Patch(facecolor=C_HEAD,  edgecolor="#444", label="Classifier Head"),
]
fig3.legend(handles=legend_items, loc="lower center", ncol=4, fontsize=8,
            bbox_to_anchor=(0.5, -0.02),
            framealpha=0.9, edgecolor="#aaaaaa")

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
fig3.savefig("docs/figures/arch_comparison.png", dpi=150, bbox_inches="tight")
print("Saved docs/figures/arch_comparison.png")

plt.close("all")
print("Done.")
