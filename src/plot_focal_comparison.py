"""
Per-class F1 comparison: VFL-Ultra (CE) vs VFL-Ultra-Focal.
Highlights the fine-grained confusable action groups where focal loss
was expected to help most (SSv2-style near-identical classes).

Run from src/:
    python plot_focal_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score

LOGITS_DIR = Path(__file__).parent.parent / "models" / "val_logits"
OUT_PATH = Path(__file__).parent.parent / "docs" / "figures" / "focal_vs_ce_per_class.png"

# SSv2-style confusable groups — classes expected to benefit most from focal loss
HARD_GROUPS = {
    "Poking": [
        "Poking_something_so_it_falls_over",
        "Poking_something_so_it_slightly_moves",
        "Poking_something_so_that_it_does_not_fall_over",
    ],
    "Pushing": [
        "Pushing_something_so_it_falls_off_the_table",
        "Pushing_something_so_it_slightly_falls_over",
        "Pushing_something_so_that_it_almost_falls_off_but_doesnt",
    ],
    "Moving L/R": [
        "Moving_something_and_something_away_from_each_other",
        "Moving_something_and_something_closer_to_each_other",
    ],
    "Spilling": [
        "Spilling_something_behind_something",
        "Spilling_something_next_to_something",
        "Spilling_something_onto_something",
    ],
}

CLASS_NAMES = [
    "Closing_something",
    "Covering_something_with_something",
    "Dropping_something_into_something",
    "Folding_something",
    "Hitting_something_with_something",
    "Holding_something_behind_something",
    "Lifting_something_up_completely_then_putting_it_down_with_the_other_hand",
    "Moving_something_and_something_away_from_each_other",
    "Moving_something_and_something_closer_to_each_other",
    "Moving_something_away_from_the_camera",
    "Moving_something_towards_the_camera",
    "Picking_something_up",
    "Poking_something_so_it_falls_over",
    "Poking_something_so_it_slightly_moves",
    "Poking_something_so_that_it_does_not_fall_over",
    "Pouring_something_into_something",
    "Pretending_to_put_something_behind_something",
    "Pretending_to_put_something_into_something",
    "Pretending_to_put_something_next_to_something",
    "Pretending_to_put_something_on_a_surface",
    "Pretending_to_put_something_onto_something",
    "Pushing_something_so_it_falls_off_the_table",
    "Pushing_something_so_it_slightly_falls_over",
    "Pushing_something_so_that_it_almost_falls_off_but_doesnt",
    "Putting_something_behind_something",
    "Putting_something_next_to_something",
    "Putting_something_on_a_surface",
    "Putting_something_onto_something",
    "Putting_something_that_cannot_stand_upright_upright_on_the_table",
    "Spilling_something_behind_something",
    "Spilling_something_next_to_something",
    "Spilling_something_onto_something",
    "Throwing_something_in_the_air_and_catching_it",
]

SHORT_NAMES = [
    "Closing", "Covering w/sth", "Dropping into", "Folding", "Hitting",
    "Holding behind", "Lifting & down", "Moving apart", "Moving closer",
    "Moving away cam", "Moving toward cam", "Picking up",
    "Poking → falls", "Poking → slight move", "Poking → stays",
    "Pouring into", "Pretend behind", "Pretend into", "Pretend next to",
    "Pretend on surface", "Pretend onto", "Pushing → falls",
    "Pushing → slight fall", "Pushing → almost falls", "Putting behind",
    "Putting next to", "Putting on surface", "Putting onto",
    "Putting upright", "Spilling behind", "Spilling next to",
    "Spilling onto", "Throwing & catching",
]

# Classes that belong to a hard confusable group
HARD_CLASS_NAMES = {c for g in HARD_GROUPS.values() for c in g}
HARD_INDICES = {i for i, name in enumerate(CLASS_NAMES) if name in HARD_CLASS_NAMES}


def load_preds(stem: str, labels: np.ndarray):
    logits = np.load(LOGITS_DIR / f"{stem}.npy")
    preds = logits.argmax(axis=1)
    f1 = f1_score(labels, preds, average=None, labels=list(range(33)), zero_division=0)
    return f1


def main():
    labels = np.load(LOGITS_DIR / "labels.npy")
    f1_ce = load_preds("video_former_lite_ultra", labels)
    f1_focal = load_preds("vfl_ultra_focal", labels)
    delta = f1_focal - f1_ce

    # Sort by hard-group membership first, then by delta magnitude within each group
    hard_idx = sorted(HARD_INDICES, key=lambda i: -abs(delta[i]))
    other_idx = sorted(set(range(33)) - HARD_INDICES, key=lambda i: delta[i])
    order = hard_idx + other_idx

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1.6, 1]})

    # ── Left panel: absolute F1 per class (hard classes only) ─────────────────
    ax = axes[0]
    hi = sorted(HARD_INDICES, key=lambda i: CLASS_NAMES[i])
    x = np.arange(len(hi))
    w = 0.35
    bars_ce    = ax.barh(x - w/2, [f1_ce[i]    for i in hi], w, label="CE (VFL-Ultra)", color="#4C72B0", alpha=0.85)
    bars_focal = ax.barh(x + w/2, [f1_focal[i] for i in hi], w, label="Focal γ=2 (VFL-Ultra-Focal)", color="#DD8452", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels([SHORT_NAMES[i] for i in hi], fontsize=9)
    ax.set_xlabel("F1 score")
    ax.set_title("Hard confusable classes — F1: CE vs Focal", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 0.85)
    ax.axvline(0, color="black", linewidth=0.5)
    for bar in bars_focal:
        ax.axvline(bar.get_x() + bar.get_width(), color="none")

    # ── Right panel: Δ F1 (focal − CE) for ALL classes ───────────────────────
    ax2 = axes[1]
    y = np.arange(33)
    colors = ["#DD8452" if i in HARD_INDICES else "#8DA0CB" for i in order]
    delta_sorted = [delta[i] for i in order]
    ax2.barh(y, delta_sorted, color=colors, alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels([SHORT_NAMES[i] for i in order], fontsize=7.5)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("ΔF1 (Focal − CE)")
    ax2.set_title("All classes — Focal gain/loss", fontweight="bold")
    hard_patch  = mpatches.Patch(color="#DD8452", alpha=0.85, label="Hard confusable group")
    other_patch = mpatches.Patch(color="#8DA0CB", alpha=0.85, label="Other classes")
    ax2.legend(handles=[hard_patch, other_patch], fontsize=8, loc="lower right")

    overall_ce    = float((np.load(LOGITS_DIR / "video_former_lite_ultra.npy").argmax(1) == labels).mean())
    overall_focal = float((np.load(LOGITS_DIR / "vfl_ultra_focal.npy").argmax(1) == labels).mean())
    fig.suptitle(
        f"Focal loss effect on VFL-Ultra  |  Overall top-1: CE={overall_ce:.3f}  Focal={overall_focal:.3f}  Δ={overall_focal-overall_ce:+.3f}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")

    # Print summary table for hard classes
    print(f"\n{'Class':<40} {'CE':>6} {'Focal':>6} {'Δ':>7}")
    print("-" * 62)
    for i in sorted(HARD_INDICES, key=lambda i: CLASS_NAMES[i]):
        print(f"{SHORT_NAMES[i]:<40} {f1_ce[i]:>6.3f} {f1_focal[i]:>6.3f} {delta[i]:>+7.3f}")
    print(f"\nOverall top-1: CE={overall_ce:.4f}  Focal={overall_focal:.4f}  Δ={overall_focal-overall_ce:+.4f}")


if __name__ == "__main__":
    main()
