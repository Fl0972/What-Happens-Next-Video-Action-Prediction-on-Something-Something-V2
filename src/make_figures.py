#!/usr/bin/env python3
"""Generate every figure described in PRESENTATION_DATA.md / PRESENTATION_STATS.md.

Outputs PNGs to ../presentation_figures/. Run from this directory:

    python make_figures.py

The script is self-contained: it reads the cached softmax tensors and val
labels from the project, plus a few hardcoded scalars (Kaggle scores,
training curves, frame-fraction measurements) that come from the logs and
are documented in PRESENTATION_DATA.md.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless

import glob
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from create_submission import discover_all_test_videos
from dataset.video_dataset import _list_frame_paths, collect_video_samples

# ---------------------------------------------------------------------------
# Paths and style
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "models" / "_softmax_cache"
TRAIN = Path("/Data/florian.guillaumey/val2/train")
VAL_WIN = Path("/Data/florian.guillaumey/val2_win16/val")
TEST_DIR = Path("/Data/florian.guillaumey/val2/test")
PSEUDO_CSV = Path("/Data/florian.guillaumey/challenge_models/vjepa_pseudo.csv")
OUT = ROOT / "presentation_figures"
OUT.mkdir(exist_ok=True)

NUM_CLASSES = 33
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})
CLEAN = "#3a76b5"
LEAKY = "#c4453d"
HIGHLIGHT = "#e6a23c"
GREEN = "#3f8a3f"


def save(fig, name: str) -> None:
    p = OUT / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")


# ---------------------------------------------------------------------------
# Shared data loaders (cached softmax + labels)
# ---------------------------------------------------------------------------
def _class_names() -> Dict[int, str]:
    names = {}
    for d in sorted(TRAIN.iterdir()):
        if d.is_dir():
            idx = int(d.name.split("_", 1)[0])
            tail = d.name.split("_", 1)[1].replace("_", " ") if "_" in d.name else d.name
            names[idx] = tail
    for i in range(NUM_CLASSES):
        names.setdefault(i, f"class_{i:03d}")
    return names


def _train_counts() -> Dict[int, int]:
    c = {i: 0 for i in range(NUM_CLASSES)}
    for d in TRAIN.iterdir():
        if d.is_dir():
            idx = int(d.name.split("_", 1)[0])
            c[idx] = len([p for p in d.iterdir() if p.is_dir()])
    return c


def _find_cache(stem: str, split: str):
    fs = sorted(glob.glob(str(CACHE / f"{stem}_{split}_*.pt")))
    return torch.load(fs[-1], map_location="cpu").numpy() if fs else None


CLASS_NAMES = _class_names()
TRAIN_COUNTS = _train_counts()
VAL_SAMPLES = collect_video_samples(VAL_WIN)
VAL_LABELS = np.array([l for _, l in VAL_SAMPLES])
VAL_COUNTS = Counter(VAL_LABELS.tolist())


def short(i: int, n: int = 30) -> str:
    s = CLASS_NAMES[i]
    return s if len(s) <= n else s[: n - 1] + "…"


def verb_of(i: int) -> str:
    return CLASS_NAMES[i].split(" ", 1)[0].capitalize()


# Cached val/test softmax used by many figures
V_GROUPS = {
    "VideoMAE-Base K400 4f": [f"videomae_ovn1_k400_top{i}" for i in (1, 2, 3)],
    "VideoMAE-Base SSv2 4f": [f"videomae_ovn1_ssv2_top{i}" for i in (1, 2, 3)],
    "TSM-ResNet50 4f":       [f"tsm_r50_ovn1_top{i}" for i in (1, 2, 3)],
    "VideoMAE-Large 4f":     ["videomae_ovn2_large_attempt1_top2", "videomae_ovn2_large_attempt1_top3"],
}
FAMILY_VAL_SM: Dict[str, np.ndarray] = {}
for fam, stems in V_GROUPS.items():
    sms = [_find_cache(s, "val") for s in stems]
    sms = [s for s in sms if s is not None]
    if sms:
        FAMILY_VAL_SM[fam] = np.mean(sms, axis=0)

ALL_VAL_SMS = [s for sl in V_GROUPS.values() for s in (_find_cache(x, "val") for x in sl) if s is not None]
ENS_VAL_SM = np.mean(ALL_VAL_SMS, axis=0)
ENS_VAL_PREDS = ENS_VAL_SM.argmax(1)
ENS_VAL_ACC = (ENS_VAL_PREDS == VAL_LABELS).mean()


# ===========================================================================
# Figures
# ===========================================================================

def fig_A_kaggle_journey():
    """Bar chart of the honest journey + the discarded leaky bar."""
    rows = [
        ("V-JEPA (full video + ssv2)", 0.81, "leaky"),
        ("V-JEPA-ft (clean)", 0.6361, "clean"),
        ("Honest v1 (9-uniform)", 0.6418, "clean"),
        ("V-JEPA-pseudo seul", 0.6410, "clean"),
        ("V-JEPA-only (6 snap.)", 0.6410, "clean"),
        ("V-JEPA-heavy (3:1)", 0.6499, "clean"),
        ("Honest v3 (12-uniform)", 0.6546, "clean"),
        ("Honest v4 (14-uniform)", 0.6586, "clean"),
        ("Honest v5 (15-uniform) ★", 0.6592, "clean"),
    ]
    labels, scores, phases = zip(*rows)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [LEAKY if p == "leaky" else CLEAN for p in phases]
    bars = ax.bar(range(len(scores)), scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Kaggle Top-1")
    ax.set_title("Trajectoire honnête vers le score final\n(barre rouge = run écarté pour fuite de données)")
    ax.axhline(0.74, ls="--", color=GREEN, label="2e place du leaderboard")
    ax.axhline(0.64, ls=":", color="gray", label="baseline pré-V-JEPA")
    ax.set_ylim(0.55, 0.85)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.003, f"{s:.4f}", ha="center", fontsize=8)
    bars[0].set_hatch("///")
    ax.text(0, 0.81 + 0.012, "ÉCARTÉ", color=LEAKY, ha="center", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right")
    save(fig, "A_kaggle_journey")


def fig_B_ensemble_size_vs_score():
    rows = [
        (1, "V-JEPA-ft seul", 0.6361),
        (1, "V-JEPA-pseudo seul", 0.6410),
        (6, "V-JEPA-only", 0.6410),
        (9, "9-uniform (v1)", 0.6418),
        (12, "12-uniform (v3)", 0.6546),
        (14, "14-uniform (v4)", 0.6586),
        (15, "15-uniform (v5)", 0.6592),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for n, lbl, s in rows:
        ax.scatter(n, s, s=140, color=CLEAN, edgecolor="black", zorder=3)
        ax.text(n + 0.15, s, lbl, va="center", fontsize=8)
    ax.set_xlabel("Nombre de modèles dans l'ensemble")
    ax.set_ylabel("Kaggle Top-1")
    ax.set_title("Ensemble size vs Kaggle — \"diversity wins\"")
    ax.set_xlim(0, 17)
    ax.set_ylim(0.63, 0.67)
    ax.annotate("perd de la précision\nen retirant k400+TSM",
                xy=(6, 0.6410), xytext=(2, 0.633), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))
    save(fig, "B_ensemble_size_vs_score")


def fig_C_weighting_strategies():
    rows = [
        ("V-JEPA-only\n6 modèles uniformes", 0.6410, "alt"),
        ("V-JEPA-heavy 3:1\n12 modèles pondérés", 0.6499, "alt"),
        ("9-uniform", 0.6418, "uniform"),
        ("12-uniform", 0.6546, "uniform"),
        ("14-uniform", 0.6586, "uniform"),
        ("15-uniform", 0.6592, "uniform"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [HIGHLIGHT if t == "alt" else CLEAN for *_, t in rows]
    bars = ax.barh(range(len(rows)), [r[1] for r in rows], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Kaggle Top-1")
    ax.set_xlim(0.63, 0.67)
    ax.set_title("Pondération vs uniforme — l'uniforme bat les variantes pondérées")
    for b, (_, s, _) in zip(bars, rows):
        ax.text(s + 0.0005, b.get_y() + b.get_height() / 2, f"{s:.4f}", va="center", fontsize=8)
    ax.legend(handles=[Patch(color=HIGHLIGHT, label="Variante pondérée"),
                       Patch(color=CLEAN, label="Uniforme")], loc="lower right")
    save(fig, "C_weighting_strategies")


def fig_D_training_curves():
    stages = [
        ("Frozen probe (V-JEPA SSL)", [0.5755, 0.6203, 0.6386, 0.6476, 0.6635, 0.6725], "#7fa8c4"),
        ("FT stage 1 (top-6 + LLRD)", [0.6764, 0.6914, 0.7006, 0.7052], "#3a76b5"),
        ("FT stage 2 (full + LLRD)",  [0.7015, 0.7184, 0.7267, 0.7281], "#0e3d6a"),
        ("Pseudo-label retrain",       [0.7180, 0.7273, 0.7370, 0.7415], "#e6a23c"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_cursor = 0
    for label, ys, color in stages:
        xs = list(range(x_cursor + 1, x_cursor + 1 + len(ys)))
        ax.plot(xs, ys, "o-", color=color, label=label, linewidth=2, markersize=6)
        x_cursor = xs[-1]
    ax.set_xlabel("Époque (cumulée)")
    ax.set_ylabel("EMA val accuracy (train-split)")
    ax.set_title("Courbes de finetuning V-JEPA — déblocage progressif + LLRD + pseudo-labels")
    ax.legend(loc="lower right")
    save(fig, "D_training_curves")


def fig_E_frame_position_histogram():
    fracs = [0.16, 0.37, 0.39, 0.40, 0.40, 0.40, 0.40, 0.40, 0.45, 0.46, 0.48, 0.51, 0.52, 0.90]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(fracs, bins=np.arange(0, 1.01, 0.05), color=CLEAN, edgecolor="black")
    ax.axvspan(0.50, 1.0, alpha=0.25, color=LEAKY, label="Zone INTERDITE\n(outcome retenu par les organisateurs)")
    ax.axvline(0.40, ls="--", color="black", label="Médiane = 0.40")
    ax.set_xlabel("Fraction de la vidéo source où se trouve frame_003 (dernière frame fournie)")
    ax.set_ylabel("# clips (mesure sur 14 échantillons)")
    ax.set_title("La compétition fournit ~40% de la vidéo — pas 60% comme indicatif")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right")
    save(fig, "E_frame_position_histogram")


def fig_F_mapping_verification():
    same = (28.3, 29.1, 70.6)
    rand = (67.6, 69.4, 38.6)
    labels = ["SAME id\n(20bn-v2/<id>.webm)", "RANDOM id"]
    medians = [same[0], rand[0]]
    means = [same[1], rand[1]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, medians, w, label="médiane", color=CLEAN)
    ax.bar(x + w / 2, means, w, label="moyenne", color=HIGHLIGHT)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Abs Error (échelle 0..255)")
    ax.set_title("Mapping ID vérifié — SAME ≪ RANDOM\n(diagnostic discriminant sur 40 clips)")
    for i, v in enumerate(medians): ax.text(i - w / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    for i, v in enumerate(means):   ax.text(i + w / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax.legend()
    save(fig, "F_mapping_verification")


def fig_G_calibration_train_val_kaggle():
    models = ["V-JEPA-ft", "V-JEPA-pseudo"]
    train_split = [0.7281, 0.7415]
    val_dir = [0.6411, 0.6519]
    kaggle = [0.6361, 0.6410]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    w = 0.25
    ax.bar(x - w, train_split, w, label="train-split (interne)", color="#aac7e0")
    ax.bar(x, val_dir, w, label="val_dir (honnête)", color=CLEAN)
    ax.bar(x + w, kaggle, w, label="Kaggle (réel)", color="#0e3d6a")
    for i, v in enumerate(train_split): ax.text(x[i] - w, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(val_dir):     ax.text(x[i], v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(kaggle):      ax.text(x[i] + w, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy Top-1")
    ax.set_title("Calibration : l'écart train-split / val_dir / Kaggle se creuse avec les pseudo-labels")
    ax.legend()
    ax.set_ylim(0.60, 0.80)
    save(fig, "G_calibration_train_val_kaggle")


def fig_H_single_model_val_dir():
    # measurement on the 4-frame val pipeline (per CHANGES + gradient ensemble logs)
    rows = [
        ("V-JEPA-pseudo (16f win-cap, top-3 + TTA)", 0.6519, "vjepa"),
        ("V-JEPA-ft (16f win-cap, top-3 + TTA)", 0.6411, "vjepa"),
        ("videomae_ovn1_ssv2_top1", 0.6059, "vmae_ssv2"),
        ("videomae_ovn2_large_attempt1_top2", 0.5999, "vmae_l"),
        ("videomae_ovn1_ssv2_top2", 0.5956, "vmae_ssv2"),
        ("videomae_ovn1_ssv2_top3", 0.5927, "vmae_ssv2"),
        ("videomae_ovn2_large_attempt1_top3", 0.5686, "vmae_l"),
        ("videomae_ovn1_k400_top2", 0.5566, "k400"),
        ("videomae_ovn1_k400_top1", 0.5543, "k400"),
        ("videomae_ovn1_k400_top3", 0.5537, "k400"),
    ]
    colors_map = {"vjepa": "#0e3d6a", "vmae_ssv2": "#3a76b5", "vmae_l": "#e6a23c", "k400": "#9ec5e8"}
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(rows)), [r[1] for r in rows],
                    color=[colors_map[r[2]] for r in rows], edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("val_dir Top-1")
    ax.set_xlim(0.5, 0.7)
    ax.set_title("Précision par modèle individuel sur val_dir")
    for b, r in zip(bars, rows):
        ax.text(r[1] + 0.002, b.get_y() + b.get_height()/2, f"{r[1]:.4f}", va="center", fontsize=8)
    save(fig, "H_single_model_val_dir")


def fig_I_learned_weights_collapse():
    rows = [
        ("large_attempt1_top2", 0.4286),
        ("ssv2_top1", 0.3922),
        ("ssv2_top2", 0.1363),
        ("ssv2_top3", 0.0168),
        ("k400_top2", 0.0151),
        ("k400_top1", 0.0049),
        ("k400_top3", 0.0048),
        ("large_attempt1_top3", 0.0013),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(rows)), [r[1] for r in rows], color=CLEAN, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right")
    ax.axhline(1/len(rows), ls="--", color="gray", label=f"uniforme = {1/len(rows):.3f}")
    ax.set_ylabel("Poids appris (softmax-normalisé)")
    ax.set_title("Les poids appris par gradient descent se concentrent sur les 2 meilleurs modèles")
    for b, (_, v) in zip(bars, rows):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend()
    save(fig, "I_learned_weights_collapse")


def fig_J_class_imbalance():
    counts = [(c, TRAIN_COUNTS[c], VAL_COUNTS.get(c, 0)) for c in range(NUM_CLASSES)]
    counts.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(counts))
    ax.bar(x, [r[1] for r in counts], color=CLEAN, edgecolor="black", linewidth=0.3, label="train")
    ax.set_ylabel("# clips d'entraînement", color=CLEAN)
    ax2 = ax.twinx()
    ax2.plot(x, [r[2] for r in counts], "o-", color=HIGHLIGHT, label="val", markersize=4)
    ax2.set_ylabel("# clips val", color=HIGHLIGHT)
    ax.set_xticks(x); ax.set_xticklabels([f"{c:03d}" for c, _, _ in counts], rotation=90, fontsize=7)
    ax.set_xlabel("Classe (triée par # train, décroissant)")
    ax.set_title("Déséquilibre des classes — ratio max/min ≈ 30×, classe 027 absente du train")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "J_class_imbalance")


def fig_K_per_family_val_acc():
    fams = list(FAMILY_VAL_SM.keys())
    accs = [(FAMILY_VAL_SM[f].argmax(1) == VAL_LABELS).mean() for f in fams]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(fams)), accs, color=CLEAN, edgecolor="black", linewidth=0.4)
    bars2 = ax.bar(len(fams), ENS_VAL_ACC, color=GREEN, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(fams) + 1)); ax.set_xticklabels(fams + ["Ensemble\nuniforme"], rotation=15, ha="right")
    ax.set_ylabel("val_dir Top-1 (single-view)")
    ax.set_title("Aucune famille seule n'atteint l'ensemble")
    for b, a in zip(list(bars) + list(bars2), accs + [ENS_VAL_ACC]):
        ax.text(b.get_x() + b.get_width()/2, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
    save(fig, "K_per_family_val_acc")


def fig_L_per_class_lift():
    fam_pc = {fam: {c: ((sm.argmax(1)[VAL_LABELS == c] == c).mean()
                        if VAL_COUNTS.get(c, 0) > 0 else float("nan"))
                    for c in range(NUM_CLASSES)} for fam, sm in FAMILY_VAL_SM.items()}
    ens_pc = {c: ((ENS_VAL_PREDS[VAL_LABELS == c] == c).mean() if VAL_COUNTS.get(c, 0) > 0 else float("nan"))
              for c in range(NUM_CLASSES)}
    rows = []
    for c in range(NUM_CLASSES):
        if VAL_COUNTS.get(c, 0) == 0: continue
        best = max((fam_pc[f][c] for f in fam_pc if not np.isnan(fam_pc[f][c])), default=0)
        rows.append((c, ens_pc[c] - best))
    rows.sort(key=lambda r: -r[1])
    top = rows[:10] + rows[-5:]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [GREEN if r[1] > 0 else LEAKY for r in top]
    bars = ax.barh(range(len(top)), [r[1] for r in top], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([f"{c:03d}  {short(c, 28)}" for c, _ in top])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ accuracy (ensemble − meilleure famille seule)")
    ax.set_title("Où l'ensemble aide le plus — et où il n'aide pas")
    save(fig, "L_per_class_lift")
    return rows


def fig_N_confusion_matrix():
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(VAL_LABELS, ENS_VAL_PREDS):
        cm[t, p] += 1
    # row-normalise so rare classes are visible
    row_sums = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels([f"{c:03d}" for c in range(NUM_CLASSES)], rotation=90, fontsize=7)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels([f"{c:03d}" for c in range(NUM_CLASSES)], fontsize=7)
    ax.set_xlabel("Classe prédite"); ax.set_ylabel("Vraie classe")
    ax.set_title("Matrice de confusion 33×33 (normalisée par ligne)\nEnsemble uniforme sur val_dir")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    save(fig, "N_confusion_matrix_33x33")


def fig_O_semantic_groups():
    group_classes = defaultdict(list)
    for c in range(NUM_CLASSES):
        group_classes[verb_of(c)].append(c)
    rows = []
    for grp, cs in group_classes.items():
        n_val = sum(VAL_COUNTS.get(c, 0) for c in cs)
        if n_val == 0: continue
        correct = sum((ENS_VAL_PREDS[VAL_LABELS == c] == c).sum() for c in cs if VAL_COUNTS.get(c, 0) > 0)
        rows.append((grp, len(cs), n_val, correct / n_val))
    rows.sort(key=lambda r: -r[3])
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(rows)), [r[3] for r in rows], color=CLEAN, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"{r[0]}\n({r[1]} cl., n={r[2]})" for r in rows], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy ensemble")
    ax.set_title("Précision par groupe sémantique (verbe d'action)")
    ax.set_ylim(0, 1)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width()/2, r[3] + 0.01, f"{r[3]:.2f}", ha="center", fontsize=7)
    ax.axhline(ENS_VAL_ACC, ls="--", color="red", label=f"Moyenne globale = {ENS_VAL_ACC:.3f}")
    ax.legend()
    save(fig, "O_semantic_groups")


def fig_P_top1_vs_top5():
    top5 = np.argsort(-ENS_VAL_SM, axis=1)[:, :5]
    in_top5 = (top5 == VAL_LABELS[:, None]).any(axis=1)
    rows = []
    for c in range(NUM_CLASSES):
        if VAL_COUNTS.get(c, 0) == 0: continue
        mask = VAL_LABELS == c
        t1 = (ENS_VAL_PREDS[mask] == c).mean()
        t5 = in_top5[mask].mean()
        rows.append((c, t1, t5))
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [r[1] for r in rows]; ys = [r[2] for r in rows]
    ax.scatter(xs, ys, s=70, color=CLEAN, edgecolor="black", linewidth=0.4, alpha=0.8)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="y = x")
    ax.set_xlabel("Top-1 accuracy par classe"); ax.set_ylabel("Top-5 accuracy par classe")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Top-1 vs Top-5 par classe — récupération du top-5 est ~constante (~+0.26)")
    # annotate biggest top-5 recoveries
    rows.sort(key=lambda r: -(r[2] - r[1]))
    for c, t1, t5 in rows[:4]:
        ax.annotate(f"{c:03d}", (t1, t5), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.legend()
    save(fig, "P_top1_vs_top5")


def fig_Q_family_wins():
    """Bar par classe, colorée par la famille qui a la meilleure précision sur cette classe."""
    fam_pc = {fam: {c: ((sm.argmax(1)[VAL_LABELS == c] == c).mean()
                        if VAL_COUNTS.get(c, 0) > 0 else float("nan"))
                    for c in range(NUM_CLASSES)} for fam, sm in FAMILY_VAL_SM.items()}
    fams = list(FAMILY_VAL_SM.keys())
    fam_colors = {f: c for f, c in zip(fams, ["#0e3d6a", "#3a76b5", "#e6a23c", "#9ec5e8"])}
    winners: Dict[int, str] = {}
    win_acc: Dict[int, float] = {}
    for c in range(NUM_CLASSES):
        if VAL_COUNTS.get(c, 0) == 0: continue
        accs = {f: fam_pc[f][c] for f in fams if not np.isnan(fam_pc[f][c])}
        if not accs: continue
        best_f = max(accs, key=accs.get)
        winners[c] = best_f
        win_acc[c] = accs[best_f]

    classes_with_win = sorted(winners.keys())
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bar_colors = [fam_colors[winners[c]] for c in classes_with_win]
    bars = ax.bar(range(len(classes_with_win)), [win_acc[c] for c in classes_with_win],
                  color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(classes_with_win)))
    ax.set_xticklabels([f"{c:03d}" for c in classes_with_win], rotation=90, fontsize=8)
    ax.set_ylabel("Précision de la famille gagnante sur cette classe (val_dir)")
    ax.set_xlabel("Classe (n° à 3 chiffres ; nom complet en annexe)")
    counts = Counter(winners.values())
    title_parts = [f"{f} : {counts[f]}" for f in fams if counts[f] > 0]
    ax.set_title("Spécialisation par famille — pour chaque classe, la famille qui gagne\n"
                 f"({' · '.join(title_parts)})")
    ax.set_ylim(0, 1)
    legend_handles = [Patch(color=fam_colors[f], label=f"{f} ({counts[f]} classes)") for f in fams if counts[f] > 0]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    # Annotation : pour chaque famille, lister les n° de classes gagnées (à droite)
    txt_lines = []
    for f in fams:
        won = sorted([c for c, w in winners.items() if w == f])
        if not won: continue
        cls_str = ", ".join(f"{c:03d}" for c in won)
        txt_lines.append(f"{f} ({counts[f]}): {cls_str}")
    ax.text(1.005, 0.5, "\n\n".join(txt_lines), transform=ax.transAxes, va="center",
            ha="left", fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    fig.subplots_adjust(right=0.72)
    save(fig, "Q_family_wins")


def fig_R_size_vs_accuracy():
    xs, ys, cs = [], [], []
    for c in range(NUM_CLASSES):
        if VAL_COUNTS.get(c, 0) == 0 or TRAIN_COUNTS[c] == 0: continue
        mask = VAL_LABELS == c
        acc = (ENS_VAL_PREDS[mask] == c).mean()
        xs.append(TRAIN_COUNTS[c]); ys.append(acc); cs.append(c)
    xs, ys = np.array(xs), np.array(ys)
    r = np.corrcoef(xs, ys)[0, 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=60, color=CLEAN, edgecolor="black", linewidth=0.4, alpha=0.8)
    z = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(xline, z[0] * xline + z[1], "--", color="red", label=f"régression linéaire (r = {r:.3f})")
    ax.set_xlabel("# clips train de la classe"); ax.set_ylabel("Accuracy ensemble (val)")
    ax.set_title("Taille de la classe ≠ explication principale de la précision")
    # annotate a few outliers
    for x, y, c in zip(xs, ys, cs):
        if y > 0.85 or y < 0.3:
            ax.annotate(f"{c:03d}", (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.legend()
    save(fig, "R_size_vs_accuracy")


def fig_S_agreement_intra_inter():
    """Matrice d'accord pair-à-pair, version lisible : labels plus grands, x rotés 45°, blocs familles visibles."""
    test_stems = [
        *[f"vjepa_ft_s2_top{i}" for i in (1, 2, 3)],
        *[f"vjepa_ft_s2_pseudo_top{i}" for i in (1, 2, 3)],
        *[f"videomae_ovn1_k400_top{i}" for i in (1, 2, 3)],
        *[f"tsm_r50_ovn1_top{i}" for i in (1, 2, 3)],
        "videomae_ovn2_large_attempt1_top2", "videomae_ovn2_large_attempt1_top3",
        "vmae_l_k400_win_top1",
    ]
    # short pretty labels (groupes visibles d'un coup d'œil)
    pretty = {
        "vjepa_ft_s2_top1": "V-JEPA-ft #1",
        "vjepa_ft_s2_top2": "V-JEPA-ft #2",
        "vjepa_ft_s2_top3": "V-JEPA-ft #3",
        "vjepa_ft_s2_pseudo_top1": "V-JEPA-pseudo #1",
        "vjepa_ft_s2_pseudo_top2": "V-JEPA-pseudo #2",
        "vjepa_ft_s2_pseudo_top3": "V-JEPA-pseudo #3",
        "videomae_ovn1_k400_top1": "VMAE-Base K400 #1",
        "videomae_ovn1_k400_top2": "VMAE-Base K400 #2",
        "videomae_ovn1_k400_top3": "VMAE-Base K400 #3",
        "tsm_r50_ovn1_top1": "TSM R50 #1",
        "tsm_r50_ovn1_top2": "TSM R50 #2",
        "tsm_r50_ovn1_top3": "TSM R50 #3",
        "videomae_ovn2_large_attempt1_top2": "VMAE-Large 4f #1",
        "videomae_ovn2_large_attempt1_top3": "VMAE-Large 4f #2",
        "vmae_l_k400_win_top1": "VMAE-L 16f-win",
    }
    sms = {s: _find_cache(s, "test") for s in test_stems}
    sms = {k: v for k, v in sms.items() if v is not None}
    preds = {k: v.argmax(1) for k, v in sms.items()}
    keys = list(preds.keys())
    labels = [pretty.get(k, k) for k in keys]
    n = len(keys)
    agree = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a = (preds[keys[i]] == preds[keys[j]]).mean()
            agree[i, j] = agree[j, i] = a

    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(agree, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=15)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=15)

    # group separators (lignes noires entre familles)
    boundaries = [3, 6, 9, 12, 14]  # après VFT, VPS, K400, TSM, Large-4f
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", linewidth=2)
        ax.axvline(b - 0.5, color="black", linewidth=2)

    ax.set_title("Matrice d'accord pair-à-pair sur le test\n"
                 "Vert = forte corrélation (intra-famille) · Rouge = forte diversité (inter-familles)",
                 fontsize=14, pad=15)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Proportion de prédictions identiques sur le test", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    save(fig, "S_agreement_matrix")


def fig_AA_augmentations():
    """Tableau visuel catégorisé de toutes les techniques de data augmentation utilisées."""
    sections = [
        ("Spatiales (par frame, temporellement consistantes)", "#3a76b5", [
            ("RandomResizedCrop",  "scale ∈ [0.5, 1.0], ratio ∈ [3/4, 4/3]",  "Recadrage + redim. aléatoire"),
            ("HorizontalFlip",     "p = 0.5",                                  "Miroir horizontal (cf. label-aware)"),
            ("Rotation",           "±15°",                                     "Rotation aléatoire"),
            ("Color jitter — brightness", "×0.6 à ×1.4",                       "Variation de luminosité"),
            ("Color jitter — contrast",   "×0.6 à ×1.4",                       "Variation de contraste"),
            ("Color jitter — saturation", "×0.6 à ×1.4",                       "Variation de saturation"),
            ("Color jitter — hue",        "±0.1",                              "Variation de teinte"),
            ("RandomGrayscale",    "p = 0.1",                                  "Conversion N&B (3 canaux)"),
            ("GaussianBlur",       "p = 0.2, σ ∈ [0.1, 2.0]",                  "Flou gaussien"),
            ("RandomErase",        "p = 0.3, taille 2–33 %",                   "Masque rectangulaire (Cutout)"),
        ]),
        ("Temporelles", "#9b6db8", [
            ("TSN-style temporal jitter", "stratifié + offset random",         "Échantillonnage de N frames par segments + offset"),
            ("Multi-clip sampling",       "n_clips ≥ 2 (optionnel)",           "Plusieurs sous-échantillonnages du même clip"),
        ]),
        ("Batch-level (mélange entre échantillons)", "#e6a23c", [
            ("MixUp",              "α = 0.4 (Beta)",                           "Combinaison linéaire de 2 clips + labels"),
            ("CutMix",             "p = 0.5",                                  "Patch d'un autre clip, label mixé"),
            ("Label smoothing",    "0.1",                                      "Régularisation CE (anti-overfit)"),
        ]),
        ("Labels (semi-supervisé)", "#3f8a3f", [
            ("Label-aware HFlip",  "1 paire miroir (018 ↔ 019)",               "Quand on flippe, on échange aussi le label"),
            ("Pseudo-labeling",    "seuil conf ≥ 0.85",                        "Test set ajouté au train avec labels V-JEPA"),
        ]),
        ("TTA — inférence (× vues)", "#c4453d", [
            ("Spatial 10-crop",    "×10 (5 positions × 2 flips)",              "Logits moyennés sur 10 vues spatiales"),
            ("Flip-aware logit remap", "intégré au 10-crop",                   "Logits des vues flippées re-permutés (label-aware)"),
            ("Multi-clip TTA",     "×n_clips",                                 "Plusieurs échantillonnages temporels"),
        ]),
    ]

    # Layout
    n_rows = sum(1 + len(items) for _, _, items in sections)
    fig_h = max(11, 0.42 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, n_rows + 1.5); ax.invert_yaxis(); ax.axis("off")

    cols = [
        ("Technique",   0.02, 0.32, "left"),
        ("Paramètre",   0.32, 0.55, "left"),
        ("Description", 0.55, 0.98, "left"),
    ]

    # Header
    for label, x0, x1, ha in cols:
        ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 0.9, facecolor="#2c3e50", edgecolor="white"))
        tx = x0 + 0.01 if ha == "left" else (x0 + x1) / 2
        ax.text(tx, 0.45, label, color="white", ha=ha, va="center",
                fontsize=13, fontweight="bold")

    y = 1.0
    for section_title, section_color, items in sections:
        # Section header row
        ax.add_patch(plt.Rectangle((0, y), 1, 1.1, facecolor=section_color, edgecolor="white"))
        ax.text(0.015, y + 0.55, section_title, color="white", ha="left", va="center",
                fontsize=13, fontweight="bold", style="italic")
        y += 1.1
        # Items
        for i, (tech, param, desc) in enumerate(items):
            bg = "#fafafa" if i % 2 == 0 else "white"
            ax.add_patch(plt.Rectangle((0, y), 1, 1, facecolor=bg, edgecolor="none"))
            # technique name (left col, bold)
            ax.text(cols[0][1] + 0.01, y + 0.5, tech, ha="left", va="center",
                    fontsize=11, fontweight="bold", color="#1a1a1a")
            # parameter (middle col, monospace-ish)
            ax.text(cols[1][1] + 0.01, y + 0.5, param, ha="left", va="center",
                    fontsize=10, color="#444", family="monospace")
            # description (right col)
            ax.text(cols[2][1] + 0.01, y + 0.5, desc, ha="left", va="center",
                    fontsize=10, color="#333")
            y += 1.0

    fig.suptitle("Techniques de data augmentation utilisées dans le pipeline",
                 fontsize=14, fontweight="bold", y=0.997)
    fig.tight_layout()
    save(fig, "AA_augmentations")


def fig_Z_convergence_curves():
    """Vraies courbes de convergence (train + val loss/acc) extraites des logs."""
    import re
    rx = re.compile(
        r"Epoch (\d+)/(\d+) \| train loss ([\d.]+) acc ([\d.]+) "
        r"\| val loss ([\d.]+) acc ([\d.]+) "
        r"\| ema val loss ([\d.]+) acc ([\d.]+)"
    )
    def parse(p: Path):
        if not p.is_file(): return []
        rows = []
        for line in p.read_text(errors="ignore").splitlines():
            m = rx.search(line)
            if m:
                rows.append({
                    "ep": int(m.group(1)), "total": int(m.group(2)),
                    "tr_loss": float(m.group(3)), "tr_acc": float(m.group(4)),
                    "val_loss": float(m.group(5)), "val_acc": float(m.group(6)),
                    "ema_loss": float(m.group(7)), "ema_acc": float(m.group(8)),
                })
        return rows

    LOGS = ROOT / "logs"
    stages = [
        ("Stage 1 (top-6 + LLRD)",         parse(LOGS / "vjepaft_ft_stage1.log"),       "#7fa8c4"),
        ("Stage 2 (full + LLRD)",          parse(LOGS / "vjepaft_ft_stage2.log"),       "#0e3d6a"),
        ("Pseudo-label retrain (full)",    parse(LOGS / "vjepapseudo_ft_pseudo.log"),   "#e6a23c"),
    ]

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    cumul = 0
    for label, rows, color in stages:
        if not rows: continue
        xs = list(range(cumul + 1, cumul + 1 + len(rows)))
        # LOSS (top)
        ax_loss.plot(xs, [r["tr_loss"] for r in rows], "o-", color=color, linewidth=2,
                     label=f"{label} — train", alpha=0.55)
        ax_loss.plot(xs, [r["val_loss"] for r in rows], "s--", color=color, linewidth=2,
                     label=f"{label} — val")
        # ACCURACY (bottom)
        ax_acc.plot(xs, [r["tr_acc"] for r in rows], "o-", color=color, linewidth=2,
                    label=f"{label} — train", alpha=0.55)
        ax_acc.plot(xs, [r["val_acc"] for r in rows], "s--", color=color, linewidth=2,
                    label=f"{label} — val")
        ax_acc.plot(xs, [r["ema_acc"] for r in rows], "D:", color=color, linewidth=2,
                    label=f"{label} — EMA val", markersize=5)
        cumul = xs[-1]
        # stage boundary line
        ax_loss.axvline(xs[-1] + 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax_acc.axvline(xs[-1] + 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Courbes de convergence — V-JEPA finetune en 3 étapes\n"
                      "Train (rond plein) vs Val (carré, pointillé) vs EMA val (losange, points)",
                      fontsize=12, pad=10)
    ax_loss.legend(loc="upper right", fontsize=8, ncol=2)
    ax_loss.grid(alpha=0.3)

    ax_acc.set_ylabel("Accuracy Top-1")
    ax_acc.set_xlabel("Époque (cumulée)")
    ax_acc.legend(loc="lower right", fontsize=8, ncol=2)
    ax_acc.grid(alpha=0.3)
    ax_acc.set_ylim(0.20, 0.80)

    # annotate the train-val gap on accuracy (overfit signal)
    # use the last point of pseudo as illustration
    if stages[-1][1]:
        last = stages[-1][1][-1]
        x = cumul
        ax_acc.annotate(
            f"écart train↔val\n= {last['val_acc']-last['tr_acc']:+.3f} pp",
            xy=(x, (last['val_acc'] + last['tr_acc']) / 2),
            xytext=(x - 4, 0.78), fontsize=9,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))

    fig.tight_layout()
    save(fig, "Z_convergence_curves")


def fig_T_pseudo_coverage():
    if not PSEUDO_CSV.exists(): return
    import csv
    with PSEUDO_CSV.open() as f:
        r = csv.reader(f); next(r)
        rows = [row for row in r if len(row) >= 2]
    counts = Counter(int(row[1]) for row in rows)
    n_total = len(rows)
    bars = [counts.get(c, 0) for c in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = [LEAKY if v == 0 else CLEAN for v in bars]
    ax.bar(range(NUM_CLASSES), bars, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels([f"{c:03d}" for c in range(NUM_CLASSES)], rotation=90, fontsize=7)
    ax.set_xlabel("Classe"); ax.set_ylabel("# clips test pseudo-étiquetés")
    n_zero = sum(1 for v in bars if v == 0)
    ax.set_title(f"Couverture des pseudo-labels par classe — {n_zero}/{NUM_CLASSES} classes jamais prédites (rouge)")
    save(fig, "T_pseudo_label_coverage")


def fig_U_confidence_histogram():
    test_stems = [
        *[f"vjepa_ft_s2_top{i}" for i in (1, 2, 3)],
        *[f"vjepa_ft_s2_pseudo_top{i}" for i in (1, 2, 3)],
        *[f"videomae_ovn1_k400_top{i}" for i in (1, 2, 3)],
        *[f"tsm_r50_ovn1_top{i}" for i in (1, 2, 3)],
        "videomae_ovn2_large_attempt1_top2", "videomae_ovn2_large_attempt1_top3",
        "vmae_l_k400_win_top1",
    ]
    sms = [_find_cache(s, "test") for s in test_stems]
    sms = [s for s in sms if s is not None]
    ens = np.mean(sms, axis=0)
    conf = ens.max(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(conf, bins=np.arange(0, 1.01, 0.05), color=CLEAN, edgecolor="black")
    ax.axvline(conf.mean(), ls="--", color="red", label=f"moyenne = {conf.mean():.3f}")
    ax.axvline(np.median(conf), ls=":", color="orange", label=f"médiane = {np.median(conf):.3f}")
    ax.axvline(1/NUM_CLASSES, ls=":", color="gray", label=f"uniforme = 1/33 = {1/NUM_CLASSES:.3f}")
    ax.set_xlabel("Confiance max (softmax) de l'ensemble par clip")
    ax.set_ylabel("# clips")
    ax.set_title("Distribution de la confiance de l'ensemble sur le test")
    ax.legend()
    save(fig, "U_confidence_histogram")


def fig_V_leave_one_out():
    all_stems = [s for sl in V_GROUPS.values() for s in sl]
    rows = []
    for stem in all_stems:
        others = [_find_cache(s, "val") for s in all_stems if s != stem]
        others = [o for o in others if o is not None]
        if not others: continue
        sm = np.mean(others, axis=0)
        acc = (sm.argmax(1) == VAL_LABELS).mean()
        rows.append((stem, ENS_VAL_ACC - acc))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [GREEN if r[1] > 0 else LEAKY for r in rows]
    bars = ax.barh(range(len(rows)), [r[1] for r in rows], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ accuracy (ensemble complet − leave-one-out)")
    ax.set_title("Valeur marginale de chaque snapshot — Δ > 0 = il apporte vraiment")
    save(fig, "V_leave_one_out")


def fig_Y_class_table():
    """Tableau visuel des 33 classes avec gradients de couleur — pour la slide annexe.
    Trié par précision de l'ensemble (plus facile → plus dur)."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    # data per class: idx, name, train, val, acc
    rows = []
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        train_n = TRAIN_COUNTS[c]
        val_n = VAL_COUNTS.get(c, 0)
        if val_n > 0:
            mask = VAL_LABELS == c
            acc = (ENS_VAL_PREDS[mask] == c).mean()
        else:
            acc = float("nan")
        rows.append((c, name, train_n, val_n, acc))
    # sort: missing acc at the bottom, otherwise by acc desc
    rows.sort(key=lambda r: (0 if np.isnan(r[4]) else 1, -r[4] if not np.isnan(r[4]) else 0, r[0]))

    # color maps
    max_train = max(r[2] for r in rows) or 1
    train_norm = Normalize(vmin=0, vmax=max_train)
    train_cmap = LinearSegmentedColormap.from_list("blues_light", ["#f5f9fc", "#3a76b5"])
    acc_norm = Normalize(vmin=0.0, vmax=1.0)
    acc_cmap = LinearSegmentedColormap.from_list("redgreen", ["#d4382c", "#f2c94c", "#3f8a3f"])

    n = len(rows)
    fig_h = max(10, 0.35 * n + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, n + 1.2); ax.invert_yaxis(); ax.axis("off")

    # Column positions (proportions)
    cols = [
        ("N°",       0.02, 0.07, "center"),
        ("Classe",   0.07, 0.62, "left"),
        ("Train",    0.62, 0.72, "center"),
        ("Val",      0.72, 0.80, "center"),
        ("Acc ens.", 0.80, 0.92, "center"),
    ]
    # Header row
    header_y = 0.4
    for label, x0, x1, ha in cols:
        ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 0.9, facecolor="#2c3e50", edgecolor="white"))
        ax.text((x0 + x1) / 2, header_y, label, color="white", ha="center", va="center",
                fontsize=11, fontweight="bold")

    # Body rows
    for i, (c, name, tr, vl, acc) in enumerate(rows):
        y = 1 + i
        # zebra background (light)
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y), 1, 1, facecolor="#fafafa", edgecolor="none"))
        # N°
        ax.add_patch(plt.Rectangle((cols[0][1], y), cols[0][2] - cols[0][1], 1,
                                   facecolor="#eef2f7", edgecolor="white"))
        ax.text((cols[0][1] + cols[0][2]) / 2, y + 0.5, f"{c:03d}",
                ha="center", va="center", fontsize=10, fontweight="bold")
        # Name
        ax.text(cols[1][1] + 0.005, y + 0.5,
                name if len(name) <= 60 else name[:59] + "…",
                ha="left", va="center", fontsize=9)
        # Train (gradient)
        tcol = train_cmap(train_norm(tr))
        ax.add_patch(plt.Rectangle((cols[2][1], y), cols[2][2] - cols[2][1], 1,
                                   facecolor=tcol, edgecolor="white"))
        ax.text((cols[2][1] + cols[2][2]) / 2, y + 0.5,
                str(tr), ha="center", va="center", fontsize=9,
                color="white" if tr > max_train * 0.55 else "black")
        # Val
        vcol = train_cmap(train_norm(vl * (max_train / max(1, max(r[3] for r in rows)))))
        ax.add_patch(plt.Rectangle((cols[3][1], y), cols[3][2] - cols[3][1], 1,
                                   facecolor=vcol, edgecolor="white"))
        ax.text((cols[3][1] + cols[3][2]) / 2, y + 0.5,
                str(vl), ha="center", va="center", fontsize=9,
                color="white" if vl > max(r[3] for r in rows) * 0.55 else "black")
        # Accuracy (red→green gradient)
        if np.isnan(acc):
            ax.add_patch(plt.Rectangle((cols[4][1], y), cols[4][2] - cols[4][1], 1,
                                       facecolor="#e0e0e0", edgecolor="white"))
            ax.text((cols[4][1] + cols[4][2]) / 2, y + 0.5,
                    "—", ha="center", va="center", fontsize=9, fontstyle="italic", color="#666")
        else:
            ax.add_patch(plt.Rectangle((cols[4][1], y), cols[4][2] - cols[4][1], 1,
                                       facecolor=acc_cmap(acc_norm(acc)), edgecolor="white"))
            ax.text((cols[4][1] + cols[4][2]) / 2, y + 0.5,
                    f"{acc:.2f}", ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if abs(acc - 0.5) > 0.3 else "black")

    fig.suptitle("Les 33 classes du challenge — triées par précision de l'ensemble\n"
                 "(couleur train/val = volume · couleur accuracy = rouge→vert)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "Y_class_table")


def fig_W_resilience_timeline():
    """Show real wall-clock progression of two training runs across kill/restart events.
    Data: systemd 'loop_until_done: attempt N' timestamps + checkpoint file mtimes."""
    import re
    from datetime import datetime
    import matplotlib.dates as mdates

    LOGS = ROOT / "logs"
    LM = Path("/Data/florian.guillaumey/challenge_models")

    def parse_attempts(systemd_log):
        if not systemd_log.is_file(): return []
        out = []
        rx = re.compile(r"^\[(.*?)\] loop_until_done: attempt (\d+)/")
        for line in systemd_log.read_text(errors="ignore").splitlines():
            m = rx.match(line)
            if not m: continue
            try:
                dt = datetime.strptime(re.sub(r"\s+(AM|PM)\s+CEST\s+\d{4}",
                                              lambda x: " " + x.group(1) + " 2026",
                                              m.group(1)), "%a %b %d %I:%M:%S %p %Y")
                out.append(dt)
            except ValueError:
                try:
                    dt = datetime.strptime(m.group(1).replace(" CEST", ""), "%a %b %d %H:%M:%S %Y")
                    out.append(dt)
                except ValueError:
                    pass
        return out

    def ck_mtime(name):
        p = LM / name
        return datetime.fromtimestamp(p.stat().st_mtime) if p.is_file() else None

    # ─── V-JEPA finetune pipeline (stage 1 + stage 2 + pseudo) ─────────────
    vjepa_kills = (parse_attempts(LOGS / "vjepaft_systemd.log") +
                   parse_attempts(LOGS / "vjepapseudo_systemd.log"))
    vjepa_epochs = []  # (datetime, cumulative_epoch, stage_label)
    # Stage 1: 4 epochs. top3=ep2, top2=ep3, top1=ep4. Epoch 1 estimated from gap.
    for ep, fname in [(2, "vjepa_ft_s1_top3.pt"), (3, "vjepa_ft_s1_top2.pt"), (4, "vjepa_ft_s1_top1.pt")]:
        t = ck_mtime(fname)
        if t: vjepa_epochs.append((t, ep, "Stage 1"))
    # Stage 2
    for ep, fname in [(6, "vjepa_ft_s2_top3.pt"), (7, "vjepa_ft_s2_top2.pt"), (8, "vjepa_ft_s2_top1.pt")]:
        t = ck_mtime(fname)
        if t: vjepa_epochs.append((t, ep, "Stage 2"))
    # Pseudo retrain
    for ep, fname in [(10, "vjepa_ft_s2_pseudo_top3.pt"), (11, "vjepa_ft_s2_pseudo_top2.pt"), (12, "vjepa_ft_s2_pseudo_top1.pt")]:
        t = ck_mtime(fname)
        if t: vjepa_epochs.append((t, ep, "Pseudo retrain"))
    vjepa_epochs.sort()

    # ─── VideoMAE-L 16f/224 run ────────────────────────────────────────────
    vmae_kills = parse_attempts(LOGS / "vmaewin_systemd.log")
    vmae_epochs = []
    for ep, fname in [(1, "vmae_l_k400_win_top3.pt"), (2, "vmae_l_k400_win_top2.pt"), (3, "vmae_l_k400_win_top1.pt")]:
        t = ck_mtime(fname)
        if t: vmae_epochs.append((t, ep))
    vmae_epochs.sort()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fmt = mdates.DateFormatter("%d/%m %H:%M")

    # ─── Top panel: V-JEPA pipeline ──────────────────────────────────────
    ax = axes[0]
    if vjepa_epochs:
        # plot step function of cumulative epochs vs time
        times = [e[0] for e in vjepa_epochs]
        eps = [e[1] for e in vjepa_epochs]
        # add a starting point at the first kill (= run start)
        if vjepa_kills:
            start = min(vjepa_kills + times)
            times = [start] + times
            eps = [0] + eps
        ax.step(times, eps, where="post", color=CLEAN, linewidth=2.5, marker="o", markersize=7, label="Époques complétées (cumulé)")
        # colour-segment by stage
        for stage, color in [("Stage 1", "#7fa8c4"), ("Stage 2", "#3a76b5"), ("Pseudo retrain", HIGHLIGHT)]:
            stage_pts = [(t, ep) for t, ep, s in vjepa_epochs if s == stage]
            if stage_pts:
                tt, ee = zip(*stage_pts)
                ax.scatter(tt, ee, s=120, color=color, edgecolor="black", linewidth=0.6, zorder=4, label=stage)
    for kt in vjepa_kills:
        ax.axvline(kt, color=LEAKY, linestyle="--", alpha=0.5, linewidth=1)
    if vjepa_kills:
        ax.scatter(vjepa_kills, [0.3] * len(vjepa_kills), marker="x", color=LEAKY, s=80,
                   label=f"systemd / loop_until_done relance ({len(vjepa_kills)})", zorder=5)
    ax.set_ylabel("Époques complétées (cumulé)")
    ax.set_title("Pipeline V-JEPA finetune + pseudo-label — 12 époques sur 3 jours, plusieurs kills")
    ax.xaxis.set_major_formatter(fmt)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-0.5, 13)
    ax.set_yticks([0, 4, 8, 12])
    ax.set_yticklabels(["start", "fin Stage 1", "fin Stage 2", "fin Pseudo"])
    ax.grid(alpha=0.3)

    # ─── Bottom panel: VideoMAE-L 16f/224 ───────────────────────────────
    ax = axes[1]
    if vmae_epochs:
        times = [e[0] for e in vmae_epochs]
        eps = [e[1] for e in vmae_epochs]
        if vmae_kills:
            start = min(vmae_kills + times)
            times = [start] + times
            eps = [0] + eps
        ax.step(times, eps, where="post", color=CLEAN, linewidth=2.5, marker="o", markersize=8, label="Époques complétées (cumulé)")
    for kt in vmae_kills:
        ax.axvline(kt, color=LEAKY, linestyle="--", alpha=0.5, linewidth=1)
    if vmae_kills:
        ax.scatter(vmae_kills, [0.3] * len(vmae_kills), marker="x", color=LEAKY, s=80,
                   label=f"systemd / loop_until_done relance ({len(vmae_kills)})", zorder=5)
    ax.set_ylabel("Époques complétées (cumulé)")
    ax.set_xlabel("Temps réel (jour/mois heure:min)")
    ax.set_title("Pipeline VideoMAE-Large 16f/224 — 3 époques, 17h temps-réel, 4 redémarrages")
    ax.xaxis.set_major_formatter(fmt)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(alpha=0.3)

    fig.suptitle("Résilience aux kills/reboots — la progression est monotone malgré les redémarrages",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "W_resilience_timeline")


def fig_M_top_confusions():
    """Horizontal bar of top confused (true → predicted) pairs."""
    err_mask = ENS_VAL_PREDS != VAL_LABELS
    pairs = Counter(zip(VAL_LABELS[err_mask].tolist(), ENS_VAL_PREDS[err_mask].tolist()))
    top = pairs.most_common(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{t:03d} {short(t, 18)}  →  {p:03d} {short(p, 18)}" for (t, p), _ in top]
    counts = [n for _, n in top]
    bars = ax.barh(range(len(top)), counts, color=LEAKY, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("# erreurs (val)")
    ax.set_title("Top 12 confusions de l'ensemble — paires sémantiquement proches surtout")
    for b, c in zip(bars, counts):
        ax.text(c + 0.5, b.get_y() + b.get_height()/2, str(c), va="center", fontsize=8)
    ax.invert_yaxis()
    save(fig, "M_top_confusions")


# ---------------------------------------------------------------------------
def main():
    print(f"Generating figures in {OUT}/")
    for fn in [
        fig_A_kaggle_journey,
        fig_B_ensemble_size_vs_score,
        fig_C_weighting_strategies,
        fig_D_training_curves,
        fig_E_frame_position_histogram,
        fig_F_mapping_verification,
        fig_G_calibration_train_val_kaggle,
        fig_H_single_model_val_dir,
        fig_I_learned_weights_collapse,
        fig_J_class_imbalance,
        fig_K_per_family_val_acc,
        fig_L_per_class_lift,
        fig_M_top_confusions,
        fig_N_confusion_matrix,
        fig_O_semantic_groups,
        fig_P_top1_vs_top5,
        fig_Q_family_wins,
        fig_R_size_vs_accuracy,
        fig_S_agreement_intra_inter,
        fig_T_pseudo_coverage,
        fig_U_confidence_histogram,
        fig_V_leave_one_out,
        fig_W_resilience_timeline,
        fig_Y_class_table,
        fig_Z_convergence_curves,
        fig_AA_augmentations,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\nDone. {len(list(OUT.glob('*.png')))} figures in {OUT}/")


if __name__ == "__main__":
    main()
