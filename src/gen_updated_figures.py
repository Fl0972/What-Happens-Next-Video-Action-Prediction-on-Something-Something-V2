#!/usr/bin/env python3
"""
Generate updated + new figures for the post-ResNet50 and ablation results.

Overwrites figures that are now outdated (fig01, fig04, fig11) and creates
new ones (fig16–fig18) for the ablation series and backbone comparison.
Does NOT delete any existing figure.

Run from project root:
    python src/gen_updated_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VLOG = ROOT / "models" / "val_logits"
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})

# ---------------------------------------------------------------- helpers
labels = np.load(VLOG / "labels.npy")
N, NC = len(labels), 33


def load(stem: str) -> np.ndarray:
    return np.load(VLOG / f"{stem}.npy").astype(np.float64)


def safe_logsoftmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def log_softmax_avg(*stems: str) -> np.ndarray:
    return np.mean([safe_logsoftmax(load(s)) for s in stems], axis=0)


def topk(logits: np.ndarray, k: int) -> float:
    return float(
        (np.argpartition(-logits, k - 1, axis=1)[:, :k] == labels[:, None])
        .any(1)
        .mean()
    )


def per_class_f1(logits: np.ndarray):
    pred = logits.argmax(1)
    f1s, sup = [], []
    for c in range(NC):
        tp = int(((pred == c) & (labels == c)).sum())
        fp = int(((pred == c) & (labels != c)).sum())
        fn = int(((pred != c) & (labels == c)).sum())
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
        sup.append(tp + fn)
    return np.array(f1s), np.array(sup)


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- class names
def load_class_names() -> list[str]:
    pc = json.load(open(ROOT / "docs" / "analysis_results.json"))["per_class"]
    out = []
    for k in pc:
        head = k.split("_")[0]
        if head.isdigit() and "_" in k:
            out.append(k.split("_", 1)[1].replace("_", " ").strip())
        else:
            out.append(k.replace("_", " "))
    assert len(out) == 33, f"expected 33, got {len(out)}"
    return out


CLASS_NAMES = load_class_names()

# ---------------------------------------------------------------- pre-compute logits
stems_orig = {
    "TSM-Ultra": "tsm_ultra",
    "TSM-Ultra +TTA": "tsm_ultra_tta",
    "TSM-Ultra-v2": "tsm_ultra_v2",
    "TSM-Ultra-v2 +TTA": "tsm_ultra_v2_tta",
    "TSM-Ultra-v2 (rot)": "tsm_ultra_v2_rotating",
    "TSM-Ultra-v2 (rot) +TTA": "tsm_ultra_v2_rotating_tta",
    "VFL-focal": "vfl_ultra_focal",
    "VFL-focal +TTA": "vfl_ultra_focal_tta",
    "VFL-Ultra": "video_former_lite_ultra",
    "VFL-Ultra +TTA": "video_former_lite_ultra_tta",
    "VFL-Ultra (rot)": "video_former_lite_ultra_rotating",
    "VFL-Ultra (rot) +TTA": "video_former_lite_ultra_rotating_tta",
}

stems_new = {
    "TSM-R50 (rot)": "tsm_ultra_resnet50_rotating_rotating_44%",
    "Abl-A: TSM no-focal": "ablation_tsm_no_focal_rotating",
    "Abl-C: TSM focal-γ1": "ablation_tsm_focal_g1_rotating",
    "Abl-E: TSM SGDR-T2": "ablation_tsm_sgdr_t2_rotating",
    "Abl-B: VFL 2-layer": "ablation_vfl_2layers_rotating",
    "Abl-D: VFL 10-fold": "ablation_vfl_folds10_rotating",
    "Abl-F: VFL2 SGDR-T2": "ablation_vfl2_sgdr_t2_rotating",
}

all_stems = {**stems_orig, **stems_new}
per_model = {name: {"top1": topk(load(s), 1), "top5": topk(load(s), 5)}
             for name, s in all_stems.items()}

# 5-model best ensemble
BEST5 = [
    "tsm_ultra_resnet50_rotating_rotating_44%",
    "tsm_ultra_v2_tta",
    "tsm_ultra_v2_rotating_tta",
    "video_former_lite_ultra_rotating_tta",
    "ablation_tsm_no_focal_rotating",
]
# old 3-model best
BEST3 = ["tsm_ultra_v2_tta", "tsm_ultra_v2_rotating_tta", "video_former_lite_ultra_rotating_tta"]
# 4-model (R50 + best3)
BEST4 = ["tsm_ultra_resnet50_rotating_rotating_44%"] + BEST3

ens_best5 = log_softmax_avg(*BEST5)
ens_best4 = log_softmax_avg(*BEST4)
ens_best3 = log_softmax_avg(*BEST3)
ens_tsm2  = log_softmax_avg("tsm_ultra_v2_tta", "tsm_ultra_v2_rotating_tta")
ens_all6  = log_softmax_avg(
    "tsm_ultra_v2_tta", "tsm_ultra_v2_rotating_tta",
    "video_former_lite_ultra_rotating_tta",
    "vfl_ultra_focal_tta", "tsm_ultra_tta", "video_former_lite_ultra_tta",
)

f1_best5, support = per_class_f1(ens_best5)
f1_best_single, _ = per_class_f1(load("tsm_ultra_resnet50_rotating_rotating_44%"))

# ================================================================ FIG 01 — per-model accuracy (updated)
items = sorted(per_model.items(), key=lambda kv: kv[1]["top1"])
names = [k for k, _ in items]
t1 = [v["top1"] * 100 for _, v in items]
t5 = [v["top5"] * 100 for _, v in items]

# colour: red for new models, blue for original
colors_t1 = ["#e74c3c" if n in stems_new else "#2b7bba" for n in names]

fig, ax = plt.subplots(figsize=(10, 9))
y = np.arange(len(names))
ax.barh(y, t5, color=["#f5b7b1" if n in stems_new else "#cfe3f3" for n in names], label="Top-5")
ax.barh(y, t1, color=colors_t1, label="Top-1 (new models = red)")
for i, (a, b) in enumerate(zip(t1, t5)):
    ax.text(a + 0.3, i, f"{a:.1f}", va="center", fontsize=7.5,
            color="#c0392b" if names[i] in stems_new else "#0b3d61")
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Val-dir accuracy (%)")
ax.set_xlim(0, 88)
ax.set_title("Per-model val-dir accuracy — all from-scratch models\n"
             "(red = new models: TSM-R50 and ablation series A–F)")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
save(fig, "fig01_per_model_accuracy.png")

# ================================================================ FIG 04 — ensemble comparison (updated)
elabels = [
    "Best single\n(R50 rot)",
    "Prev. best single\n(TSM-v2-rot +TTA)",
    "2x TSM-v2\nlog-avg",
    "All-6 TTA\nlog-avg",
    "Old best 3\n(R18 only)",
    "R50 + old best 3\n(4-model)",
    "NEW BEST\n5-model",
]
evals = [
    per_model["TSM-R50 (rot)"]["top1"] * 100,
    per_model["TSM-Ultra-v2 (rot) +TTA"]["top1"] * 100,
    topk(ens_tsm2, 1) * 100,
    topk(ens_all6, 1) * 100,
    topk(ens_best3, 1) * 100,
    topk(ens_best4, 1) * 100,
    topk(ens_best5, 1) * 100,
]
colors = ["#d5e8d4", "#bdbdbd", "#fdae6b", "#fd8d3c", "#e6550d", "#d45500", "#8b0000"]
fig, ax = plt.subplots(figsize=(10, 5))
b = ax.bar(elabels, evals, color=colors, edgecolor="white", linewidth=1.2)
for r, v in zip(b, evals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.12, f"{v:.2f}",
            ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("Val-dir top-1 (%)")
ax.set_ylim(38, 50)
ax.set_title("Ensemble progression — from scratch (closed track)")
ax.axhline(44.03, color="#e6550d", lw=1.2, ls="--", alpha=0.6,
           label="Previous best: 44.03% (3-model, R18 only)")
ax.axhline(46.75, color="#8b0000", lw=1.2, ls="--", alpha=0.6,
           label="New best: 46.75% (5-model with R50)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
save(fig, "fig04_ensemble.png")

# ================================================================ FIG 11 — leave-one-out (updated, 5-model)
loo_names_disp = {
    BEST5[0]: "TSM-R50 (rot)",
    BEST5[1]: "TSM-v2 +TTA",
    BEST5[2]: "TSM-v2-rot +TTA",
    BEST5[3]: "VFL-rot +TTA",
    BEST5[4]: "TSM-no-focal",
}
full_t1 = topk(ens_best5, 1) * 100
loo_vals = []
for drop in BEST5:
    rem = [s for s in BEST5 if s != drop]
    loo_vals.append(topk(log_softmax_avg(*rem), 1) * 100)

xl = ["Full\n5-model"] + [f"drop\n{loo_names_disp[s]}" for s in BEST5]
lv = [full_t1] + loo_vals
deltas = [""] + [f"\n({v - full_t1:+.2f})" for v in loo_vals]

fig, ax = plt.subplots(figsize=(9, 5))
cols = ["#8b0000"] + ["#fdae6b"] * 5
b = ax.bar(xl, lv, color=cols)
for r, v, d in zip(b, lv, deltas):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.08,
            f"{v:.2f}{d}", ha="center", fontsize=8)
ax.set_ylabel("Val-dir top-1 (%)")
ax.set_ylim(43, 48.5)
ax.set_title("Leave-one-out: contribution of each component — best 5-model ensemble")
ax.axhline(44.03, color="#e6550d", lw=1.0, ls="--", alpha=0.5, label="Prev. 3-model best: 44.03%")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
save(fig, "fig11_leave_one_out.png")

# ================================================================ FIG 10 — checkpoint internal val (updated)
stats = json.load(open(ROOT / "docs" / "model_stats.json"))
ckpt_meta = stats["checkpoints"]
cm_models = [m for m in ckpt_meta if m.get("internal_val_acc") is not None]
cm_models.sort(key=lambda m: m["internal_val_acc"])
lab = [f'{m["name"]}\n({m.get("params_M", "?")}M, {m["num_frames"]}f)' for m in cm_models]
vals = [m["internal_val_acc"] * 100 for m in cm_models]
# mark which ones are new
new_names = {
    "tsm_ultra_resnet50_rotating_rotating_44%",
    "ablation_tsm_no_focal_rotating",
    "ablation_tsm_focal_g1_rotating",
    "ablation_tsm_sgdr_t2_rotating",
    "ablation_vfl_2layers_rotating",
    "ablation_vfl_folds10_rotating",
    "ablation_vfl2_sgdr_t2_rotating",
}
cols = ["#e74c3c" if m["name"] in new_names else "#7f8c8d" for m in cm_models]
fig, ax = plt.subplots(figsize=(12, 7))
b = ax.barh(range(len(vals)), vals, color=cols)
for r, v in zip(b, vals):
    ax.text(v + 0.4, r.get_y() + r.get_height() / 2, f"{v:.1f}", va="center", fontsize=7.5)
ax.set_yticks(range(len(lab)))
ax.set_yticklabels(lab, fontsize=7)
ax.set_xlabel("Internal-val / rolling-avg accuracy (%) — NOT val-dir")
ax.set_title("All from-scratch checkpoints: training-side accuracy (red = new)\n"
             "⚠ Rotating-fold values are rolling averages; not comparable to fixed-split values")
ax.grid(axis="x", alpha=0.3)
save(fig, "fig10_checkpoint_internal_val.png")

# ================================================================ FIG 05 — per-class F1, new best ensemble
order = np.argsort(f1_best5)
present = support[order] > 0
oi = order[present]
fig, ax = plt.subplots(figsize=(8, 9))
ax.barh(np.arange(len(oi)), f1_best5[oi],
        color=plt.cm.viridis(support[oi] / support.max()))
ax.set_yticks(np.arange(len(oi)))
ax.set_yticklabels([CLASS_NAMES[i] for i in oi], fontsize=7)
ax.set_xlabel("F1 (new best 5-model ensemble)")
ax.set_title("Per-class F1 — new best ensemble (colour = #val samples)")
sm = plt.cm.ScalarMappable(
    cmap="viridis",
    norm=plt.Normalize(vmin=float(support.min()), vmax=float(support.max())),
)
fig.colorbar(sm, ax=ax, label="val support", fraction=0.046, pad=0.04)
ax.grid(axis="x", alpha=0.3)
save(fig, "fig05_per_class_f1.png")

# ================================================================ FIG 09 — ensemble vs best single, per-class delta
delta = f1_best5 - f1_best_single
m = support > 0
idx = np.where(m)[0]
idx = idx[np.argsort(delta[idx])]
fig, ax = plt.subplots(figsize=(8, 9))
cols09 = ["#2ca02c" if delta[i] >= 0 else "#d62728" for i in idx]
ax.barh(range(len(idx)), delta[idx] * 100, color=cols09)
ax.set_yticks(range(len(idx)))
ax.set_yticklabels([CLASS_NAMES[i] for i in idx], fontsize=7)
ax.set_xlabel("F1 change: 5-model ensemble − best single (TSM-R50 rot) (pp)")
ax.set_title("Where the 5-model ensemble helps vs TSM-R50 (best single)")
ax.axvline(0, color="k", lw=0.8)
ax.grid(axis="x", alpha=0.3)
save(fig, "fig09_ensemble_vs_single_perclass.png")

# ================================================================ FIG 16 — TSM ablation γ sweep + SGDR
ref_t3 = per_model["TSM-Ultra-v2 (rot)"]["top1"] * 100
ablation_tsm = {
    "T3 ref\n(γ=2, SGDR-T1)": ref_t3,
    "Exp A\n(γ=0, CE)": per_model["Abl-A: TSM no-focal"]["top1"] * 100,
    "Exp C\n(γ=1)": per_model["Abl-C: TSM focal-γ1"]["top1"] * 100,
    "Exp E\n(SGDR T_mult=2)": per_model["Abl-E: TSM SGDR-T2"]["top1"] * 100,
}
xl16 = list(ablation_tsm.keys())
vl16 = list(ablation_tsm.values())
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ["#2b7bba", "#e74c3c", "#f39c12", "#9b59b6"]
b = ax.bar(xl16, vl16, color=bar_colors, width=0.55)
for r, v in zip(b, vl16):
    delta_s = f"({v - ref_t3:+.2f})" if r.get_x() != b[0].get_x() else "(ref)"
    ax.text(r.get_x() + r.get_width() / 2, v + 0.08, f"{v:.2f}\n{delta_s}",
            ha="center", fontsize=9)
ax.set_ylabel("Val-dir top-1 (%)")
ax.set_ylim(37.5, 41)
ax.set_title("TSM ablations: focal γ and SGDR schedule (all others frozen)\n"
             "Reference T3: TSM-Ultra-v2 rotating, γ=2, SGDR T0=25 T_mult=1")
ax.axhline(ref_t3, color="#2b7bba", lw=1.0, ls="--", alpha=0.5)
ax.grid(axis="y", alpha=0.3)
save(fig, "fig16_ablation_tsm.png")

# ================================================================ FIG 17 — VFL ablation depth + folds + SGDR
ref_v3 = per_model["VFL-Ultra (rot)"]["top1"] * 100
ablation_vfl = {
    "V3 ref\n(3-layer, k=5, T1)": ref_v3,
    "Exp B\n(2-layer, k=5)": per_model["Abl-B: VFL 2-layer"]["top1"] * 100,
    "Exp D\n(3-layer, k=10)": per_model["Abl-D: VFL 10-fold"]["top1"] * 100,
    "Exp F\n(2-layer, k=5, T2)": per_model["Abl-F: VFL2 SGDR-T2"]["top1"] * 100,
}
xl17 = list(ablation_vfl.keys())
vl17 = list(ablation_vfl.values())
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors17 = ["#6a3d9a", "#e74c3c", "#f39c12", "#c0392b"]
b = ax.bar(xl17, vl17, color=bar_colors17, width=0.55)
for r, v in zip(b, vl17):
    delta_s = f"({v - ref_v3:+.2f})" if r.get_x() != b[0].get_x() else "(ref)"
    ax.text(r.get_x() + r.get_width() / 2, v + 0.06, f"{v:.2f}\n{delta_s}",
            ha="center", fontsize=9)
ax.set_ylabel("Val-dir top-1 (%)")
ax.set_ylim(34, 38)
ax.set_title("VFL ablations: Transformer depth, fold count, SGDR schedule\n"
             "Reference V3: VFL-Ultra rotating, 3-layer, k=5, SGDR T0=25 T_mult=1")
ax.axhline(ref_v3, color="#6a3d9a", lw=1.0, ls="--", alpha=0.5)
ax.grid(axis="y", alpha=0.3)
save(fig, "fig17_ablation_vfl.png")

# ================================================================ FIG 18 — backbone R18 vs R50 comparison
backbone_data = {
    "Model": ["TSM-Ultra\n(R18, T=5)", "TSM-Ultra-v2\n(R18, T=4)", "TSM-v2 rot\n(R18, T=4)",
               "TSM-v2 rot\n+TTA", "TSM-R50 rot\n(R50, T=4)"],
    "top1": [34.91, 38.25, 39.45, 41.68, 42.85],
    "backbone": ["R18", "R18", "R18", "R18", "R50"],
    "rot": [False, False, True, True, True],
}
x18 = np.arange(len(backbone_data["Model"]))
t1_18 = backbone_data["top1"]
# color: R50 = different shade
c18 = ["#7fb3d5", "#2b7bba", "#1a5276", "#154360", "#c0392b"]
fig, ax = plt.subplots(figsize=(10, 5))
b = ax.bar(x18, t1_18, color=c18, width=0.6)
for r, v, bb in zip(b, t1_18, backbone_data["backbone"]):
    label = f"{v:.2f}%"
    ax.text(r.get_x() + r.get_width() / 2, v + 0.15, label, ha="center",
            fontsize=9, fontweight="bold")
# annotate gains
for i in range(1, len(t1_18)):
    delta = t1_18[i] - t1_18[i - 1]
    ax.annotate("", xy=(x18[i], t1_18[i] + 0.5), xytext=(x18[i - 1], t1_18[i - 1] + 0.5),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.text((x18[i] + x18[i - 1]) / 2, max(t1_18[i], t1_18[i - 1]) + 1.0,
            f"+{delta:.2f} pp", ha="center", fontsize=8, color="gray")
ax.set_xticks(x18)
ax.set_xticklabels(backbone_data["Model"], fontsize=9)
ax.set_ylabel("Val-dir top-1 (%)")
ax.set_ylim(30, 48)
ax.set_title("TSM progression: from T=5 R18 baseline to R50 rotating-fold model\n"
             "(blue = ResNet18, red = ResNet50)")
ax.grid(axis="y", alpha=0.3)
# legend
from matplotlib.patches import Patch
legend_els = [Patch(facecolor="#2b7bba", label="ResNet18"),
              Patch(facecolor="#c0392b", label="ResNet50")]
ax.legend(handles=legend_els, loc="upper left")
save(fig, "fig18_backbone_r18_vs_r50.png")

print(f"\nDone. Updated/created figures in {FIGS}")
print("Updated : fig01, fig04, fig05, fig09, fig10, fig11")
print("New     : fig16 (TSM ablations), fig17 (VFL ablations), fig18 (backbone)")
