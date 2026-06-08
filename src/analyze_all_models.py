#!/usr/bin/env python3
"""Full from-scratch model report: metrics, ensembles, per-class, confusion + figures.

Reads cached val-dir logits (models/val_logits/*.npy) and checkpoint metadata
(models/*.pt), computes every statistic with numpy, and writes:
  - docs/figures/*.png   (all graphs)
  - docs/model_stats.json (all computed numbers)

Run from the project root:  python src/analyze_all_models.py
No GPU / dataset / re-inference needed — works entirely off cached logits.

Only FROM-SCRATCH (closed-track) models are analysed. The pretrained/open-track
ViT logits present in val_logits/ are deliberately excluded.
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

# ---------------------------------------------------------------- class names
def load_class_names() -> list[str]:
    # The per_class dict is in label-index order (verified: support matches
    # labels.npy for all 33). The NNN_ prefix is the *folder* digit, which
    # skips 027, so it must NOT be used as the index.
    pc = json.load(open(ROOT / "docs" / "analysis_results.json"))["per_class"]
    out = []
    for k in pc:
        head = k.split("_")[0]
        if head.isdigit() and "_" in k:
            out.append(k.split("_", 1)[1].replace("_", " ").strip())
        else:
            out.append(k.replace("_", " "))
    assert len(out) == 33, f"expected 33 class names, got {len(out)}"
    return out

CLASS_NAMES = load_class_names()
LABELS = np.load(VLOG / "labels.npy")
N, NC = LABELS.shape[0], 33

# logit file -> display name (FROM-SCRATCH only; pretrained ViT excluded)
FILE2NAME = {
    "tsm_ultra": "TSM-Ultra",
    "tsm_ultra_tta": "TSM-Ultra +TTA",
    "tsm_ultra_v2": "TSM-Ultra-v2",
    "tsm_ultra_v2_tta": "TSM-Ultra-v2 +TTA",
    "tsm_ultra_v2_rotating": "TSM-Ultra-v2 (rot)",
    "tsm_ultra_v2_rotating_tta": "TSM-Ultra-v2 (rot) +TTA",
    "vfl_ultra_focal": "VFL-focal",
    "vfl_ultra_focal_tta": "VFL-focal +TTA",
    "video_former_lite_ultra": "VFL-Ultra",
    "video_former_lite_ultra_tta": "VFL-Ultra +TTA",
    "video_former_lite_ultra_rotating": "VFL-Ultra (rot)",
    "video_former_lite_ultra_rotating_tta": "VFL-Ultra (rot) +TTA",
}
# short keys (match ENSEMBLE_EXP.md) -> filename
KEY2FILE = {
    "tsm": "tsm_ultra", "tsm_tta": "tsm_ultra_tta",
    "tsm_v2": "tsm_ultra_v2", "tsm_v2_tta": "tsm_ultra_v2_tta",
    "tsm_v2_rot": "tsm_ultra_v2_rotating", "tsm_v2_rot_tta": "tsm_ultra_v2_rotating_tta",
    "vfl_focal": "vfl_ultra_focal", "vfl_focal_tta": "vfl_ultra_focal_tta",
    "vfl": "video_former_lite_ultra", "vfl_tta": "video_former_lite_ultra_tta",
    "vfl_rot": "video_former_lite_ultra_rotating", "vfl_rot_tta": "video_former_lite_ultra_rotating_tta",
}

LOGITS = {f: np.load(VLOG / f"{f}.npy").astype(np.float64) for f in FILE2NAME}

# ------------------------------------------------------------------- metrics
def log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

def topk_acc(logits, k):
    topk = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    return float(np.any(topk == LABELS[:, None], axis=1).mean())

def confusion(pred):
    cm = np.zeros((NC, NC), dtype=np.int64)
    np.add.at(cm, (LABELS, pred), 1)
    return cm

def per_class_prf(pred):
    cm = confusion(pred)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    pred_tot = cm.sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_tot > 0, tp / pred_tot, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1, support, cm

def metrics_from_logits(logits):
    pred = logits.argmax(1)
    prec, rec, f1, support, _ = per_class_prf(pred)
    present = support > 0
    return {
        "top1": float((pred == LABELS).mean()),
        "top5": topk_acc(logits, 5),
        "macro_precision": float(prec[present].mean()),
        "macro_recall": float(rec[present].mean()),
        "macro_f1": float(f1[present].mean()),
    }

# ---------------------------------------------------------- per-model results
per_model = {}
for f, name in FILE2NAME.items():
    per_model[name] = metrics_from_logits(LOGITS[f])
    per_model[name]["file"] = f

# --------------------------------------------------------------- ensembles
def log_avg(keys):
    acc = sum(log_softmax(LOGITS[KEY2FILE[k]]) for k in keys) / len(keys)
    return acc

def weighted(keys_w):
    acc = sum(w * LOGITS[KEY2FILE[k]] for k, w in keys_w)
    return acc

ens = {}
best_keys = ["tsm_v2_tta", "tsm_v2_rot_tta", "vfl_rot_tta"]
ens["best_logavg (tsm_v2_tta+tsm_v2_rot_tta+vfl_rot_tta)"] = metrics_from_logits(log_avg(best_keys))
ens["logavg2 (tsm_v2_tta+tsm_v2_rot_tta)"] = metrics_from_logits(log_avg(["tsm_v2_tta", "tsm_v2_rot_tta"]))
all6 = ["tsm_v2_tta", "tsm_v2_rot_tta", "tsm_tta", "vfl_tta", "vfl_focal_tta", "vfl_rot_tta"]
ens["all6_TTA_logavg"] = metrics_from_logits(log_avg(all6))

# leave-one-out on the BEST log-avg ensemble (per-component contribution)
loo = {}
full_best = ens["best_logavg (tsm_v2_tta+tsm_v2_rot_tta+vfl_rot_tta)"]["top1"]
for drop in best_keys:
    rem = [k for k in best_keys if k != drop]
    loo[f"-{drop}"] = {"top1": metrics_from_logits(log_avg(rem))["top1"]}
    loo[f"-{drop}"]["delta_pp"] = round(100 * (loo[f"-{drop}"]["top1"] - full_best), 2)

# ----------------------------------------------------- best-ensemble deep dive
best_logits = log_avg(best_keys)
best_pred = best_logits.argmax(1)
prec, rec, f1, support, cm = per_class_prf(best_pred)

# TTA trade-off on direction-sensitive classes (flip TTA inverts left/right)
DIR_IDX = [18, 19]  # Pulling L->R, Pulling R->L
best_noTTA_logits = log_avg(["tsm_v2", "tsm_v2_rot", "vfl_rot"])
_, _, f1_noTTA, _, _ = per_class_prf(best_noTTA_logits.argmax(1))
tta_tradeoff = {
    "overall_top1_noTTA": round(float((best_noTTA_logits.argmax(1) == LABELS).mean()), 4),
    "overall_top1_TTA": round(float(best_pred.__eq__(LABELS).mean()), 4),
    "dir_f1_noTTA": {CLASS_NAMES[i]: round(float(f1_noTTA[i]), 3) for i in DIR_IDX},
    "dir_f1_TTA": {CLASS_NAMES[i]: round(float(f1[i]), 3) for i in DIR_IDX},
}
best_single_file = max(((f, m["top1"]) for f, m in
                        ((f, metrics_from_logits(LOGITS[f])) for f in FILE2NAME)),
                       key=lambda t: t[1])[0]
single_pred = LOGITS[best_single_file].argmax(1)
_, _, f1_single, _, _ = per_class_prf(single_pred)

# top confused pairs (best ensemble)
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
pairs = []
for i in range(NC):
    for j in range(NC):
        if cm_off[i, j] > 0:
            pairs.append((int(cm_off[i, j]), i, j))
pairs.sort(reverse=True)
top_pairs = pairs[:15]

# ------------------------------------------------------- checkpoint metadata
import torch
CKPTS = sorted(p for p in (ROOT / "models").glob("*.pt") if ".prev." not in p.name)
ckpt_meta = []
for p in CKPTS:
    try:
        c = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        ckpt_meta.append({"name": p.stem, "error": str(e)[:80]})
        continue
    sd = c.get("model_state_dict", {})
    nparams = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    ckpt_meta.append({
        "name": p.stem,
        "internal_val_acc": (round(float(c["val_accuracy"]), 4) if c.get("val_accuracy") is not None else None),
        "num_frames": c.get("num_frames"),
        "pretrained": bool(c.get("pretrained", False)),
        "params_M": round(nparams / 1e6, 1),
        "has_valdir_logits": p.stem in FILE2NAME,
    })
    del c, sd

# ============================================================ FIGURES
def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print("  wrote", name)

# 1. per-variant top1/top5 (sorted)
items = sorted(per_model.items(), key=lambda kv: kv[1]["top1"])
names = [k for k, _ in items]
t1 = [v["top1"] * 100 for _, v in items]
t5 = [v["top5"] * 100 for _, v in items]
fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(names))
ax.barh(y, t5, color="#cfe3f3", label="Top-5")
ax.barh(y, t1, color="#2b7bba", label="Top-1")
ax.set_yticks(y); ax.set_yticklabels(names)
for i, (a, b) in enumerate(zip(t1, t5)):
    ax.text(a + 0.4, i, f"{a:.1f}", va="center", fontsize=8, color="#0b3d61")
    ax.text(b + 0.4, i, f"{b:.1f}", va="center", fontsize=8, color="#5a5a5a")
ax.set_xlabel("Val-dir accuracy (%)"); ax.set_xlim(0, 85)
ax.set_title("Per-model accuracy on val-dir (6,745 clips) — from-scratch models")
ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.3)
save(fig, "fig01_per_model_accuracy.png")

# 2. TTA effect (base vs +TTA)
bases = [("TSM-Ultra", "TSM-Ultra +TTA"), ("TSM-Ultra-v2", "TSM-Ultra-v2 +TTA"),
         ("TSM-Ultra-v2 (rot)", "TSM-Ultra-v2 (rot) +TTA"), ("VFL-focal", "VFL-focal +TTA"),
         ("VFL-Ultra", "VFL-Ultra +TTA"), ("VFL-Ultra (rot)", "VFL-Ultra (rot) +TTA")]
labels = [b[0] for b in bases]
no = [per_model[b[0]]["top1"] * 100 for b in bases]
yes = [per_model[b[1]]["top1"] * 100 for b in bases]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, no, w, label="no TTA", color="#9bbf85")
ax.bar(x + w/2, yes, w, label="+10-crop TTA", color="#3f7a1f")
for i, (a, b) in enumerate(zip(no, yes)):
    ax.text(x[i] + w/2, b + 0.2, f"+{b-a:.1f}", ha="center", fontsize=8, color="#234d10")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Val-dir top-1 (%)"); ax.set_title("Effect of 10-crop test-time augmentation")
ax.legend(); ax.grid(axis="y", alpha=.3)
save(fig, "fig02_tta_effect.png")

# 3. rotating-fold effect
groups = [("TSM-Ultra-v2", "TSM-Ultra-v2", "TSM-Ultra-v2 (rot)"),
          ("TSM-Ultra-v2 +TTA", "TSM-Ultra-v2 +TTA", "TSM-Ultra-v2 (rot) +TTA"),
          ("VFL-Ultra", "VFL-Ultra", "VFL-Ultra (rot)"),
          ("VFL-Ultra +TTA", "VFL-Ultra +TTA", "VFL-Ultra (rot) +TTA")]
lab = [g[0] for g in groups]
single = [per_model[g[1]]["top1"] * 100 for g in groups]
rot = [per_model[g[2]]["top1"] * 100 for g in groups]
x = np.arange(len(lab)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, single, w, label="single split", color="#c2a5cf")
ax.bar(x + w/2, rot, w, label="rotating 5-fold", color="#6a3d9a")
for i, (a, b) in enumerate(zip(single, rot)):
    ax.text(x[i] + w/2, b + 0.2, f"{b-a:+.1f}", ha="center", fontsize=8, color="#3d2259")
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=15, ha="right")
ax.set_ylabel("Val-dir top-1 (%)"); ax.set_title("Effect of rotating 5-fold training")
ax.legend(); ax.grid(axis="y", alpha=.3)
save(fig, "fig03_rotating_effect.png")

# 4. ensembles vs best single
elabels = ["Best single\n(tsm_v2_rot +TTA)", "log-avg 2\n(2x TSM-v2)",
           "all-6 TTA\nlog-avg", "Best ensemble\nlog-avg (3)"]
evals = [per_model["TSM-Ultra-v2 (rot) +TTA"]["top1"] * 100,
         ens["logavg2 (tsm_v2_tta+tsm_v2_rot_tta)"]["top1"] * 100,
         ens["all6_TTA_logavg"]["top1"] * 100,
         ens["best_logavg (tsm_v2_tta+tsm_v2_rot_tta+vfl_rot_tta)"]["top1"] * 100]
colors = ["#bdbdbd", "#fdae6b", "#fd8d3c", "#e6550d"]
fig, ax = plt.subplots(figsize=(8, 5))
b = ax.bar(elabels, evals, color=colors)
for r, v in zip(b, evals):
    ax.text(r.get_x() + r.get_width()/2, v + 0.15, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_ylabel("Val-dir top-1 (%)"); ax.set_ylim(38, 46)
ax.set_title("Ensembling vs best single model")
ax.grid(axis="y", alpha=.3)
save(fig, "fig04_ensemble.png")

# 5. per-class F1 (best ensemble), sorted, colored by support
order = np.argsort(f1)
present = support[order] > 0
oi = order[present]
fig, ax = plt.subplots(figsize=(8, 9))
sc = ax.barh(np.arange(len(oi)), f1[oi],
             color=plt.cm.viridis((support[oi] / support.max())))
ax.set_yticks(np.arange(len(oi)))
ax.set_yticklabels([CLASS_NAMES[i] for i in oi], fontsize=7)
ax.set_xlabel("F1 (best ensemble)")
ax.set_title("Per-class F1 — best ensemble (colour = #val samples)")
sm = plt.cm.ScalarMappable(cmap="viridis",
        norm=plt.Normalize(vmin=float(support.min()), vmax=float(support.max())))
fig.colorbar(sm, ax=ax, label="val support", fraction=0.046, pad=0.04)
ax.grid(axis="x", alpha=.3)
save(fig, "fig05_per_class_f1.png")

# 6. confusion matrix (row-normalised), best ensemble
cmn = cm.astype(float)
rs = cmn.sum(1, keepdims=True)
cmn = np.divide(cmn, rs, out=np.zeros_like(cmn), where=rs > 0)
fig, ax = plt.subplots(figsize=(11, 9.5))
im = ax.imshow(cmn, cmap="magma", vmin=0, vmax=1)
ax.set_xticks(range(NC)); ax.set_yticks(range(NC))
ax.set_xticklabels([CLASS_NAMES[i] for i in range(NC)], rotation=90, fontsize=6)
ax.set_yticklabels([CLASS_NAMES[i] for i in range(NC)], fontsize=6)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Row-normalised confusion matrix — best ensemble")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(pred | true)")
save(fig, "fig06_confusion_matrix.png")

# 7. top confused pairs
plabels = [f"{CLASS_NAMES[i]}  →  {CLASS_NAMES[j]}" for _, i, j in top_pairs][::-1]
pvals = [c for c, _, _ in top_pairs][::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(range(len(pvals)), pvals, color="#c0392b")
ax.set_yticks(range(len(pvals))); ax.set_yticklabels(plabels, fontsize=8)
ax.set_xlabel("# misclassified clips"); ax.set_title("Top-15 confused class pairs (best ensemble)")
ax.grid(axis="x", alpha=.3)
save(fig, "fig07_confused_pairs.png")

# 8. F1 vs support scatter
fig, ax = plt.subplots(figsize=(8, 5.5))
m = support > 0
ax.scatter(support[m], f1[m], c="#2b7bba", s=30)
for i in np.where(m)[0]:
    if f1[i] < 0.18 or f1[i] > 0.55:
        ax.annotate(CLASS_NAMES[i], (support[i], f1[i]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Val support (#clips)"); ax.set_ylabel("F1 (best ensemble)")
ax.set_title("Per-class F1 vs class frequency (20x imbalance)")
ax.grid(alpha=.3)
save(fig, "fig08_f1_vs_support.png")

# 9. ensemble vs best single, per-class F1 delta
delta = f1 - f1_single
m = support > 0
idx = np.where(m)[0]
idx = idx[np.argsort(delta[idx])]
fig, ax = plt.subplots(figsize=(8, 9))
cols = ["#2ca02c" if delta[i] >= 0 else "#d62728" for i in idx]
ax.barh(range(len(idx)), delta[idx] * 100, color=cols)
ax.set_yticks(range(len(idx))); ax.set_yticklabels([CLASS_NAMES[i] for i in idx], fontsize=7)
ax.set_xlabel("F1 change: best ensemble − best single model (pp)")
ax.set_title("Where the ensemble helps vs the best single model")
ax.axvline(0, color="k", lw=.8); ax.grid(axis="x", alpha=.3)
save(fig, "fig09_ensemble_vs_single_perclass.png")

# 10. checkpoint internal-val accuracy (from-scratch checkpoints)
cm_models = [m for m in ckpt_meta if m.get("internal_val_acc") is not None]
cm_models.sort(key=lambda m: m["internal_val_acc"])
lab = [f'{m["name"]}\n({m["params_M"]}M, {m["num_frames"]}f)' for m in cm_models]
vals = [m["internal_val_acc"] * 100 for m in cm_models]
fig, ax = plt.subplots(figsize=(11, 6))
b = ax.barh(range(len(vals)), vals, color="#7f8c8d")
for r, v in zip(b, vals):
    ax.text(v + 0.4, r.get_y() + r.get_height()/2, f"{v:.1f}", va="center", fontsize=8)
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=7)
ax.set_xlabel("Internal-val accuracy (%) — train-split, NOT val-dir")
ax.set_title("All from-scratch checkpoints: internal-val accuracy (params, frames)")
ax.grid(axis="x", alpha=.3)
save(fig, "fig10_checkpoint_internal_val.png")

# 11. leave-one-out on the best ensemble
disp = {"tsm_v2_tta": "TSM-v2 +TTA", "tsm_v2_rot_tta": "TSM-v2(rot) +TTA", "vfl_rot_tta": "VFL(rot) +TTA"}
lk = ["full"] + best_keys
lv = [full_best * 100] + [loo[f"-{k}"]["top1"] * 100 for k in best_keys]
xl = ["Full\nensemble"] + [f"drop\n{disp[k]}" for k in best_keys]
fig, ax = plt.subplots(figsize=(7.8, 4.8))
cols = ["#e6550d"] + ["#fdae6b"] * 3
b = ax.bar(xl, lv, color=cols)
for i, (r, v, k) in enumerate(zip(b, lv, lk)):
    d = "" if k == "full" else f"\n({loo[f'-{k}']['delta_pp']:+.2f})"
    ax.text(r.get_x() + r.get_width()/2, v + 0.08, f"{v:.2f}{d}", ha="center", fontsize=8)
ax.set_ylabel("Val-dir top-1 (%)"); ax.set_ylim(40, 45)
ax.set_title("Leave-one-out: contribution of each component to the best ensemble")
ax.grid(axis="y", alpha=.3)
save(fig, "fig11_leave_one_out.png")

# 12. TTA directional trade-off
labd = ["Pulling L→R", "Pulling R→L", "Overall top-1"]
no = [f1_noTTA[18] * 100, f1_noTTA[19] * 100, tta_tradeoff["overall_top1_noTTA"] * 100]
yes = [f1[18] * 100, f1[19] * 100, tta_tradeoff["overall_top1_TTA"] * 100]
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(7.6, 5))
ax.bar(x - w/2, no, w, label="ensemble, no TTA", color="#7fb3d5")
ax.bar(x + w/2, yes, w, label="ensemble, +TTA (5 crops × 2 flips)", color="#1f618d")
for i, (a, b) in enumerate(zip(no, yes)):
    ax.text(x[i] - w/2, a + 0.5, f"{a:.1f}", ha="center", fontsize=8)
    ax.text(x[i] + w/2, b + 0.5, f"{b:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labd)
ax.set_ylabel("F1 / accuracy (%)")
ax.set_title("TTA trade-off: horizontal-flip TTA raises overall accuracy\nbut halves F1 on direction-sensitive classes")
ax.legend(); ax.grid(axis="y", alpha=.3)
save(fig, "fig12_tta_directional_tradeoff.png")

# 13. pairwise agreement matrix (ensemble-diversity diagnostic)
agree_order = [
    "tsm_ultra_v2", "tsm_ultra_v2_tta", "tsm_ultra_v2_rotating", "tsm_ultra_v2_rotating_tta",
    "tsm_ultra", "tsm_ultra_tta",
    "video_former_lite_ultra", "video_former_lite_ultra_tta",
    "video_former_lite_ultra_rotating", "video_former_lite_ultra_rotating_tta",
    "vfl_ultra_focal", "vfl_ultra_focal_tta",
]
apred = {f: LOGITS[f].argmax(1) for f in agree_order}
na = len(agree_order)
A = np.zeros((na, na))
for i in range(na):
    for j in range(na):
        A[i, j] = float((apred[agree_order[i]] == apred[agree_order[j]]).mean())
SHORT = {
    "tsm_ultra_v2": "TSM-v2", "tsm_ultra_v2_tta": "TSM-v2·TTA",
    "tsm_ultra_v2_rotating": "TSM-v2·rot", "tsm_ultra_v2_rotating_tta": "TSM-v2·rot·TTA",
    "tsm_ultra": "TSM", "tsm_ultra_tta": "TSM·TTA",
    "video_former_lite_ultra": "VFL", "video_former_lite_ultra_tta": "VFL·TTA",
    "video_former_lite_ultra_rotating": "VFL·rot", "video_former_lite_ultra_rotating_tta": "VFL·rot·TTA",
    "vfl_ultra_focal": "VFL·foc", "vfl_ultra_focal_tta": "VFL·foc·TTA",
}
alabels = [SHORT[f] for f in agree_order]
vmin = float(np.floor(A[~np.eye(na, dtype=bool)].min() * 20) / 20)  # round down to .05
fig, ax = plt.subplots(figsize=(16, 14.5))
im = ax.imshow(A, cmap="RdYlGn", vmin=vmin, vmax=1.0)
ax.set_xticks(range(na)); ax.set_yticks(range(na))
ax.set_xticklabels(alabels, rotation=45, ha="right", fontsize=18, fontweight="bold")
ax.set_yticklabels(alabels, fontsize=18, fontweight="bold")
ax.tick_params(length=0)
for i in range(na):
    for j in range(na):
        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                fontsize=20, fontweight="bold",
                color="black" if A[i, j] > vmin + 0.12 else "white")
ax.set_title("Matrice d'accord pair-a-pair sur val-dir\n(rouge = tres divergent, vert = tres accord)",
             fontsize=24, fontweight="bold", pad=16)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("prop. de predictions identiques", fontsize=18)
cb.ax.tick_params(labelsize=15)
save(fig, "fig13_agreement_matrix.png")

# ============================================================ DUMP + PRINT
out = {
    "n_val": N, "n_classes": NC, "n_classes_present": int((np.bincount(LABELS, minlength=NC) > 0).sum()),
    "per_model_valdir": {k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "file"}
                         for k, v in per_model.items()},
    "ensembles": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ens.items()},
    "leave_one_out_best_ensemble": loo,
    "tta_directional_tradeoff": tta_tradeoff,
    "pairwise_agreement": {"models": [FILE2NAME[f] for f in agree_order],
                           "matrix": np.round(A, 3).tolist()},
    "best_ensemble": "log_avg(tsm_v2_tta + tsm_v2_rot_tta + vfl_rot_tta)",
    "best_single_file": best_single_file,
    "macro_best_ensemble": {
        "precision": round(float(prec[support > 0].mean()), 4),
        "recall": round(float(rec[support > 0].mean()), 4),
        "f1": round(float(f1[support > 0].mean()), 4),
    },
    "best5_classes": [{"class": CLASS_NAMES[i], "f1": round(float(f1[i]), 3),
                       "support": int(support[i])}
                      for i in np.argsort(f1)[::-1][:5]],
    "worst5_classes": [{"class": CLASS_NAMES[i], "f1": round(float(f1[i]), 3),
                        "support": int(support[i])}
                       for i in np.argsort(f1)[support[np.argsort(f1)] > 0][:5]],
    "top_confused_pairs": [{"true": CLASS_NAMES[i], "pred": CLASS_NAMES[j], "count": c}
                           for c, i, j in top_pairs],
    "checkpoints": ckpt_meta,
}
json.dump(out, open(ROOT / "docs" / "model_stats.json", "w"), indent=2)

print("\n================ VAL-DIR (from-scratch) ================")
for k, v in sorted(per_model.items(), key=lambda kv: -kv[1]["top1"]):
    print(f"  {k:28s} top1={v['top1']*100:5.2f}  top5={v['top5']*100:5.2f}  macroF1={v['macro_f1']*100:5.2f}")
print("\n================ ENSEMBLES ================")
for k, v in ens.items():
    print(f"  {k:48s} top1={v['top1']*100:5.2f}  top5={v['top5']*100:5.2f}")
print("  leave-one-out (best ensemble):", {k: v["delta_pp"] for k, v in loo.items()})
print("\nTTA directional trade-off:", json.dumps(tta_tradeoff))
print("\nbest single:", best_single_file)
print("macro (best ensemble): P/R/F1 =",
      round(float(prec[support>0].mean()),3), round(float(rec[support>0].mean()),3),
      round(float(f1[support>0].mean()),3))
print("\n================ CHECKPOINTS ================")
for m in ckpt_meta:
    print(" ", m)
print("\nWrote docs/model_stats.json and", len(list(FIGS.glob('*.png'))), "figures to docs/figures/")
