#!/usr/bin/env python3
"""Head-to-head: TSM-Ultra-v2 vs VideoFormerLite-Ultra and their 2-model ensembles.

Reads the two models' cached val-dir logits (models/val_logits/*.npy), computes
per-model and per-class metrics, then evaluates a panel of 2-model ensembling
techniques (raw-logit avg, softmax-prob avg, log-prob / geometric-mean avg,
weighted-logit sweep, max-confidence routing, rank averaging). Also reports the
oracle ceiling (>=1 model correct) that bounds any fusion of the two.

Writes:
  - docs/tsm_vs_vfl_stats.json
  - docs/figures/fig_cmp_*.png

Run from project root:  python src/analyze_tsm_vs_vfl.py
No GPU / dataset / re-inference needed — works entirely off cached logits.
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
    pc = json.load(open(ROOT / "docs" / "analysis_results.json"))["per_class"]
    out = []
    for k in pc:
        head = k.split("_")[0]
        if head.isdigit() and "_" in k:
            out.append(k.split("_", 1)[1].replace("_", " ").strip())
        else:
            out.append(k.replace("_", " "))
    assert len(out) == 33
    return out

CLASS_NAMES = load_class_names()
LABELS = np.load(VLOG / "labels.npy")
N, NC = LABELS.shape[0], 33

T = np.load(VLOG / "tsm_ultra_v2.npy").astype(np.float64)            # TSM-Ultra-v2
V = np.load(VLOG / "video_former_lite_ultra.npy").astype(np.float64) # VFL-Ultra
T_tta = np.load(VLOG / "tsm_ultra_v2_tta.npy").astype(np.float64)
V_tta = np.load(VLOG / "video_former_lite_ultra_tta.npy").astype(np.float64)

# ------------------------------------------------------------------- helpers
def log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

def softmax(x):
    return np.exp(log_softmax(x))

def topk_acc(logits, k=5):
    topk = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    return float(np.any(topk == LABELS[:, None], axis=1).mean())

def per_class_prf(pred):
    cm = np.zeros((NC, NC), dtype=np.int64)
    np.add.at(cm, (LABELS, pred), 1)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    pred_tot = cm.sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_tot > 0, tp / pred_tot, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1, support, cm

def metrics(logits_or_pred, is_pred=False):
    if is_pred:
        pred = logits_or_pred
        top5 = None
    else:
        pred = logits_or_pred.argmax(1)
        top5 = topk_acc(logits_or_pred, 5)
    prec, rec, f1, support, _ = per_class_prf(pred)
    present = support > 0
    return {
        "top1": float((pred == LABELS).mean()),
        "top5": top5,
        "macro_precision": float(prec[present].mean()),
        "macro_recall": float(rec[present].mean()),
        "macro_f1": float(f1[present].mean()),
    }

def rank_scores(logits):
    # higher logit -> higher rank (0..NC-1); average over models then argmax
    order = logits.argsort(axis=1)            # ascending
    ranks = np.empty_like(order)
    rows = np.arange(logits.shape[0])[:, None]
    ranks[rows, order] = np.arange(NC)[None, :]
    return ranks.astype(np.float64)

# ============================================================ TECHNIQUES
def technique_preds(A, B):
    """Return dict: technique name -> predictions (and fused scores where useful)."""
    sA, sB = softmax(A), softmax(B)
    lA, lB = log_softmax(A), log_softmax(B)
    out = {}
    out["raw_logit_avg"]   = (0.5 * (A + B)).argmax(1)
    out["softmax_prob_avg"] = (sA + sB).argmax(1)
    out["logprob_avg(geo-mean)"] = (lA + lB).argmax(1)
    out["rank_avg"]        = (rank_scores(A) + rank_scores(B)).argmax(1)
    # max-confidence routing: per-sample take the more confident model's prediction
    routeA = sA.max(1) >= sB.max(1)
    pr = np.where(routeA, sA.argmax(1), sB.argmax(1))
    out["max_confidence_route"] = pr
    return out

# weighted-logit sweep (on log-prob to be scale-fair): w*lT + (1-w)*lV
def weighted_sweep(A, B):
    lA, lB = log_softmax(A), log_softmax(B)
    ws = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    accs = []
    for w in ws:
        pred = (w * lA + (1 - w) * lB).argmax(1)
        accs.append(float((pred == LABELS).mean()))
    return ws, np.array(accs)

# ---- run all on base (no-TTA) pair -------------------------------------
res = {}
res["TSM-Ultra-v2 (single)"] = metrics(T)
res["VFL-Ultra (single)"]    = metrics(V)
for name, pred in technique_preds(T, V).items():
    res[name] = metrics(pred, is_pred=True)

# weighted sweep
ws, sweep = weighted_sweep(T, V)
best_w = float(ws[sweep.argmax()])
res[f"weighted_logprob (w*TSM, best w={best_w})"] = {"top1": float(sweep.max()),
    "top5": None, "macro_precision": None, "macro_recall": None, "macro_f1": None}

# ---- same panel WITH TTA ------------------------------------------------
res_tta = {}
res_tta["TSM-Ultra-v2 +TTA (single)"] = metrics(T_tta)
res_tta["VFL-Ultra +TTA (single)"]    = metrics(V_tta)
for name, pred in technique_preds(T_tta, V_tta).items():
    res_tta[name + " +TTA"] = metrics(pred, is_pred=True)

# ============================================================ DIVERSITY / ORACLE
predT, predV = T.argmax(1), V.argmax(1)
cT, cV = predT == LABELS, predV == LABELS
agreement = float((predT == predV).mean())
both = float((cT & cV).mean())
onlyT = float((cT & ~cV).mean())
onlyV = float((~cT & cV).mean())
neither = float((~cT & ~cV).mean())
oracle = float((cT | cV).mean())          # ceiling for any 2-model fusion
diversity = {
    "agreement_rate": round(agreement, 4),
    "both_correct": round(both, 4),
    "only_TSM_correct": round(onlyT, 4),
    "only_VFL_correct": round(onlyV, 4),
    "neither_correct": round(neither, 4),
    "oracle_at_least_one_correct": round(oracle, 4),
    "tsm_single_top1": round(float(cT.mean()), 4),
    "vfl_single_top1": round(float(cV.mean()), 4),
}

# logit-scale diagnostic (why raw-logit avg is unfair)
scale = {
    "TSM_logit_std": round(float(T.std()), 3),
    "VFL_logit_std": round(float(V.std()), 3),
    "TSM_mean_max_logit": round(float(T.max(1).mean()), 3),
    "VFL_mean_max_logit": round(float(V.max(1).mean()), 3),
}

# ============================================================ PER-CLASS
_, _, f1T, supportT, _ = per_class_prf(predT)
_, _, f1V, _, _ = per_class_prf(predV)
best_pred = (log_softmax(T) + log_softmax(V)).argmax(1)   # log-prob avg = best technique
_, _, f1E, support, _ = per_class_prf(best_pred)

present = support > 0
per_class = []
for i in range(NC):
    if not present[i]:
        continue
    per_class.append({
        "class": CLASS_NAMES[i], "support": int(support[i]),
        "f1_TSM": round(float(f1T[i]), 3),
        "f1_VFL": round(float(f1V[i]), 3),
        "f1_ens": round(float(f1E[i]), 3),
        "VFL_minus_TSM": round(float(f1V[i] - f1T[i]), 3),
        "ens_minus_bestsingle": round(float(f1E[i] - max(f1T[i], f1V[i])), 3),
    })

vfl_wins = sorted([c for c in per_class if c["VFL_minus_TSM"] > 0],
                  key=lambda c: -c["VFL_minus_TSM"])[:8]
tsm_wins = sorted(per_class, key=lambda c: c["VFL_minus_TSM"])[:8]
ens_helps = sorted(per_class, key=lambda c: -c["ens_minus_bestsingle"])[:8]

# ============================================================ FIGURES
def save(fig, name):
    fig.savefig(FIGS / name); plt.close(fig); print("  wrote", name)

# A. ensemble-technique comparison (base pair)
order_names = ["VFL-Ultra (single)", "TSM-Ultra-v2 (single)", "raw_logit_avg",
               "max_confidence_route", "rank_avg", "softmax_prob_avg",
               "logprob_avg(geo-mean)"]
vals = [res[n]["top1"] * 100 for n in order_names]
labels = ["VFL\n(single)", "TSM-v2\n(single)", "raw-logit\navg", "max-conf\nroute",
          "rank\navg", "softmax\nprob avg", "log-prob\navg (geo)"]
colors = ["#6a3d9a", "#2b7bba"] + ["#fdae6b"] * 3 + ["#fd8d3c", "#e6550d"]
fig, ax = plt.subplots(figsize=(9.5, 5))
b = ax.bar(labels, vals, color=colors)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width()/2, v + 0.1, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.axhline(res["TSM-Ultra-v2 (single)"]["top1"] * 100, ls="--", color="#2b7bba", lw=.8)
ax.set_ylabel("Val-dir top-1 (%)"); ax.set_ylim(30, 42)
ax.set_title("TSM-v2 + VFL: 2-model ensembling techniques (no TTA)")
ax.grid(axis="y", alpha=.3)
save(fig, "fig_cmp_techniques.png")

# B. weighted-logit sweep curve
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(ws, sweep * 100, "-o", color="#e6550d", ms=4)
ax.axvline(best_w, ls="--", color="#888", lw=.8)
ax.scatter([best_w], [sweep.max() * 100], color="#c0392b", zorder=5,
           label=f"best w={best_w} ({sweep.max()*100:.2f}%)")
ax.text(0.02, res["VFL-Ultra (single)"]["top1"]*100, "←  all VFL", fontsize=8, va="bottom")
ax.text(0.98, res["TSM-Ultra-v2 (single)"]["top1"]*100, "all TSM  →", fontsize=8, va="bottom", ha="right")
ax.set_xlabel("weight on TSM-v2  (w);  VFL weight = 1−w")
ax.set_ylabel("Val-dir top-1 (%)")
ax.set_title("Weighted log-prob blend: top-1 vs mixing weight")
ax.legend(); ax.grid(alpha=.3)
save(fig, "fig_cmp_weight_sweep.png")

# C. per-class F1: TSM vs VFL vs ensemble (sorted by ensemble F1)
idx = [i for i in range(NC) if present[i]]
idx = sorted(idx, key=lambda i: f1E[i])
y = np.arange(len(idx)); h = 0.27
fig, ax = plt.subplots(figsize=(8.5, 10))
ax.barh(y - h, [f1T[i] for i in idx], h, label="TSM-Ultra-v2", color="#2b7bba")
ax.barh(y,     [f1V[i] for i in idx], h, label="VFL-Ultra", color="#6a3d9a")
ax.barh(y + h, [f1E[i] for i in idx], h, label="log-prob ensemble", color="#e6550d")
ax.set_yticks(y); ax.set_yticklabels([CLASS_NAMES[i] for i in idx], fontsize=7)
ax.set_xlabel("F1 (val-dir)"); ax.legend(loc="lower right")
ax.set_title("Per-class F1: TSM-v2 vs VFL vs their log-prob ensemble")
ax.grid(axis="x", alpha=.3)
save(fig, "fig_cmp_perclass_f1.png")

# D. diversity / oracle stacked bar
fig, ax = plt.subplots(figsize=(7.5, 3.2))
segs = [("both correct", both, "#2ca02c"), ("only TSM", onlyT, "#2b7bba"),
        ("only VFL", onlyV, "#6a3d9a"), ("neither", neither, "#bbbbbb")]
left = 0
for name, frac, col in segs:
    ax.barh(0, frac * 100, left=left * 100, color=col, label=f"{name} ({frac*100:.1f}%)")
    left += frac
ax.axvline(oracle * 100, color="k", lw=1.2, ls="--")
ax.text(oracle * 100 - 1, 0.42, f"oracle ceiling = {oracle*100:.1f}%", ha="right", fontsize=8)
ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("% of 6,745 val-dir clips")
ax.set_title("Error overlap of TSM-v2 and VFL (decorrelation drives the ensemble)")
ax.legend(ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.55))
save(fig, "fig_cmp_diversity.png")

# ============================================================ DUMP
out = {
    "n_val": N, "n_classes": NC,
    "models": {"TSM-Ultra-v2": "models/val_logits/tsm_ultra_v2.npy",
               "VFL-Ultra": "models/val_logits/video_former_lite_ultra.npy"},
    "param_split": {
        "TSM-Ultra-v2": {"total_M": 11.20, "temporal_params_pct": 0.0, "spatial_pct": 99.8,
                         "temporal_mechanism": "parameter-free channel shift (fold_div=4)"},
        "VFL-Ultra": {"total_M": 17.51, "temporal_params_pct": 36.0, "spatial_pct": 63.9,
                      "temporal_mechanism": "2-layer global self-attention (6.31M params)"},
    },
    "logit_scale": scale,
    "diversity": diversity,
    "techniques_base": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                            for kk, vv in v.items()} for k, v in res.items()},
    "techniques_tta": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                           for kk, vv in v.items()} for k, v in res_tta.items()},
    "weighted_sweep": {"w_on_tsm": ws.tolist(), "top1": np.round(sweep, 4).tolist(),
                       "best_w": best_w, "best_top1": round(float(sweep.max()), 4)},
    "per_class": per_class,
    "vfl_wins_classes": vfl_wins,
    "tsm_wins_classes": tsm_wins,
    "ensemble_helps_classes": ens_helps,
}
json.dump(out, open(ROOT / "docs" / "tsm_vs_vfl_stats.json", "w"), indent=2)

# ============================================================ PRINT
print("\n=== SINGLE + ENSEMBLE TECHNIQUES (base, no TTA) ===")
for k, v in sorted(res.items(), key=lambda kv: (kv[1]["top1"] is None, kv[1]["top1"])):
    t5 = f" top5={v['top5']*100:5.2f}" if v["top5"] else ""
    mf = f" macroF1={v['macro_f1']*100:5.2f}" if v["macro_f1"] else ""
    print(f"  {k:42s} top1={v['top1']*100:5.2f}{t5}{mf}")
print("\n=== WITH TTA ===")
for k, v in sorted(res_tta.items(), key=lambda kv: kv[1]["top1"]):
    print(f"  {k:42s} top1={v['top1']*100:5.2f}")
print("\n=== DIVERSITY ===", json.dumps(diversity, indent=2))
print("=== LOGIT SCALE ===", json.dumps(scale))
print(f"\nbest weighted-logprob: w_on_TSM={best_w} -> {sweep.max()*100:.2f}%")
print("\nVFL beats TSM on:", [(c['class'], c['VFL_minus_TSM']) for c in vfl_wins])
print("ensemble helps most on:", [(c['class'], c['ens_minus_bestsingle']) for c in ens_helps])
print("\nWrote docs/tsm_vs_vfl_stats.json and 4 figures.")
