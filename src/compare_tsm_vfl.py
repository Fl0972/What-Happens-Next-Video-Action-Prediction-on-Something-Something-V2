"""
Two-model comparison: TSM-Ultra-v2 vs VideoFormerLite-Ultra.

Computes overall and per-class metrics for each model alone and for several
ensembling techniques combining the two. All numbers are derived directly from
the saved validation logits in models/val_logits/ (val_dir, 6745 clips, 33
classes), so they are reproducible and not hand-copied.

Run from src/:  python compare_tsm_vfl.py
Outputs:        docs/tsm_vs_vfl_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGITS = ROOT / "models" / "val_logits"
ANALYSIS = ROOT / "docs" / "analysis_results.json"
OUT = ROOT / "docs" / "tsm_vs_vfl_stats.json"


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def rankavg(prob):
    # average of per-clip rank of each class (higher rank = more confident)
    order = prob.argsort(axis=1).argsort(axis=1)  # 0..C-1, C-1 best
    return order.astype(np.float64)


def metrics(pred, labels, n_classes):
    top1 = float((pred == labels).mean())
    f1 = {}
    macro = []
    for c in range(n_classes):
        tp = int(((pred == c) & (labels == c)).sum())
        fp = int(((pred == c) & (labels != c)).sum())
        fn = int(((pred != c) & (labels == c)).sum())
        sup = int((labels == c).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fc = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1[c] = {"precision": prec, "recall": rec, "f1": fc, "support": sup, "tp": tp}
        if sup > 0:
            macro.append(fc)
    return top1, float(np.mean(macro)), f1


def topk(scores, labels, k=5):
    idx = scores.argsort(axis=1)[:, -k:]
    return float(np.any(idx == labels[:, None], axis=1).mean())


def main():
    labels = np.load(LOGITS / "labels.npy")
    n = labels.shape[0]
    n_classes = 33

    # class index -> readable name
    pc = json.load(open(ANALYSIS))["per_class"]
    names = {}
    for key in pc:
        head = key.split("_")[0]
        if not head.isdigit():
            continue
        names[int(head)] = key[4:].replace("_", " ")

    variants = {
        "tsm": "tsm_ultra_v2.npy",
        "tsm_tta": "tsm_ultra_v2_tta.npy",
        "vfl": "video_former_lite_ultra.npy",
        "vfl_tta": "video_former_lite_ultra_tta.npy",
    }
    raw = {k: np.load(LOGITS / v).astype(np.float64) for k, v in variants.items()}
    prob = {k: softmax(v) for k, v in raw.items()}
    logp = {k: log_softmax(v) for k, v in raw.items()}

    results = {"n_val": n, "n_classes": n_classes, "names": names,
               "singles": {}, "ensembles": {}, "per_class": {}}

    # ---- singles (no TTA, the subjects of the two analysis docs) ----
    for tag in ["tsm", "vfl"]:
        t1, mf1, f1 = metrics(prob[tag].argmax(1), labels, n_classes)
        results["singles"][tag] = {
            "top1": t1, "top5": topk(prob[tag], labels), "macro_f1": mf1}
        results["per_class"][tag] = f1

    # ---- ensembling techniques on the two NON-TTA singles ----
    a, b = "tsm", "vfl"
    ens = {}

    # 1. logit averaging (mean raw logits)
    ens["logit_avg"] = (raw[a] + raw[b]) / 2
    # 2. probability (softmax) averaging — arithmetic mean of probs
    ens["prob_avg"] = (prob[a] + prob[b]) / 2
    # 3. log-prob averaging — geometric mean of probs (mean of log-softmax)
    ens["logprob_avg"] = (logp[a] + logp[b]) / 2
    # 4. max-confidence — per clip take the model that is more peaked
    conf_a = prob[a].max(1, keepdims=True)
    conf_b = prob[b].max(1, keepdims=True)
    ens["max_conf"] = np.where(conf_a >= conf_b, prob[a], prob[b])
    # 5. rank averaging
    ens["rank_avg"] = rankavg(prob[a]) + rankavg(prob[b])

    for name, sc in ens.items():
        t1, mf1, f1 = metrics(sc.argmax(1), labels, n_classes)
        results["ensembles"][name] = {
            "top1": t1, "top5": topk(sc, labels), "macro_f1": mf1}
        results["per_class"][f"ens_{name}"] = f1

    # weight sweep for log-prob averaging (best technique typically)
    sweep = {}
    best = (-1, None)
    for w in np.round(np.arange(0.0, 1.01, 0.1), 2):
        sc = w * logp[a] + (1 - w) * logp[b]
        t1, _, _ = metrics(sc.argmax(1), labels, n_classes)
        sweep[f"{w:.1f}"] = round(t1, 4)
        if t1 > best[0]:
            best = (t1, float(w))
    results["weight_sweep_logprob"] = sweep
    results["weight_sweep_best"] = {"w_tsm": best[1], "top1": round(best[0], 4)}

    # same ensembles on the TTA variants, for reference
    ra, rb = raw["tsm_tta"], raw["vfl_tta"]
    pa, pb = prob["tsm_tta"], prob["vfl_tta"]
    la, lb = logp["tsm_tta"], logp["vfl_tta"]
    tta = {
        "logit_avg": (ra + rb) / 2,
        "prob_avg": (pa + pb) / 2,
        "logprob_avg": (la + lb) / 2,
    }
    results["ensembles_tta"] = {}
    for name, sc in tta.items():
        t1, mf1, _ = metrics(sc.argmax(1), labels, n_classes)
        results["ensembles_tta"][name] = {
            "top1": t1, "top5": topk(sc, labels), "macro_f1": mf1}

    # disagreement / complementarity on non-TTA singles
    pa_, pb_ = prob[a].argmax(1), prob[b].argmax(1)
    both = ((pa_ == labels) & (pb_ == labels)).sum()
    only_a = ((pa_ == labels) & (pb_ != labels)).sum()
    only_b = ((pa_ != labels) & (pb_ == labels)).sum()
    neither = ((pa_ != labels) & (pb_ != labels)).sum()
    agree = (pa_ == pb_).mean()
    results["complementarity"] = {
        "both_correct": int(both), "only_tsm": int(only_a),
        "only_vfl": int(only_b), "neither": int(neither),
        "agreement_rate": float(agree),
        "oracle_top1": float((both + only_a + only_b) / n)}

    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT)
    print("singles:", {k: round(v["top1"], 4) for k, v in results["singles"].items()})
    print("ensembles (noTTA):", {k: round(v["top1"], 4) for k, v in results["ensembles"].items()})
    print("ensembles (TTA):", {k: round(v["top1"], 4) for k, v in results["ensembles_tta"].items()})
    print("best logprob weight:", results["weight_sweep_best"])
    print("complementarity:", results["complementarity"])


if __name__ == "__main__":
    main()
