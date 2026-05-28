"""
Comprehensive ensemble benchmark over all models with val-dir logits.

Strategies tested:
  1. Individual model baselines
  2. All subsets of size 2–4 (uniform avg, acc-weighted avg, log avg, rank vote)
  3. Deduplicated models (one entry per TTA pair): all subsets of size 2–5
  4. Temperature scaling on best 3-model combo (T ∈ {0.5, 0.75, 1.0, 1.5, 2.0})
  5. Grid search over weights (top-3 and top-4 dedup models, step=0.1)
  6. Greedy forward selection (dedup models, up to 8 members)

Run from src/:
    python ensemble_benchmark.py

Writes results to ../docs/ENSEMBLE_EXP.md
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGITS_DIR = Path(__file__).resolve().parent.parent / "models" / "val_logits"
OUT_MD = Path(__file__).resolve().parent.parent / "docs" / "ENSEMBLE_EXP.md"

# --------------------------------------------------------------------------
# Model registry: display name → logits file stem
# --------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, str] = {
    # --- from-scratch models (closed-track valid) ---
    "tsm_v2":          "tsm_ultra_v2",
    "tsm_v2_tta":      "tsm_ultra_v2_tta",
    "tsm_v2_rot":      "tsm_ultra_v2_rotating",
    "tsm_v2_rot_tta":  "tsm_ultra_v2_rotating_tta",
    "tsm":             "tsm_ultra",
    "tsm_tta":         "tsm_ultra_tta",
    "vfl":             "video_former_lite_ultra",
    "vfl_tta":         "video_former_lite_ultra_tta",
    "vfl_focal":       "vfl_ultra_focal",
    "vfl_focal_tta":   "vfl_ultra_focal_tta",
    "vfl_rot":         "video_former_lite_ultra_rotating",
    "vfl_rot_tta":     "video_former_lite_ultra_rotating_tta",
    # --- pretrained ViT models (open-track only) ---
    "vit_bigru":       "vit_bigru_rotating",
    "vit_bigru_tta":   "vit_bigru_rotating_tta",
    "vit_attn":        "vit_bigru_attn_rotating",
    "vit_attn_tta":    "vit_bigru_attn_rotating_tta",
    "vit_perc":        "vit_perceiver_rotating",
    "vit_perc_tta":    "vit_perceiver_rotating_tta",
}

# TTA pairs: for "dedup" experiments keep only the TTA variant when both exist
TTA_PAIRS: Dict[str, str] = {
    "tsm_v2":     "tsm_v2_tta",
    "tsm_v2_rot": "tsm_v2_rot_tta",
    "tsm":        "tsm_tta",
    "vfl":        "vfl_tta",
    "vfl_focal":  "vfl_focal_tta",
    "vfl_rot":    "vfl_rot_tta",
    "vit_bigru":  "vit_bigru_tta",
    "vit_attn":   "vit_attn_tta",
    "vit_perc":   "vit_perc_tta",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_available(logits_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    labels = np.load(logits_dir / "labels.npy")
    logits: Dict[str, np.ndarray] = {}
    for name, stem in MODEL_REGISTRY.items():
        p = logits_dir / f"{stem}.npy"
        if p.exists():
            logits[name] = np.load(p)
    return logits, labels


def dedup_logits(logits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return one entry per model: TTA version preferred over non-TTA."""
    out = dict(logits)
    for base, tta in TTA_PAIRS.items():
        if tta in out and base in out:
            del out[base]
    return out


def top1(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(1) == labels).mean())


def top5(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argsort(logits, axis=1)[:, -5:]
    return float(np.any(preds == labels[:, None], axis=1).mean())


def softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    x = x / T
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def rank_scores(logits: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(logits, axis=1), axis=1).astype(np.float32)


def ensemble_avg(arrs: List[np.ndarray], weights: Optional[List[float]] = None,
                 T: float = 1.0) -> np.ndarray:
    probs = [softmax(a, T) for a in arrs]
    if weights is None:
        weights = [1.0] * len(probs)
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    result = np.zeros_like(probs[0], dtype=np.float64)
    for wi, p in zip(w, probs):
        result += wi * p
    return result


def ensemble_log_avg(arrs: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    log_p = [np.log(softmax(a) + 1e-10) for a in arrs]
    if weights is None:
        weights = [1.0] * len(log_p)
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    result = np.zeros_like(log_p[0], dtype=np.float64)
    for wi, lp in zip(w, log_p):
        result += wi * lp
    return result


def ensemble_rank(arrs: List[np.ndarray]) -> np.ndarray:
    total = np.zeros((arrs[0].shape[0], arrs[0].shape[1]), dtype=np.float32)
    for a in arrs:
        total += rank_scores(a)
    return total


# --------------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------------

def run_all(logits: Dict[str, np.ndarray], labels: np.ndarray) -> List[dict]:
    results: List[dict] = []

    def record(label: str, ens: np.ndarray, strategy: str, models: List[str]) -> float:
        t1 = top1(ens, labels)
        t5 = top5(ens, labels)
        results.append({"label": label, "strategy": strategy, "models": models,
                         "top1": t1, "top5": t5})
        return t1

    all_names = list(logits.keys())
    all_arrs  = [logits[n] for n in all_names]
    model_acc = {n: top1(logits[n], labels) for n in all_names}

    # ----------------------------------------------------------------
    # 1. Individual baselines (all models)
    # ----------------------------------------------------------------
    for name, arr in logits.items():
        record(f"[single] {name}", arr, "baseline", [name])

    # ----------------------------------------------------------------
    # 2. All subsets of size 2–4 (full model set incl TTA variants)
    # ----------------------------------------------------------------
    for r in range(2, 5):
        for combo in itertools.combinations(range(len(all_names)), r):
            cnames = [all_names[i] for i in combo]
            carrs  = [all_arrs[i]  for i in combo]
            weights = [model_acc[n] for n in cnames]
            record("avg(" + "+".join(cnames) + ")",
                   ensemble_avg(carrs), "simple_avg", cnames)
            record("wt(" + "+".join(cnames) + ")",
                   ensemble_avg(carrs, weights), "acc_weighted", cnames)
            record("log(" + "+".join(cnames) + ")",
                   ensemble_log_avg(carrs), "log_avg", cnames)
            record("rank(" + "+".join(cnames) + ")",
                   ensemble_rank(carrs), "rank_vote", cnames)

    # ----------------------------------------------------------------
    # 3. Dedup models (TTA preferred): subsets of size 2–5
    # ----------------------------------------------------------------
    dd = dedup_logits(logits)
    dd_names = list(dd.keys())
    dd_arrs  = [dd[n] for n in dd_names]
    dd_acc   = {n: top1(dd[n], labels) for n in dd_names}

    print(f"  Dedup model set ({len(dd_names)}): {dd_names}")

    for r in range(2, 6):
        for combo in itertools.combinations(range(len(dd_names)), r):
            cnames = [dd_names[i] for i in combo]
            carrs  = [dd_arrs[i]  for i in combo]
            weights = [dd_acc[n] for n in cnames]
            record("dd_avg(" + "+".join(cnames) + ")",
                   ensemble_avg(carrs), "dd_simple_avg", cnames)
            record("dd_wt(" + "+".join(cnames) + ")",
                   ensemble_avg(carrs, weights), "dd_acc_weighted", cnames)
            record("dd_log(" + "+".join(cnames) + ")",
                   ensemble_log_avg(carrs), "dd_log_avg", cnames)
            record("dd_rank(" + "+".join(cnames) + ")",
                   ensemble_rank(carrs), "dd_rank_vote", cnames)

    # ----------------------------------------------------------------
    # 4. Temperature scaling on best 3-model dedup combo
    # ----------------------------------------------------------------
    best3_base = sorted(dd_acc, key=dd_acc.get, reverse=True)[:3]
    best3_arrs = [dd[n] for n in best3_base]
    for T in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        label = f"temp(T={T}," + "+".join(best3_base) + ")"
        record(label, ensemble_avg(best3_arrs, T=T),
               f"temp_T{T}", best3_base)

    # Also temperature on best 5-model dedup combo (by acc)
    best5_base = sorted(dd_acc, key=dd_acc.get, reverse=True)[:5]
    best5_arrs = [dd[n] for n in best5_base]
    for T in [0.5, 0.75, 1.0, 1.5]:
        label = f"temp5(T={T}," + "+".join(best5_base) + ")"
        record(label, ensemble_avg(best5_arrs, T=T),
               f"temp5_T{T}", best5_base)

    # ----------------------------------------------------------------
    # 5. Grid search — top-3 dedup models (step 0.05)
    # ----------------------------------------------------------------
    n0, n1, n2 = best3_base
    a0, a1, a2 = dd[n0], dd[n1], dd[n2]
    best_w3, best_t1_3 = None, 0.0
    grid = [i / 20 for i in range(1, 20)]  # 0.05 .. 0.95
    for w0 in grid:
        for w1 in grid:
            w2 = round(1.0 - w0 - w1, 4)
            if w2 <= 0:
                continue
            t1 = top1(ensemble_avg([a0, a1, a2], [w0, w1, w2]), labels)
            if t1 > best_t1_3:
                best_t1_3 = t1
                best_w3 = (w0, w1, w2)
    if best_w3:
        label = f"grid3({n0}:{best_w3[0]},{n1}:{best_w3[1]},{n2}:{best_w3[2]})"
        record(label, ensemble_avg([a0, a1, a2], list(best_w3)),
               "grid3", best3_base)

    # ----------------------------------------------------------------
    # 6. Grid search — top-4 dedup models (step 0.1)
    # ----------------------------------------------------------------
    top4 = sorted(dd_acc, key=dd_acc.get, reverse=True)[:4]
    n0, n1, n2, n3 = top4
    a0, a1, a2, a3 = dd[n0], dd[n1], dd[n2], dd[n3]
    best_w4, best_t1_4 = None, 0.0
    grid10 = [i / 10 for i in range(1, 9)]
    for w0 in grid10:
        for w1 in grid10:
            for w2 in grid10:
                w3 = round(1.0 - w0 - w1 - w2, 2)
                if w3 <= 0:
                    continue
                t1 = top1(ensemble_avg([a0, a1, a2, a3], [w0, w1, w2, w3]), labels)
                if t1 > best_t1_4:
                    best_t1_4 = t1
                    best_w4 = (w0, w1, w2, w3)
    if best_w4:
        label = f"grid4({n0}:{best_w4[0]},{n1}:{best_w4[1]},{n2}:{best_w4[2]},{n3}:{best_w4[3]})"
        record(label, ensemble_avg([a0, a1, a2, a3], list(best_w4)),
               "grid4", top4)

    # Also grid4 with log avg
    best_w4l, best_t1_4l = None, 0.0
    for w0 in grid10:
        for w1 in grid10:
            for w2 in grid10:
                w3 = round(1.0 - w0 - w1 - w2, 2)
                if w3 <= 0:
                    continue
                t1 = top1(ensemble_log_avg([a0, a1, a2, a3], [w0, w1, w2, w3]), labels)
                if t1 > best_t1_4l:
                    best_t1_4l = t1
                    best_w4l = (w0, w1, w2, w3)
    if best_w4l:
        label = f"grid4log({n0}:{best_w4l[0]},{n1}:{best_w4l[1]},{n2}:{best_w4l[2]},{n3}:{best_w4l[3]})"
        record(label, ensemble_log_avg([a0, a1, a2, a3], list(best_w4l)),
               "grid4_log", top4)

    # ----------------------------------------------------------------
    # 7. Greedy forward selection (dedup, up to 8 members)
    # ----------------------------------------------------------------
    greedy_pool = sorted(dd_acc, key=dd_acc.get, reverse=True)
    selected: List[str] = [greedy_pool[0]]
    pool = greedy_pool[1:]
    print(f"  Greedy start: {selected[0]}  acc={dd_acc[selected[0]]:.4f}")
    for step in range(7):
        best_cand, best_cand_t1 = None, 0.0
        for cand in pool:
            combo_arrs = [dd[n] for n in selected] + [dd[cand]]
            t1 = top1(ensemble_avg(combo_arrs), labels)
            if t1 > best_cand_t1:
                best_cand_t1 = t1
                best_cand = cand
        if best_cand is None:
            break
        selected.append(best_cand)
        pool.remove(best_cand)
        cnames_str = "+".join(selected)
        record(f"greedy({cnames_str})",
               ensemble_avg([dd[n] for n in selected]),
               "greedy", list(selected))
        print(f"    step {step+1}: add {best_cand}  ensemble_top1={best_cand_t1:.4f}")
        if best_cand_t1 < top1(ensemble_avg([dd[n] for n in selected[:-1]]), labels):
            print("    -> no improvement, stopping greedy")
            break

    # Also do greedy with acc-weighted
    selected_w: List[str] = [greedy_pool[0]]
    pool_w = greedy_pool[1:]
    for step in range(7):
        best_cand, best_cand_t1 = None, 0.0
        for cand in pool_w:
            combo = selected_w + [cand]
            combo_arrs = [dd[n] for n in combo]
            weights = [dd_acc[n] for n in combo]
            t1 = top1(ensemble_avg(combo_arrs, weights), labels)
            if t1 > best_cand_t1:
                best_cand_t1 = t1
                best_cand = cand
        if best_cand is None:
            break
        selected_w.append(best_cand)
        pool_w.remove(best_cand)
        cnames_str = "+".join(selected_w)
        record(f"greedy_wt({cnames_str})",
               ensemble_avg([dd[n] for n in selected_w],
                            [dd_acc[n] for n in selected_w]),
               "greedy_weighted", list(selected_w))
        prev = top1(ensemble_avg([dd[n] for n in selected_w[:-1]],
                                  [dd_acc[n] for n in selected_w[:-1]]), labels)
        if best_cand_t1 < prev:
            break

    return results


# --------------------------------------------------------------------------
# Markdown report builder
# --------------------------------------------------------------------------

def build_markdown(results: List[dict], labs: np.ndarray,
                   logits: Dict[str, np.ndarray]) -> str:
    lines: List[str] = []
    a = lines.append

    a("# Ensemble Experiment Report")
    a("")
    a("Evaluated on **val-dir** (6,745 clips, 33 classes).")
    a("Logits generated by `gen_val_logits.py`; experiments by `ensemble_benchmark.py`.")
    a("")

    # ------- Available models -------
    a("## Available Models")
    a("")
    a("| Model key | Val-dir Top-1 | Val-dir Top-5 | Track | Notes |")
    a("|-----------|:---:|:---:|-------|-------|")

    scratch_models = [k for k in MODEL_REGISTRY if not k.startswith("vit")]
    pretrained_models = [k for k in MODEL_REGISTRY if k.startswith("vit")]

    def _row(name: str, track: str, note: str) -> None:
        stem = MODEL_REGISTRY[name]
        p = LOGITS_DIR / f"{stem}.npy"
        if p.exists():
            arr = np.load(p)
            t1 = top1(arr, labs)
            t5 = top5(arr, labs)
            a(f"| `{name}` | {t1:.4f} | {t5:.4f} | {track} | {note} |")
        else:
            a(f"| `{name}` | — | — | {track} | not generated |")

    for n in scratch_models:
        note = "TTA (10-crop)" if "_tta" in n else ("rotating k-fold" if "_rot" in n else "")
        _row(n, "closed", note)
    for n in pretrained_models:
        note = "TTA (10-crop)" if "_tta" in n else "ImageNet pretrained"
        _row(n, "open", note)
    a("")

    # ------- Strategy sections -------
    strategy_headers = {
        "baseline":        ("## 1. Individual Baselines", True),
        "simple_avg":      ("## 2. Simple Average — Full Set (size 2–4)", False),
        "acc_weighted":    ("## 3. Accuracy-Weighted Average — Full Set (size 2–4)", False),
        "log_avg":         ("## 4. Log-Softmax Average — Full Set (size 2–4)", False),
        "rank_vote":       ("## 5. Rank-Vote — Full Set (size 2–4)", False),
        "dd_simple_avg":   ("## 6. Dedup Simple Average (size 2–5)", False),
        "dd_acc_weighted": ("## 7. Dedup Acc-Weighted Average (size 2–5)", False),
        "dd_log_avg":      ("## 8. Dedup Log-Softmax Average (size 2–5)", False),
        "dd_rank_vote":    ("## 9. Dedup Rank-Vote (size 2–5)", False),
        "greedy":          ("## 10. Greedy Forward Selection (uniform)", True),
        "greedy_weighted": ("## 11. Greedy Forward Selection (acc-weighted)", True),
    }

    for strat, (header, show_all) in strategy_headers.items():
        strat_res = [r for r in results if r["strategy"] == strat]
        if not strat_res:
            continue
        strat_res.sort(key=lambda x: -x["top1"])
        a(header)
        a("")
        rows = strat_res if show_all else strat_res[:20]
        a("| Configuration | Top-1 | Top-5 |")
        a("|---------------|:---:|:---:|")
        for r in rows:
            a(f"| `{r['label']}` | {r['top1']:.4f} | {r['top5']:.4f} |")
        if not show_all and len(strat_res) > 20:
            a(f"_… {len(strat_res) - 20} more rows omitted_")
        a("")

    # ------- Temperature scaling -------
    temp3 = sorted([r for r in results if r["strategy"].startswith("temp_T")],
                   key=lambda x: -x["top1"])
    temp5 = sorted([r for r in results if r["strategy"].startswith("temp5_T")],
                   key=lambda x: -x["top1"])
    if temp3:
        a("## 12. Temperature Scaling (top-3 dedup models)")
        a("")
        a("| T | Top-1 | Top-5 |")
        a("|:-:|:---:|:---:|")
        for r in temp3:
            T = r["strategy"].replace("temp_T", "")
            a(f"| {T} | {r['top1']:.4f} | {r['top5']:.4f} |")
        a("")
    if temp5:
        a("## 13. Temperature Scaling (top-5 dedup models)")
        a("")
        a("| T | Top-1 | Top-5 |")
        a("|:-:|:---:|:---:|")
        for r in temp5:
            T = r["strategy"].replace("temp5_T", "")
            a(f"| {T} | {r['top1']:.4f} | {r['top5']:.4f} |")
        a("")

    # ------- Grid search -------
    grid_res = sorted([r for r in results if r["strategy"].startswith("grid")],
                      key=lambda x: -x["top1"])
    if grid_res:
        a("## 14. Grid Search Over Weights")
        a("")
        a("| Configuration | Top-1 | Top-5 |")
        a("|---------------|:---:|:---:|")
        for r in grid_res:
            a(f"| `{r['label']}` | {r['top1']:.4f} | {r['top5']:.4f} |")
        a("")

    # ------- Overall top-30 -------
    top30 = sorted(results, key=lambda x: -x["top1"])[:30]
    a("## Overall Top-30 Configurations")
    a("")
    a("| Rank | Configuration | Strategy | Top-1 | Top-5 |")
    a("|:----:|---------------|----------|:---:|:---:|")
    for i, r in enumerate(top30, 1):
        a(f"| {i} | `{r['label']}` | {r['strategy']} | {r['top1']:.4f} | {r['top5']:.4f} |")
    a("")

    # ------- Best per strategy -------
    a("## Best Per Strategy")
    a("")
    a("| Strategy | Top-1 | Top-5 | Configuration |")
    a("|----------|:---:|:---:|---------------|")
    seen: Dict[str, dict] = {}
    for r in sorted(results, key=lambda x: -x["top1"]):
        if r["strategy"] not in seen:
            seen[r["strategy"]] = r
    for strat, r in sorted(seen.items(), key=lambda kv: -kv[1]["top1"]):
        a(f"| {strat} | {r['top1']:.4f} | {r['top5']:.4f} | `{r['label']}` |")
    a("")

    # ------- Takeaways -------
    best = sorted(results, key=lambda x: -x["top1"])[0]
    best_base = sorted([r for r in results if r["strategy"] == "baseline"],
                       key=lambda x: -x["top1"])[0]
    gain = best["top1"] - best_base["top1"]

    a("## Analysis & Takeaways")
    a("")
    a(f"- **Best overall**: `{best['label']}` → **{best['top1']:.4f}** top-1  |  {best['top5']:.4f} top-5")
    a(f"- **Best single model**: `{best_base['label']}` → {best_base['top1']:.4f}")
    a(f"- **Ensemble gain over best single**: +{gain:.4f} ({gain*100:.2f} pp)")
    a("")
    a("### Architecture diversity")
    a("")
    a("| Group | Models | Inductive bias |")
    a("|-------|--------|----------------|")
    a("| TSM (temporal shift) | tsm_v2, tsm_v2_rot | Shift-based temporal conv, ResNet50 backbone |")
    a("| VFL (video former) | vfl, vfl_focal, vfl_rot | Cross-attention over T=4 frame tokens |")
    a("| ViT-RNN (pretrained) | vit_bigru, vit_attn, vit_perc | ImageNet ViT-B/16 + temporal aggregator |")
    a("")
    a("TSM and VFL are from-scratch (closed-track). ViT models use ImageNet pretraining (open-track).")
    a("The complementary inductive biases (local conv vs. global attention vs. pretrained features)")
    a("explain why cross-group ensembles outperform within-group ones.")
    a("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    if not LOGITS_DIR.exists():
        sys.exit(f"ERROR: {LOGITS_DIR} not found. Run gen_val_logits.py first.")

    print("Loading logits …")
    logits, labels = load_available(LOGITS_DIR)
    if not logits:
        sys.exit("No logits found.")

    print(f"Loaded {len(logits)} model(s): {list(logits.keys())}")
    print("Running experiments …")
    results = run_all(logits, labels)
    print(f"Total experiments: {len(results)}")

    md = build_markdown(results, labels, logits)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md)
    print(f"Report → {OUT_MD}")

    json_out = OUT_MD.parent / "ensemble_results.json"
    json_out.write_text(json.dumps(results, indent=2))
    print(f"Raw JSON → {json_out}")

    best = sorted(results, key=lambda x: -x["top1"])[0]
    print(f"\nBest: {best['label']}")
    print(f"      top1={best['top1']:.4f}  top5={best['top5']:.4f}")


if __name__ == "__main__":
    main()
