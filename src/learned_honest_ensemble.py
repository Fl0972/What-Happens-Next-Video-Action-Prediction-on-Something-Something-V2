#!/usr/bin/env python3
"""Learned-weight honest ensemble across cross-preprocessing models.

For each checkpoint: load its cached val + test softmax (each computed on the
model's own preprocessing — V-JEPA-pseudo on val2_win16, k400/TSM on shipped
val2). Fit per-model softmax-normalised weights by Adam on val NLL, then apply
the same weights to test and write the submission CSV. Per-class mode also
available (M*C params, with L2-toward-uniform regularisation).

Why a dedicated script: ensemble_gradient.py uses a single data loader for all
checkpoints, so it can't mix preprocessing. Here we just load the cached
softmax tensors (already produced by the appropriate per-model inference) and
fit the combination weights on top.
"""
from __future__ import annotations
import argparse
import csv
import glob
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from create_submission import discover_all_test_videos
from dataset.video_dataset import _list_frame_paths, collect_video_samples


def find_cache(stem: str, split: str, dirs: List[Path]) -> Optional[Path]:
    for d in dirs:
        m = sorted(glob.glob(str(d / f"{stem}_{split}_*.pt")))
        if m:
            return Path(m[-1])
    return None


def fit_weights(val_probs: torch.Tensor, val_labels: torch.Tensor,
                mode: str, steps: int, lr: float, l2: float) -> torch.Tensor:
    M, _, C = val_probs.shape
    theta = (torch.zeros(M) if mode == "global" else torch.zeros(M, C)).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    eps = 1e-8
    for _ in range(steps):
        opt.zero_grad()
        w = torch.softmax(theta, dim=0)
        if mode == "global":
            combined = (w.view(M, 1, 1) * val_probs).sum(dim=0)
        else:
            combined = (w.unsqueeze(1) * val_probs).sum(dim=0)
        nll = F.nll_loss(torch.log(combined.clamp_min(eps)), val_labels)
        reg = l2 * ((w - 1.0 / M) ** 2).sum()
        (nll + reg).backward()
        opt.step()
    with torch.no_grad():
        return torch.softmax(theta, dim=0).detach()


def combine(w: torch.Tensor, probs: torch.Tensor, mode: str) -> torch.Tensor:
    M = probs.shape[0]
    if mode == "global":
        return (w.view(M, 1, 1) * probs).sum(dim=0)
    return (w.unsqueeze(1) * probs).sum(dim=0)


def top1(combined: torch.Tensor, labels: torch.Tensor) -> float:
    return (combined.argmax(dim=1) == labels).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--cache-dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--val-dir", type=Path, required=True, help="val_dir with class subdirs (labels from numeric prefix)")
    ap.add_argument("--test-dir", type=Path, required=True, help="test_dir for canonical video-name ordering")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["global", "per_class"], default="global")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2-uniform", type=float, default=0.001)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    args = ap.parse_args()

    val_samples = collect_video_samples(args.val_dir)
    val_labels = torch.tensor([l for _, l in val_samples], dtype=torch.long)

    test_names, test_dirs = discover_all_test_videos(args.test_dir)
    valid_test = [n for n, p in zip(test_names, test_dirs) if _list_frame_paths(p)]

    val_list, test_list = [], []
    for stem in args.stems:
        vf = find_cache(stem, "val", args.cache_dirs)
        tf = find_cache(stem, "test", args.cache_dirs)
        if vf is None:
            raise SystemExit(f"no val softmax cache for {stem}")
        if tf is None:
            raise SystemExit(f"no test softmax cache for {stem}")
        v = torch.load(vf, map_location="cpu")
        t = torch.load(tf, map_location="cpu")
        if v.shape[0] != len(val_labels):
            raise SystemExit(f"{stem} val rows {v.shape[0]} != #val labels {len(val_labels)}")
        if t.shape[0] != len(valid_test):
            raise SystemExit(f"{stem} test rows {t.shape[0]} != #valid test {len(valid_test)}")
        print(f"  {stem:42} val={vf.name} | test={tf.name}")
        val_list.append(v); test_list.append(t)

    val = torch.stack(val_list)    # (M, N_val, C)
    test = torch.stack(test_list)  # (M, N_test, C)
    M = val.shape[0]

    print("\n=== single-model val top-1 ===")
    for i, stem in enumerate(args.stems):
        print(f"  {stem:42} {top1(val[i], val_labels):.4f}")
    uni_val = top1(val.mean(dim=0), val_labels)
    print(f"  uniform mean                                {uni_val:.4f}")

    # honest holdout check
    if args.holdout_frac > 0:
        N = val.shape[1]
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(N, generator=g)
        nh = max(1, int(round(args.holdout_frac * N)))
        hold, fit = perm[:nh], perm[nh:]
        w_cv = fit_weights(val[:, fit], val_labels[fit], args.mode, args.steps, args.lr, args.l2_uniform)
        uni_hold = top1(val[:, hold].mean(0), val_labels[hold])
        lrn_hold = top1(combine(w_cv, val[:, hold], args.mode), val_labels[hold])
        print(f"\nholdout (n={nh}): uniform={uni_hold:.4f}  learned-{args.mode}={lrn_hold:.4f}  delta={(lrn_hold-uni_hold)*100:+.2f}pp")

    # final fit on all val
    w = fit_weights(val, val_labels, args.mode, args.steps, args.lr, args.l2_uniform)
    fit_val = top1(combine(w, val, args.mode), val_labels)
    print(f"\nfit on all val: learned-{args.mode} top1 = {fit_val:.4f}  (Δ vs uniform = {(fit_val-uni_val)*100:+.2f}pp)")

    if args.mode == "global":
        print("\nlearned global weights:")
        for s, ww in zip(args.stems, w.tolist()):
            print(f"  {ww:.4f}  {s}")
    else:
        print("\nlearned per-class weights — per-model mean:")
        for i, s in enumerate(args.stems):
            print(f"  mean={w[i].mean().item():.4f}  [{w[i].min().item():.3f}, {w[i].max().item():.3f}]  {s}")

    combined = combine(w, test, args.mode)
    preds = combined.argmax(dim=1).tolist()
    pred_map = dict(zip(valid_test, preds))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["video_name", "predicted_class"])
        for n in test_names:
            wcsv.writerow([n, pred_map.get(n, 0)])
    print(f"\nwrote {len(test_names)} predictions to {args.out}")


if __name__ == "__main__":
    main()
