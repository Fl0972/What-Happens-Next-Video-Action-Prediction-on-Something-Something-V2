#!/usr/bin/env python3
"""Combine per-model cached *test* softmax tensors (each computed on the model's
own preprocessing — V-JEPA 16f/256 on val2_win16, k400/TSM 4f/224 on shipped
val2) into a single uniform-mean ensemble submission CSV.

Why this is needed: the existing ensemble scripts assume one input format for
all models. The honest ensemble crosses preprocessing — each model must infer
on its own appropriate data — so we cache once per model and combine the cached
softmax tensors. Test videos share folder names across val2/test and
val2_win16/test, so sorted ordering aligns indices.
"""
from __future__ import annotations
import argparse
import csv
import glob
from pathlib import Path
from typing import List, Optional

import torch

from create_submission import discover_all_test_videos
from dataset.video_dataset import _list_frame_paths


def find_cache(stem: str, dirs: List[Path]) -> Optional[Path]:
    for d in dirs:
        matches = sorted(glob.glob(str(d / f"{stem}_test_*.pt")))
        if matches:
            return Path(matches[-1])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="+", required=True, help="checkpoint stems (filename without .pt)")
    ap.add_argument("--cache-dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--test-dir", type=Path, required=True, help="canonical test_dir for video-name ordering")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    test_names, test_dirs = discover_all_test_videos(args.test_dir)
    valid_names = [n for n, p in zip(test_names, test_dirs) if _list_frame_paths(p)]

    softmaxes = []
    for stem in args.stems:
        f = find_cache(stem, args.cache_dirs)
        if f is None:
            raise SystemExit(f"no test softmax cache for {stem!r} under {args.cache_dirs}")
        s = torch.load(f, map_location="cpu")
        if s.ndim != 2 or s.shape[0] != len(valid_names):
            raise SystemExit(f"{stem}: cache shape {tuple(s.shape)} doesn't match {len(valid_names)} test names")
        print(f"  {stem}: {f.name} shape={tuple(s.shape)}")
        softmaxes.append(s)

    combined = torch.stack(softmaxes).mean(dim=0)
    preds = combined.argmax(dim=1).tolist()
    pred_map = dict(zip(valid_names, preds))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for n in test_names:
            w.writerow([n, pred_map.get(n, 0)])
    print(f"\nwrote {len(test_names)} predictions ({len(args.stems)}-model uniform mean) to {args.out}")


if __name__ == "__main__":
    main()
