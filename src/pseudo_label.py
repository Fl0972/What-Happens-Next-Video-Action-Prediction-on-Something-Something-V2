#!/usr/bin/env python3
"""
Generate pseudo-labels for the unlabeled test set, for semi-supervised
self-training (a.k.a. pseudo-labeling).

The script:

  1. Loads one or several trained checkpoints (same resolution rules as
     ``create_submission.py``: ``training.checkpoint_paths`` or auto-discovery
     of top-K snapshots from ``training.checkpoint_path``).
  2. Runs TTA inference over every video under ``dataset.test_dir`` and
     averages softmax probabilities across views and checkpoints.
  3. Writes a CSV at ``dataset.pseudo_labels_output`` with one row per test
     video::

         video_path,pseudo_label,confidence

  4. Prints a confidence histogram so you can pick a sensible threshold.

To then USE the resulting pseudo-labels for training, point train.py at it::

    python train.py experiment=videomae \\
        dataset.pseudo_labels_path=/abs/path/to/pseudo_labels.csv \\
        dataset.pseudo_threshold=0.85

Reference:
    Lee, "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning
    Method for Deep Neural Networks", ICML Workshop on Challenges in
    Representation Learning (2013).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from create_submission import (
    _resolve_checkpoint_paths,
    build_model_from_checkpoint,
    discover_all_test_videos,
    run_inference_logits,
)
from dataset.video_dataset import VideoFrameDataset, _list_frame_paths
from utils import build_flip_perm, build_transforms, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_paths = _resolve_checkpoint_paths(cfg)
    missing = [p for p in checkpoint_paths if not p.is_file()]
    if missing:
        raise SystemExit(f"Checkpoint(s) not found: {missing}")
    print(f"Pseudo-labeling with {len(checkpoint_paths)} checkpoint(s):")
    for p in checkpoint_paths:
        print(f"  - {p}")

    first_ckpt: Dict[str, Any] = torch.load(checkpoint_paths[0], map_location="cpu")
    num_frames = int(first_ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(first_ckpt.get("pretrained", cfg.model.pretrained))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)
    num_classes = int(first_ckpt.get("num_classes", cfg.model.num_classes))

    test_root = Path(cfg.dataset.test_dir).resolve()
    print(f"Indexing video folders under: {test_root}", flush=True)
    video_names, video_dirs = discover_all_test_videos(test_root)
    print(f"Found {len(video_dirs)} test videos.", flush=True)

    valid_names: List[str] = []
    valid_dirs: List[Path] = []
    empty_names: List[str] = []
    for name, p in zip(video_names, video_dirs):
        if _list_frame_paths(p):
            valid_names.append(name)
            valid_dirs.append(p)
        else:
            empty_names.append(name)
    if empty_names:
        print(f"Warning: {len(empty_names)} folders have no frames; skipped.",
              flush=True)

    # Placeholder label 0; only the input matters.
    sample_list = [(p, 0) for p in valid_dirs]

    use_tta = bool(cfg.dataset.get("tta", True))
    n_clips = max(1, int(cfg.dataset.get("n_clips", 1)))
    n_views = (10 if use_tta else 1) * n_clips
    multi_view = n_views > 1
    if multi_view:
        print(f"TTA enabled: {n_views} views/video "
              f"({n_clips} temporal clip(s) × {10 if use_tta else 1} spatial crop(s)).",
              flush=True)

    dataset = VideoFrameDataset(
        root_dir=test_root,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=sample_list,
        tta=use_tta,
        n_clips=n_clips,
    )
    batch_size = int(cfg.training.batch_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model_from_checkpoint(first_ckpt)
    model.to(device)

    use_amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    amp_dtype_str = str(cfg.training.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    eval_view_chunk = max(1, int(cfg.training.get("eval_view_chunk", 4)))

    flip_pairs_dict = first_ckpt.get("flip_pairs") or {}
    flip_pairs_dict = {int(k): int(v) for k, v in flip_pairs_dict.items()}
    flip_perm = (
        build_flip_perm(num_classes, flip_pairs_dict).to(device)
        if (use_tta and flip_pairs_dict) else None
    )
    if flip_perm is not None:
        print(f"Label-aware TTA: remapping {len(flip_pairs_dict)} class indices "
              f"on flipped views.", flush=True)

    summed_probs: torch.Tensor | None = None
    for i, ckpt_path in enumerate(checkpoint_paths, start=1):
        print(f"[{i}/{len(checkpoint_paths)}] Loading {ckpt_path.name}", flush=True)
        ckpt = first_ckpt if i == 1 else torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        probs = run_inference_logits(
            model, loader, device, total_videos=len(dataset),
            multi_view=multi_view, use_amp=use_amp, amp_dtype=amp_dtype,
            eval_view_chunk=eval_view_chunk,
            flip_perm=flip_perm, spatial_tta=use_tta,
        )
        summed_probs = probs if summed_probs is None else summed_probs + probs

    assert summed_probs is not None
    avg_probs = summed_probs / len(checkpoint_paths)             # (N, num_classes)
    confidences, pseudo_labels = avg_probs.max(dim=1)             # (N,) (N,)

    # ---- Diagnostics -----------------------------------------------------
    print("\nConfidence distribution:", flush=True)
    for thr in [0.30, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        kept = int((confidences >= thr).sum().item())
        pct = 100.0 * kept / max(1, len(valid_names))
        print(f"  threshold ≥ {thr:.2f}: keep {kept:6d} / {len(valid_names)}  ({pct:5.1f}%)")

    # Per-class counts at the suggested threshold (0.85), plus a "lite" 0.70
    for thr in (0.85, 0.70):
        mask = confidences >= thr
        if mask.any():
            print(f"\nPer-class pseudo-label counts at threshold ≥ {thr}:")
            for c in range(num_classes):
                n_c = int(((pseudo_labels == c) & mask).sum().item())
                print(f"  class {c:2d}: {n_c:5d}")

    # ---- Write CSV --------------------------------------------------------
    output_path = cfg.dataset.get("pseudo_labels_output")
    if output_path is None:
        output_path = (
            Path(cfg.training.checkpoint_path).parent / "pseudo_labels.csv"
        )
    output_path = Path(str(output_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "pseudo_label", "confidence"])
        for path, lbl, conf in zip(valid_dirs, pseudo_labels.tolist(), confidences.tolist()):
            w.writerow([str(path), int(lbl), f"{float(conf):.6f}"])

    print(f"\nWrote {len(valid_dirs)} pseudo-labels to {output_path}", flush=True)
    print("Next step:")
    print(f"  python train.py experiment=videomae \\")
    print(f"      dataset.pseudo_labels_path={output_path} \\")
    print(f"      dataset.pseudo_threshold=0.85")


if __name__ == "__main__":
    main()
