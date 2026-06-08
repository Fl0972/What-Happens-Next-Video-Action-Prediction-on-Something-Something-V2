#!/usr/bin/env python3
"""
Per-class weighted ensembling for the Kaggle submission.

For each checkpoint in ``training.checkpoint_paths``:
  1. Run TTA inference on ``dataset.val_dir``  -> per-sample softmax (M, N_val, C)
  2. Run TTA inference on ``dataset.test_dir`` -> per-sample softmax (M, N_test, C)
  3. Cache the softmax tensors so reruns are cheap.

Then for every class ``c`` we compute each model's val accuracy on samples of
true label ``c`` (``acc[m, c]``) and use a temperature-softmax to derive
per-(model, class) weights::

    w[m, c] = softmax_m( acc[m, c] / T )       # T = training.class_weight_temperature

The test prediction is::

    test_prob[v, c] = Σ_m  w[m, c] · softmax_m(test)[v, c]
    pred[v]         = argmax_c  test_prob[v, c]

The script also reports:
  - per-checkpoint val top-1 (single model)
  - val top-1 under uniform soft-vote (baseline)
  - val top-1 under per-class weighted vote (proposed)
  - the (M × C) accuracy and weight matrices

Reference for the broader idea: per-class / per-region model gating is
standard in the model-ensembling / stacking literature (see e.g. §14 of
Wolpert, *Stacked Generalization*, Neural Networks 1992). The exact
temperature-softmax-over-per-class-accuracy recipe used here is the
straightforward formalisation of the "best model per class" heuristic.
"""

from __future__ import annotations

import csv
import hashlib
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
from dataset.video_dataset import VideoFrameDataset, _list_frame_paths, collect_video_samples
from utils import build_flip_perm, build_transforms, set_seed


def _cache_key(ckpt_path: Path, dataset_path: Path, n_clips: int, tta: bool, num_frames: int) -> str:
    """Stable hash over (checkpoint, dataset root, TTA settings)."""
    h = hashlib.sha1()
    h.update(str(ckpt_path).encode())
    h.update(str(ckpt_path.stat().st_mtime_ns).encode())
    h.update(str(dataset_path).encode())
    h.update(f"{n_clips}_{int(tta)}_{num_frames}".encode())
    return h.hexdigest()[:12]


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
    if len(checkpoint_paths) < 2:
        raise SystemExit(
            f"Per-class ensemble needs ≥2 checkpoints; got {len(checkpoint_paths)}. "
            "Pass them via training.checkpoint_paths=[a.pt,b.pt,...]"
        )
    print(f"Per-class weighted ensemble over {len(checkpoint_paths)} checkpoint(s):")
    for p in checkpoint_paths:
        print(f"  - {p}")

    cache_dir_cfg = cfg.training.get("softmax_cache_dir")
    if cache_dir_cfg is None:
        cache_dir_cfg = Path(checkpoint_paths[0]).parent / "_softmax_cache"
    cache_dir = Path(str(cache_dir_cfg)).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Softmax cache dir: {cache_dir}")

    # Architecture from first checkpoint
    first_ckpt: Dict[str, Any] = torch.load(checkpoint_paths[0], map_location="cpu")
    num_classes = int(first_ckpt.get("num_classes", cfg.model.num_classes))
    num_frames = int(first_ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(first_ckpt.get("pretrained", cfg.model.pretrained))
    image_size = int(first_ckpt.get("image_size", cfg.dataset.get("image_size", 224)))
    eval_transform = build_transforms(image_size=image_size, is_training=False, use_imagenet_norm=pretrained)

    use_tta = bool(cfg.dataset.get("tta", True))
    n_clips = max(1, int(cfg.dataset.get("n_clips", 1)))
    n_views = (10 if use_tta else 1) * n_clips
    multi_view = n_views > 1
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
        print(f"Label-aware TTA: remapping {len(flip_pairs_dict)} class indices on flipped views.")

    # ── Build val + test loaders ───────────────────────────────────────────
    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    val_labels = torch.tensor([lbl for _, lbl in val_samples], dtype=torch.long)
    val_dataset = VideoFrameDataset(
        root_dir=val_dir, num_frames=num_frames, transform=eval_transform,
        sample_list=val_samples, tta=use_tta, n_clips=n_clips,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    test_root = Path(cfg.dataset.test_dir).resolve()
    test_names, test_dirs = discover_all_test_videos(test_root)
    valid_test_names, valid_test_dirs, empty_test_names = [], [], []
    for name, p in zip(test_names, test_dirs):
        if _list_frame_paths(p):
            valid_test_names.append(name)
            valid_test_dirs.append(p)
        else:
            empty_test_names.append(name)
    if empty_test_names:
        print(f"Warning: {len(empty_test_names)} empty test folders skipped.")
    test_sample_list = [(p, 0) for p in valid_test_dirs]
    test_dataset = VideoFrameDataset(
        root_dir=test_root, num_frames=num_frames, transform=eval_transform,
        sample_list=test_sample_list, tta=use_tta, n_clips=n_clips,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    print(f"Val: {len(val_dataset)} clips | Test: {len(test_dataset)} clips")

    # ── Per-checkpoint val + test inference (with cache) ───────────────────
    # Model is (re)built per *uncached* checkpoint from its own stored config
    # (see note in the loop). We must NOT switch on model_name: Base and Large
    # both save model_name="videomae", so a model_name-keyed switch would load
    # weights into the wrong-size graph and crash when mixing backbones.
    model = None

    val_probs_per_ckpt: List[torch.Tensor] = []
    test_probs_per_ckpt: List[torch.Tensor] = []

    for i, ckpt_path in enumerate(checkpoint_paths, start=1):
        print(f"\n[{i}/{len(checkpoint_paths)}] {ckpt_path.name}")
        val_key  = _cache_key(ckpt_path, val_dir,   n_clips, use_tta, num_frames)
        test_key = _cache_key(ckpt_path, test_root, n_clips, use_tta, num_frames)
        val_cache  = cache_dir / f"{ckpt_path.stem}_val_{val_key}.pt"
        test_cache = cache_dir / f"{ckpt_path.stem}_test_{test_key}.pt"

        need_val  = not val_cache.exists()
        need_test = not test_cache.exists()

        if need_val or need_test:
            ckpt = first_ckpt if i == 1 else torch.load(ckpt_path, map_location="cpu")
            model = build_model_from_checkpoint(ckpt).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

        if need_val:
            print("  computing val softmax...", flush=True)
            val_p = run_inference_logits(
                model, val_loader, device, total_videos=len(val_dataset),
                multi_view=multi_view, use_amp=use_amp, amp_dtype=amp_dtype,
                eval_view_chunk=eval_view_chunk,
                flip_perm=flip_perm, spatial_tta=use_tta,
            )
            torch.save(val_p, val_cache)
        else:
            print(f"  [val cache hit] {val_cache.name}")
            val_p = torch.load(val_cache, map_location="cpu")

        if need_test:
            print("  computing test softmax...", flush=True)
            test_p = run_inference_logits(
                model, test_loader, device, total_videos=len(test_dataset),
                multi_view=multi_view, use_amp=use_amp, amp_dtype=amp_dtype,
                eval_view_chunk=eval_view_chunk,
                flip_perm=flip_perm, spatial_tta=use_tta,
            )
            torch.save(test_p, test_cache)
        else:
            print(f"  [test cache hit] {test_cache.name}")
            test_p = torch.load(test_cache, map_location="cpu")

        per_ckpt_val_top1 = (val_p.argmax(dim=1) == val_labels).float().mean().item()
        print(f"  single-model val top-1: {per_ckpt_val_top1:.4f}")
        val_probs_per_ckpt.append(val_p)
        test_probs_per_ckpt.append(test_p)

    val_probs  = torch.stack(val_probs_per_ckpt,  dim=0)   # (M, N_val, C)
    test_probs = torch.stack(test_probs_per_ckpt, dim=0)   # (M, N_test, C)
    M = val_probs.shape[0]

    # ── Per-class accuracy ─────────────────────────────────────────────────
    val_pred_per_m = val_probs.argmax(dim=2)               # (M, N_val)
    acc_mc  = torch.zeros(M, num_classes)
    count_c = torch.zeros(num_classes, dtype=torch.long)
    for c in range(num_classes):
        mask = (val_labels == c)
        n_c = int(mask.sum().item())
        count_c[c] = n_c
        if n_c == 0:
            continue
        for m in range(M):
            acc_mc[m, c] = (val_pred_per_m[m, mask] == c).float().mean()

    # ── Per-class weights via temperature softmax ──────────────────────────
    T = float(cfg.training.get("class_weight_temperature", 0.10))
    weights = torch.zeros(M, num_classes)
    for c in range(num_classes):
        if count_c[c] == 0:
            weights[:, c] = 1.0 / M
        else:
            weights[:, c] = torch.softmax(acc_mc[:, c] / max(T, 1e-9), dim=0)

    # ── Diagnostics ────────────────────────────────────────────────────────
    print(f"\nPer-class accuracy and weights (T={T:.3f}):")
    header = f"  {'c':>3}  {'n':>5} | " + "  ".join(f"acc_{m}" for m in range(M)) + "  | " + "  ".join(f"w_{m}" for m in range(M))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in range(num_classes):
        accs = "  ".join(f"{acc_mc[m, c].item():.3f}" for m in range(M))
        ws   = "  ".join(f"{weights[m, c].item():.3f}" for m in range(M))
        print(f"  {c:3d}  {int(count_c[c]):5d} | {accs}  | {ws}")

    # ── Sanity check: uniform vs per-class weighted on val ────────────────
    uniform_val_probs  = val_probs.mean(dim=0)                                 # (N_val, C)
    weighted_val_probs = (val_probs * weights.unsqueeze(1)).sum(dim=0)         # (N_val, C)
    uniform_top1  = (uniform_val_probs.argmax(dim=1)  == val_labels).float().mean().item()
    weighted_top1 = (weighted_val_probs.argmax(dim=1) == val_labels).float().mean().item()

    print("\nVal top-1 summary:")
    for m in range(M):
        per_m = (val_probs[m].argmax(dim=1) == val_labels).float().mean().item()
        print(f"  single  ({checkpoint_paths[m].name}): {per_m:.4f}")
    print(f"  uniform soft-vote        : {uniform_top1:.4f}")
    print(f"  per-class weighted (T={T}) : {weighted_top1:.4f}  "
          f"(Δ vs uniform = {(weighted_top1 - uniform_top1) * 100:+.2f} pp)")

    if weighted_top1 < uniform_top1:
        print("\nWARNING: per-class weighting hurts val accuracy. Try raising the "
              "temperature (training.class_weight_temperature) towards 0.3–1.0 to "
              "soften the weighting, or fall back to uniform soft-vote.")

    # ── Write submission ───────────────────────────────────────────────────
    if bool(cfg.training.get("skip_submission", False)):
        print("\nskip_submission=true; not writing CSV.")
        return

    test_combined = (test_probs * weights.unsqueeze(1)).sum(dim=0)             # (N_test, C)
    test_preds = test_combined.argmax(dim=1).tolist()

    output_path = Path(str(cfg.dataset.submission_output)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_map = dict(zip(valid_test_names, test_preds))
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name in test_names:
            w.writerow([name, pred_map.get(name, 0)])

    print(f"\nWrote {len(test_preds)} per-class-weighted predictions to {output_path}")


if __name__ == "__main__":
    main()
