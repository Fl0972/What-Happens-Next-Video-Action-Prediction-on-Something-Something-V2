#!/usr/bin/env python3
"""
Learned (gradient-descent) weighted soft-vote ensembling.

A third ensembling mode alongside the uniform soft-vote
(``create_submission.py``) and the heuristic per-class temperature weights
(``ensemble_per_class.py``). Instead of *deriving* the ensemble weights from a
formula, we *learn* them by minimising validation cross-entropy with Adam.

For M checkpoints, N val videos, C classes, the cached per-model softmax
tensors give ``val_probs (M, N, C)`` and ``test_probs (M, N, C)`` — the exact
same cache files that ``ensemble_per_class.py`` produces, so this reuses them.
We optimise logit-weights θ, normalised with a softmax over the *model* axis so
the combination is always a convex mixture (hence a valid probability)::

    global    : w = softmax_m(θ),      θ ∈ R^M      combined[n,c] = Σ_m w[m]   · val_probs[m,n,c]
    per_class : w = softmax_m(θ[:,c]), θ ∈ R^{M×C}  combined[n,c] = Σ_m w[m,c] · val_probs[m,n,c]

    loss = NLL(log combined, y) + λ · ‖w − 1/M‖²      (λ = training.grad_l2_uniform)

Knobs (Hydra CLI / config, with defaults):
    training.grad_weight_mode   global | per_class   (default: global)
    training.grad_steps         Adam steps           (default: 300)
    training.grad_lr            Adam lr              (default: 0.05)
    training.grad_l2_uniform    pull-to-uniform λ    (default: 1e-3)
    training.grad_holdout_frac  honest-eval split    (default: 0.0 = fit on all val)

Overfitting note: ``global`` has only M params (≈zero overfit on a real val
set) and is the safe default. ``per_class`` has M×C params and can overfit val;
use grad_l2_uniform and grad_holdout_frac to keep it honest. With
grad_holdout_frac>0 we additionally fit on a val subset and report accuracy on
the untouched remainder — an honest generalisation estimate — before refitting
on all of val for the actual test submission.

Technique: learning a convex combination of model outputs by minimising a
held-out loss is "linear stacking" (Breiman, *Stacked Regressions*, 1996;
Wolpert, *Stacked Generalization*, Neural Networks 1992). Optimising the
mixture directly is the learned counterpart to the heuristic in
``ensemble_per_class.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from create_submission import (
    _resolve_checkpoint_paths,
    build_model_from_checkpoint,
    discover_all_test_videos,
    run_inference_logits,
)
from dataset.video_dataset import VideoFrameDataset, _list_frame_paths, collect_video_samples
from ensemble_per_class import _cache_key
from utils import build_flip_perm, build_transforms, set_seed


def _combine(w: torch.Tensor, probs: torch.Tensor, mode: str) -> torch.Tensor:
    """Convex-combine per-model softmax. probs: (M, N, C); returns (N, C).

    w is (M,) for ``global`` or (M, C) for ``per_class``.
    """
    M = probs.shape[0]
    if mode == "global":
        return (w.view(M, 1, 1) * probs).sum(dim=0)
    return (w.unsqueeze(1) * probs).sum(dim=0)  # (M,1,C) * (M,N,C) -> sum_M -> (N,C)


def _top1(combined: torch.Tensor, labels: torch.Tensor) -> float:
    return (combined.argmax(dim=1) == labels).float().mean().item()


def _fit_weights(
    val_probs: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    mode: str,
    steps: int,
    lr: float,
    l2_uniform: float,
) -> torch.Tensor:
    """Optimise softmax-normalised mixture weights to minimise val NLL.

    Returns w detached: (M,) for ``global`` or (M, C) for ``per_class``.
    Runs on CPU — the parameter count (M or M×C) and tensors are tiny, so this
    costs nothing and never touches the GPU the trainer is using.
    """
    M, _, C = val_probs.shape
    theta = (torch.zeros(M) if mode == "global" else torch.zeros(M, C)).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    eps = 1e-8
    for _ in range(steps):
        opt.zero_grad()
        w = torch.softmax(theta, dim=0)
        combined = _combine(w, val_probs, mode)            # (N, C)
        nll = F.nll_loss(torch.log(combined.clamp_min(eps)), val_labels)
        reg = l2_uniform * ((w - 1.0 / M) ** 2).sum()
        (nll + reg).backward()
        opt.step()
    with torch.no_grad():
        return torch.softmax(theta, dim=0).detach()


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
            f"Learned ensemble needs ≥2 checkpoints; got {len(checkpoint_paths)}. "
            "Pass them via training.checkpoint_paths=[a.pt,b.pt,...]"
        )

    mode = str(cfg.training.get("grad_weight_mode", "global")).lower()
    if mode not in ("global", "per_class"):
        raise SystemExit(f"grad_weight_mode must be 'global' or 'per_class', got {mode!r}")
    steps = int(cfg.training.get("grad_steps", 300))
    lr = float(cfg.training.get("grad_lr", 0.05))
    l2_uniform = float(cfg.training.get("grad_l2_uniform", 1e-3))
    holdout_frac = float(cfg.training.get("grad_holdout_frac", 0.0))

    print(f"Learned ({mode}) weighted ensemble over {len(checkpoint_paths)} checkpoint(s):")
    for p in checkpoint_paths:
        print(f"  - {p}")
    print(f"  steps={steps} lr={lr} l2_uniform={l2_uniform} holdout_frac={holdout_frac}")

    cache_dir_cfg = cfg.training.get("softmax_cache_dir")
    if cache_dir_cfg is None:
        cache_dir_cfg = Path(checkpoint_paths[0]).parent / "_softmax_cache"
    cache_dir = Path(str(cache_dir_cfg)).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Softmax cache dir: {cache_dir}")

    # Architecture from first checkpoint. weights_only=False: these are trusted,
    # self-produced checkpoints that carry a config dict (torch>=2.6 would else
    # refuse the non-tensor payload).
    first_ckpt: Dict[str, Any] = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    num_classes = int(first_ckpt.get("num_classes", cfg.model.num_classes))
    num_frames = int(first_ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(first_ckpt.get("pretrained", cfg.model.pretrained))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)

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

    # ── Per-checkpoint val + test inference (shared cache) ──────────────────
    # Build the model lazily, per *uncached* checkpoint, from that checkpoint's
    # own stored config. We deliberately do NOT switch-by-model_name like the
    # sibling scripts: ssv2/k400 (Base) and the Large model all save
    # model_name="videomae", so a model_name-keyed switch would try to load
    # Large weights into a Base graph (size mismatch). Rebuilding is ~1s and
    # only happens for checkpoints whose softmax isn't already cached.
    model = None

    val_probs_per_ckpt: List[torch.Tensor] = []
    test_probs_per_ckpt: List[torch.Tensor] = []

    for i, ckpt_path in enumerate(checkpoint_paths, start=1):
        print(f"\n[{i}/{len(checkpoint_paths)}] {ckpt_path.name}")
        val_key = _cache_key(ckpt_path, val_dir, n_clips, use_tta, num_frames)
        test_key = _cache_key(ckpt_path, test_root, n_clips, use_tta, num_frames)
        val_cache = cache_dir / f"{ckpt_path.stem}_val_{val_key}.pt"
        test_cache = cache_dir / f"{ckpt_path.stem}_test_{test_key}.pt"

        need_val = not val_cache.exists()
        need_test = not test_cache.exists()

        if need_val or need_test:
            ckpt = first_ckpt if i == 1 else torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

        print(f"  single-model val top-1: {_top1(val_p, val_labels):.4f}")
        val_probs_per_ckpt.append(val_p)
        test_probs_per_ckpt.append(test_p)

    val_probs = torch.stack(val_probs_per_ckpt, dim=0)    # (M, N_val, C)
    test_probs = torch.stack(test_probs_per_ckpt, dim=0)  # (M, N_test, C)
    M = val_probs.shape[0]

    # ── Baselines ──────────────────────────────────────────────────────────
    uniform_top1 = _top1(val_probs.mean(dim=0), val_labels)

    # ── Honest holdout estimate (optional) ─────────────────────────────────
    if holdout_frac > 0.0:
        N = val_probs.shape[1]
        gen = torch.Generator().manual_seed(int(cfg.dataset.seed))
        perm = torch.randperm(N, generator=gen)
        n_hold = max(1, int(round(holdout_frac * N)))
        hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]
        w_cv = _fit_weights(
            val_probs[:, fit_idx], val_labels[fit_idx],
            mode=mode, steps=steps, lr=lr, l2_uniform=l2_uniform,
        )
        fit_top1 = _top1(_combine(w_cv, val_probs[:, fit_idx], mode), val_labels[fit_idx])
        hold_top1 = _top1(_combine(w_cv, val_probs[:, hold_idx], mode), val_labels[hold_idx])
        # The honest delta compares learned vs uniform on the SAME holdout subset
        # (comparing holdout-learned against full-val-uniform mixes sample sets
        # and is misleading). Note the holdout is small, so treat ±1-2 pp as noise.
        uni_hold_top1 = _top1(val_probs[:, hold_idx].mean(dim=0), val_labels[hold_idx])
        print(
            f"\nHoldout check (frac={holdout_frac}, n_hold={n_hold}):"
            f"\n  learned: fit top1={fit_top1:.4f}  holdout top1={hold_top1:.4f}"
            f"\n  uniform on same holdout : {uni_hold_top1:.4f}"
            f"\n  -> honest learned Δ vs uniform (same holdout) = "
            f"{(hold_top1 - uni_hold_top1) * 100:+.2f} pp"
        )

    # ── Final weights: fit on ALL val for the actual test submission ───────
    w = _fit_weights(
        val_probs, val_labels, mode=mode, steps=steps, lr=lr, l2_uniform=l2_uniform,
    )
    learned_top1 = _top1(_combine(w, val_probs, mode), val_labels)

    # ── Report ─────────────────────────────────────────────────────────────
    print("\nVal top-1 summary:")
    for m in range(M):
        print(f"  single ({checkpoint_paths[m].name}): {_top1(val_probs[m], val_labels):.4f}")
    print(f"  uniform soft-vote        : {uniform_top1:.4f}")
    print(
        f"  learned ({mode})         : {learned_top1:.4f}  "
        f"(Δ vs uniform = {(learned_top1 - uniform_top1) * 100:+.2f} pp)"
    )
    if holdout_frac == 0.0:
        print(
            "  NOTE: learned top-1 is fit and measured on the same val set "
            "(optimistic). Set training.grad_holdout_frac>0 for an honest estimate."
        )

    if mode == "global":
        print("\nLearned global weights:")
        for m in range(M):
            print(f"  {w[m].item():.4f}  {checkpoint_paths[m].name}")
    else:
        print("\nLearned per-class weights — per-model mean (min/max across classes):")
        for m in range(M):
            print(
                f"  mean={w[m].mean().item():.4f}  "
                f"[{w[m].min().item():.3f}, {w[m].max().item():.3f}]  "
                f"{checkpoint_paths[m].name}"
            )

    # ── Write submission ───────────────────────────────────────────────────
    if bool(cfg.training.get("skip_submission", False)):
        print("\nskip_submission=true; not writing CSV.")
        return

    test_combined = _combine(w, test_probs, mode)  # (N_test, C)
    test_preds = test_combined.argmax(dim=1).tolist()

    output_path = Path(str(cfg.dataset.submission_output)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_map = dict(zip(valid_test_names, test_preds))
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "predicted_class"])
        for name in test_names:
            writer.writerow([name, pred_map.get(name, 0)])

    print(f"\nWrote {len(test_preds)} learned-{mode}-weighted predictions to {output_path}")


if __name__ == "__main__":
    main()
