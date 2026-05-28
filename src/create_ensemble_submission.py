#!/usr/bin/env python3
"""
Late-fusion ensemble: average logits from N checkpoints, then argmax.

Each checkpoint reconstructs its own model and DataLoader (respecting its
own num_frames), runs inference on the test set, and contributes weighted
logits to the final prediction.

Usage (from src/):

    # Equal weights with log-softmax aggregation (recommended)
    python create_ensemble_submission.py \
        "+checkpoints=[../models/tsm_ultra_v2.pt,../models/tsm_ultra_v2_rotating.pt]" \
        dataset.submission_output=../submissions/ensemble.csv \
        dataset.tta=true +log_softmax=true

    # Per-model TTA (some with, some without)
    python create_ensemble_submission.py \
        "+checkpoints=[../models/tsm_ultra_v2.pt,../models/vit_bigru_rotating.pt]" \
        "+per_model_tta=[true,false]" \
        dataset.submission_output=../submissions/ensemble.csv \
        +log_softmax=true

    # Custom weights
    python create_ensemble_submission.py \
        "+checkpoints=[../models/tsm_ultra.pt,../models/video_former_lite_closed.pt]" \
        "+weights=[0.6,0.4]" \
        dataset.submission_output=../submissions/ensemble.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from create_submission import (
    build_model_from_checkpoint,
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
)
from dataset.video_dataset import VideoFrameDataset, _list_frame_paths
from torch.utils.data import DataLoader
from utils import build_transforms, set_seed


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool,
    label: str,
) -> torch.Tensor:
    """Run inference and return raw logits as a (N, num_classes) CPU tensor."""
    model.eval()
    all_logits: List[torch.Tensor] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 5)

    for i, (video_batch, _) in enumerate(loader, 1):
        video_batch = video_batch.to(device)
        if tta:
            B, N, T, C, H, W = video_batch.shape
            logits = model(video_batch.view(B * N, T, C, H, W))
            logits = logits.view(B, N, -1).mean(dim=1)
        else:
            logits = model(video_batch)
        all_logits.append(logits.cpu())
        if i % log_every == 0 or i == n_batches:
            print(f"  [{label}] batch {i}/{n_batches}", flush=True)

    return torch.cat(all_logits, dim=0)   # (N, num_classes)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # ------------------------------------------------------------------ setup
    checkpoint_paths: List[str] = list(cfg.get("checkpoints", []))
    if not checkpoint_paths:
        raise SystemExit(
            "No checkpoints provided.\n"
            'Pass them with: +checkpoints="[path1.pt,path2.pt]"'
        )

    raw_weights: Optional[List[float]] = list(cfg.get("weights", [])) or None
    if raw_weights is not None and len(raw_weights) != len(checkpoint_paths):
        raise SystemExit(
            f"Got {len(checkpoint_paths)} checkpoints but {len(raw_weights)} weights."
        )

    # Per-model TTA overrides dataset.tta when provided
    per_model_tta_cfg = cfg.get("per_model_tta", None)
    per_model_tta: Optional[List[bool]] = None
    if per_model_tta_cfg is not None:
        per_model_tta = [bool(v) for v in per_model_tta_cfg]
        if len(per_model_tta) != len(checkpoint_paths):
            raise SystemExit(
                f"per_model_tta has {len(per_model_tta)} entries but "
                f"{len(checkpoint_paths)} checkpoints."
            )

    use_log_softmax: bool = bool(cfg.get("log_softmax", False))

    set_seed(int(cfg.dataset.seed))
    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    global_tta = bool(cfg.dataset.get("tta", False))
    output_path = Path(cfg.dataset.submission_output).resolve()
    test_root = Path(cfg.dataset.test_dir).resolve()

    # ------------------------------------------------ discover test videos
    manifest_cfg = cfg.dataset.get("test_manifest")
    print(f"Indexing test videos under: {test_root}", flush=True)
    if manifest_cfg:
        video_names = load_manifest_video_names(Path(str(manifest_cfg)).resolve())
        video_dirs = resolve_video_dirs(test_root, video_names)
    else:
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
        print(f"Warning: {len(empty_names)} empty video folder(s) → assigned class 0.")

    sample_list: List[Tuple[Path, int]] = [(p, 0) for p in valid_dirs]

    # ------------------------------------------------ per-checkpoint inference
    ensemble_acc: Optional[torch.Tensor] = None  # accumulated log-probs or weighted logits
    weights: List[float] = raw_weights if raw_weights else [1.0] * len(checkpoint_paths)
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]   # normalise to sum=1

    agg_mode = "log-softmax" if use_log_softmax else "weighted-logit"
    print(f"Aggregation: {agg_mode}")
    print(f"Global TTA: {global_tta}" + (" (overridden per model)" if per_model_tta else ""))

    for idx, (ckpt_path_str, weight) in enumerate(zip(checkpoint_paths, weights)):
        ckpt_path = Path(ckpt_path_str).resolve()
        use_tta = per_model_tta[idx] if per_model_tta is not None else global_tta
        print(f"\n[{idx+1}/{len(checkpoint_paths)}] {ckpt_path.name}"
              f"  weight={weight:.3f}  tta={use_tta}", flush=True)

        ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model_from_checkpoint(ckpt)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
        pretrained = bool(ckpt.get("pretrained", False))
        ckpt_cfg = OmegaConf.create(ckpt["config"]) if "config" in ckpt else cfg
        interp = bool(ckpt_cfg.dataset.get("interpolate_frames", False))
        transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)

        # Use smaller batch for TTA to avoid OOM
        batch_size = int(cfg.training.batch_size)
        if use_tta:
            batch_size = min(batch_size, 2)

        dataset = VideoFrameDataset(
            root_dir=test_root,
            num_frames=num_frames,
            transform=transform,
            sample_list=sample_list,
            tta=use_tta,
            interpolate_frames=interp,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(cfg.training.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        print(f"  frames={num_frames}  pretrained={pretrained}  bs={batch_size}", flush=True)

        logits = collect_logits(model, loader, device, tta=use_tta, label=ckpt_path.stem)
        print(f"  Logits shape: {tuple(logits.shape)}", flush=True)

        del model
        torch.cuda.empty_cache()

        if use_log_softmax:
            contribution = weight * F.log_softmax(logits.float(), dim=1)
        else:
            contribution = weight * logits.float()

        if ensemble_acc is None:
            ensemble_acc = contribution
        else:
            ensemble_acc += contribution

    assert ensemble_acc is not None
    predictions = ensemble_acc.argmax(dim=1).tolist()

    # ------------------------------------------------ write CSV
    if len(predictions) != len(valid_names):
        raise RuntimeError(
            f"Prediction count {len(predictions)} != valid video count {len(valid_names)}"
        )
    pred_map = dict(zip(valid_names, predictions))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name in video_names:
            w.writerow([name, pred_map.get(name, 0)])

    print(f"\nWrote {len(predictions)} rows to {output_path}", flush=True)
    ckpt_names = " + ".join(Path(p).stem for p in checkpoint_paths)
    tta_label = "(per-model TTA)" if per_model_tta else ("(TTA)" if global_tta else "")
    print(f"Ensemble: {ckpt_names} {tta_label}  [{agg_mode}]", flush=True)


if __name__ == "__main__":
    main()
