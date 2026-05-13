#!/usr/bin/env python3
"""
Late-fusion ensemble: average logits from N checkpoints, then argmax.

Each checkpoint reconstructs its own model and DataLoader (respecting its
own num_frames), runs inference on the test set, and contributes weighted
logits to the final prediction.

Usage (from src/):

    # Equal weights (default — recommended when unsure)
    python create_ensemble_submission.py \
        "+checkpoints=[../models/tsm_ultra.pt,../models/video_former_lite_closed.pt]" \
        dataset.submission_output=../submissions/ensemble.csv

    # With TTA (10-crop averaging per model)
    python create_ensemble_submission.py \
        "+checkpoints=[../models/tsm_ultra.pt,../models/video_former_lite_closed.pt]" \
        dataset.submission_output=../submissions/ensemble_tta.csv \
        dataset.tta=true

    # Custom weights (proportional — they are L1-normalised internally)
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

    set_seed(int(cfg.dataset.seed))
    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    use_tta = bool(cfg.dataset.get("tta", False))
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
    ensemble_logits: Optional[torch.Tensor] = None
    weights: List[float] = raw_weights if raw_weights else [1.0] * len(checkpoint_paths)
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]   # normalise to sum=1

    for ckpt_path_str, weight in zip(checkpoint_paths, weights):
        ckpt_path = Path(ckpt_path_str).resolve()
        print(f"\nLoading checkpoint ({weight:.3f} weight): {ckpt_path}", flush=True)

        ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
        model = build_model_from_checkpoint(ckpt)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
        pretrained = bool(ckpt.get("pretrained", False))
        transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)

        dataset = VideoFrameDataset(
            root_dir=test_root,
            num_frames=num_frames,
            transform=transform,
            sample_list=sample_list,
            tta=use_tta,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            num_workers=int(cfg.training.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        if use_tta:
            print(f"  TTA enabled (10 crops). frames={num_frames}", flush=True)

        label = ckpt_path.stem
        logits = collect_logits(model, loader, device, tta=use_tta, label=label)
        print(f"  Logits shape: {tuple(logits.shape)}", flush=True)

        del model
        torch.cuda.empty_cache()

        if ensemble_logits is None:
            ensemble_logits = weight * logits
        else:
            ensemble_logits += weight * logits

    assert ensemble_logits is not None
    predictions = ensemble_logits.argmax(dim=1).tolist()

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
    print(f"Ensemble: {ckpt_names} {'(TTA)' if use_tta else ''}", flush=True)


if __name__ == "__main__":
    main()
