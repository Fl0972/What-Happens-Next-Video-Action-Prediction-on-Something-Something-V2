#!/usr/bin/env python3
"""
Ensemble evaluation: average logits from N checkpoints on the val set.

Usage (from src/):

    python evaluate_ensemble.py \
        "+checkpoints=[../models/tsm_ultra.pt,../models/video_former_lite_closed.pt]"

    # With TTA
    python evaluate_ensemble.py \
        "+checkpoints=[../models/tsm_ultra.pt,../models/video_former_lite_closed.pt]" \
        dataset.tta=true

    # Custom weights
    python evaluate_ensemble.py \
        "+checkpoints=[../models/tsm_ultra.pt,../models/video_former_lite_closed.pt]" \
        "+weights=[0.6,0.4]"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from create_ensemble_submission import collect_logits
from create_submission import build_model_from_checkpoint
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import build_transforms, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
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
    val_dir = Path(cfg.dataset.val_dir).resolve()

    weights: List[float] = raw_weights if raw_weights else [1.0] * len(checkpoint_paths)
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    ensemble_logits: Optional[torch.Tensor] = None
    all_labels: Optional[torch.Tensor] = None

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

        val_samples = collect_video_samples(val_dir)
        dataset = VideoFrameDataset(
            root_dir=val_dir,
            num_frames=num_frames,
            transform=transform,
            sample_list=val_samples,
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

        # Collect logits and labels in one pass
        model.eval()
        all_logits_list = []
        labels_list = []
        n_batches = len(loader)
        log_every = max(1, n_batches // 5)

        with torch.no_grad():
            for i, (video_batch, labels) in enumerate(loader, 1):
                video_batch = video_batch.to(device)
                if use_tta:
                    B, N, T, C, H, W = video_batch.shape
                    logits = model(video_batch.view(B * N, T, C, H, W))
                    logits = logits.view(B, N, -1).mean(dim=1)
                else:
                    logits = model(video_batch)
                all_logits_list.append(logits.cpu())
                labels_list.append(labels.cpu())
                if i % log_every == 0 or i == n_batches:
                    print(f"  [{ckpt_path.stem}] batch {i}/{n_batches}", flush=True)

        logits_tensor = torch.cat(all_logits_list, dim=0)
        labels_tensor = torch.cat(labels_list, dim=0)
        print(f"  Logits shape: {tuple(logits_tensor.shape)}", flush=True)

        del model
        torch.cuda.empty_cache()

        if ensemble_logits is None:
            ensemble_logits = weight * logits_tensor
            all_labels = labels_tensor
        else:
            ensemble_logits += weight * logits_tensor

    assert ensemble_logits is not None and all_labels is not None

    correct_top1 = int((ensemble_logits.argmax(1) == all_labels).sum().item())
    _, top5 = ensemble_logits.topk(5, dim=1)
    correct_top5 = int(top5.eq(all_labels.view(-1, 1)).any(dim=1).sum().item())
    total = all_labels.size(0)

    ckpt_names = " + ".join(Path(p).stem for p in checkpoint_paths)
    tta_tag = " (TTA)" if use_tta else ""
    print(f"\nEnsemble: {ckpt_names}{tta_tag}")
    print(f"Validation samples : {total}")
    print(f"Top-1 accuracy     : {correct_top1 / max(total, 1):.4f}")
    print(f"Top-5 accuracy     : {correct_top5 / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
