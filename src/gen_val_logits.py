"""
Generate and save val-dir logits for a checkpoint.

Usage (from src/):
    python gen_val_logits.py training.checkpoint_path=../models/my_model.pt
    python gen_val_logits.py training.checkpoint_path=../models/my_model.pt dataset.tta=true

Saves logits to ../models/val_logits/<stem>.npy and labels to labels.npy (once).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, set_seed


def load_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device):
    cfg = OmegaConf.create(checkpoint["config"])
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    raw: Dict[str, Any] = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = load_model_from_checkpoint(raw, device)

    pretrained_used = bool(raw.get("pretrained", cfg.model.pretrained))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)

    num_frames = int(raw.get("num_frames", cfg.dataset.num_frames))
    ckpt_cfg = OmegaConf.create(raw["config"]) if "config" in raw else cfg
    interp = bool(ckpt_cfg.dataset.get("interpolate_frames", False))
    use_tta = bool(cfg.dataset.get("tta", False))

    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
        tta=use_tta,
        interpolate_frames=interp,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for video_batch, labels in val_loader:
            if use_tta:
                B, N, T, C, H, W = video_batch.shape
                logits = model(video_batch.to(device).view(B * N, T, C, H, W))
                logits = logits.view(B, N, -1).mean(dim=1)
            else:
                logits = model(video_batch.to(device))

            all_logits.append(logits.cpu().float().numpy())
            all_labels.append(labels.numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    out_dir = checkpoint_path.parent.parent / "models" / "val_logits"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_tta" if use_tta else ""
    stem = checkpoint_path.stem
    logits_path = out_dir / f"{stem}{suffix}.npy"
    np.save(logits_path, logits_arr)

    labels_path = out_dir / "labels.npy"
    np.save(labels_path, labels_arr)

    top1 = float((logits_arr.argmax(1) == labels_arr).mean())
    top5 = float(np.any(np.argsort(logits_arr, axis=1)[:, -5:] == labels_arr[:, None], axis=1).mean())

    print(f"Saved logits ({logits_arr.shape}) → {logits_path}")
    print(f"Top-1: {top1:.4f}  Top-5: {top5:.4f}")


if __name__ == "__main__":
    main()
