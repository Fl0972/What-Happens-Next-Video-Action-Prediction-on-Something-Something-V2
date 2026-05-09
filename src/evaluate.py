"""
Evaluate a saved checkpoint on the full validation split.
Reports top-1 and top-5 accuracy. Supports TTA (dataset.tta=true).

Example (from src/)::

    python evaluate.py
    python evaluate.py dataset.tta=true
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, set_seed


def load_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    if "config" not in checkpoint or checkpoint["config"] is None:
        raise ValueError("Checkpoint has no 'config' entry.")
    cfg = OmegaConf.create(checkpoint["config"])
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    raw: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model = load_model_from_checkpoint(raw, device)

    pretrained_used = bool(raw.get("pretrained", cfg.model.pretrained))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)

    # Use ALL of val_dir — no train/val split here
    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)

    num_frames = int(raw.get("num_frames", cfg.dataset.num_frames))
    use_tta = bool(cfg.dataset.get("tta", False))
    if use_tta:
        print("TTA enabled: averaging over 10 crops.")

    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
        tta=use_tta,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for video_batch, labels in val_loader:
            labels = labels.to(device)

            if use_tta:
                # video_batch: (B, 10, T, C, H, W) — average logits over 10 crops
                B, N, T, C, H, W = video_batch.shape
                logits = model(video_batch.to(device).view(B * N, T, C, H, W))
                logits = logits.view(B, N, -1).mean(dim=1)
            else:
                logits = model(video_batch.to(device))

            correct_top1 += int((logits.argmax(1) == labels).sum().item())

            _, top5 = logits.topk(5, dim=1)
            correct_top5 += int(top5.eq(labels.view(-1, 1)).any(dim=1).sum().item())

            total += labels.size(0)

    print(f"Validation samples: {total}")
    print(f"Top-1 accuracy: {correct_top1 / max(total, 1):.4f}")
    print(f"Top-5 accuracy: {correct_top5 / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
