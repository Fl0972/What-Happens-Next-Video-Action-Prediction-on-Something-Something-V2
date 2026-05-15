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
from torch.amp import autocast
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, set_seed


def _multi_view_logits(
    model: torch.nn.Module,
    video_batch: torch.Tensor,
    max_views_per_step: int,
) -> torch.Tensor:
    """video_batch: (B, N, T, C, H, W) -> logits: (B, num_classes).

    Splits the (B*N) views into chunks of ``max_views_per_step`` to avoid OOM
    when N is large (e.g. 30 = 3 temporal clips × 10 spatial crops).
    """
    B, N, T, C, H, W = video_batch.shape
    flat = video_batch.reshape(B * N, T, C, H, W)
    out_chunks = []
    for s in range(0, B * N, max(1, max_views_per_step)):
        out_chunks.append(model(flat[s : s + max_views_per_step]))
    out = torch.cat(out_chunks, dim=0)        # (B*N, num_classes)
    return out.view(B, N, -1).mean(dim=1)     # (B, num_classes)


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
    n_clips = max(1, int(cfg.dataset.get("n_clips", 1)))
    n_views = (10 if use_tta else 1) * n_clips
    if use_tta or n_clips > 1:
        print(f"TTA enabled: {n_views} views/video "
              f"({n_clips} temporal clip(s) × {10 if use_tta else 1} spatial crop(s)).")

    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
        tta=use_tta,
        n_clips=n_clips,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    use_amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    amp_dtype_str = str(cfg.training.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    eval_view_chunk = max(1, int(cfg.training.get("eval_view_chunk", 4)))
    if n_views > 1:
        print(f"View chunk size: {eval_view_chunk} (forwards per video: {n_views // eval_view_chunk + (n_views % eval_view_chunk > 0)})")

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for video_batch, labels in val_loader:
            labels = labels.to(device)
            video_batch = video_batch.to(device)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if n_views > 1:
                    logits = _multi_view_logits(model, video_batch, eval_view_chunk)
                else:
                    logits = model(video_batch)

            correct_top1 += int((logits.argmax(1) == labels).sum().item())

            _, top5 = logits.topk(5, dim=1)
            correct_top5 += int(top5.eq(labels.view(-1, 1)).any(dim=1).sum().item())

            total += labels.size(0)

    print(f"Validation samples: {total}")
    print(f"Top-1 accuracy: {correct_top1 / max(total, 1):.4f}")
    print(f"Top-5 accuracy: {correct_top5 / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
