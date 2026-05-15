"""
Evaluate one or several saved checkpoints on the full validation split.
Reports top-1 and top-5 accuracy. Supports TTA (dataset.tta=true) and
snapshot-ensemble averaging across K checkpoints.

Example (from src/)::

    # single checkpoint
    python evaluate.py training.checkpoint_path=/abs/path/to/model.pt

    # ensemble the auto-discovered top-K snapshots ("<base>_top1.pt" ... "_topK.pt")
    python evaluate.py training.ensemble_top_k=3

    # explicit list of checkpoints to ensemble
    python evaluate.py 'training.checkpoint_paths=[/.../a.pt, /.../b.pt, /.../c.pt]'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader

from create_submission import _resolve_checkpoint_paths
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_flip_perm, build_transforms, set_seed


def _multi_view_logits(
    model: torch.nn.Module,
    video_batch: torch.Tensor,
    max_views_per_step: int,
    flip_perm: torch.Tensor | None = None,
    spatial_tta: bool = False,
) -> torch.Tensor:
    """video_batch: (B, N, T, C, H, W) -> logits: (B, num_classes).

    Splits the (B*N) views into chunks of ``max_views_per_step`` to avoid OOM
    when N is large (e.g. 30 = 3 temporal clips × 10 spatial crops).

    When ``spatial_tta=True`` and ``flip_perm`` is provided, every odd-indexed
    view (1, 3, 5, ...) within the N axis is treated as horizontally-flipped
    (matching ``VideoTransform.tta()`` which interleaves [orig, flip] per
    spatial offset) and its logits are reindexed by ``flip_perm`` before the
    mean — i.e. the model's prediction for the *flipped* content is mapped
    back to the *original* class.
    """
    B, N, T, C, H, W = video_batch.shape
    flat = video_batch.reshape(B * N, T, C, H, W)
    out_chunks = []
    for s in range(0, B * N, max(1, max_views_per_step)):
        out_chunks.append(model(flat[s : s + max_views_per_step]))
    out = torch.cat(out_chunks, dim=0)            # (B*N, num_classes)
    out = out.view(B, N, -1)                       # (B, N, num_classes)

    if flip_perm is not None and spatial_tta:
        flip_idx = torch.arange(1, N, 2, device=out.device)
        # For flipped views, reindex along the class axis: corrected[c] = raw[perm[c]]
        out[:, flip_idx, :] = out[:, flip_idx, :].index_select(2, flip_perm.to(out.device))

    return out.mean(dim=1)                         # (B, num_classes)


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

    checkpoint_paths: List[Path] = _resolve_checkpoint_paths(cfg)
    missing = [p for p in checkpoint_paths if not p.is_file()]
    if missing:
        raise SystemExit(f"Checkpoint(s) not found: {missing}")
    print(f"Ensembling {len(checkpoint_paths)} checkpoint(s):")
    for p in checkpoint_paths:
        print(f"  - {p}")

    # Load the first checkpoint to derive num_frames / pretrained for the dataset
    first_ckpt: Dict[str, Any] = torch.load(checkpoint_paths[0], map_location="cpu")
    pretrained_used = bool(first_ckpt.get("pretrained", cfg.model.pretrained))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)
    num_frames = int(first_ckpt.get("num_frames", cfg.dataset.num_frames))

    # Use ALL of val_dir — no train/val split here
    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)

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
        print(f"View chunk size: {eval_view_chunk} (forwards per video: "
              f"{n_views // eval_view_chunk + (n_views % eval_view_chunk > 0)})")

    # Build the model once from the first checkpoint's saved cfg, then reload
    # state_dict per checkpoint to ensemble.
    model = load_model_from_checkpoint(first_ckpt, device)

    # Label-aware TTA: if the saved checkpoint declares mirror flip pairs,
    # build the class permutation that remaps flipped TTA views' logits.
    flip_pairs_dict = first_ckpt.get("flip_pairs") or {}
    flip_pairs_dict = {int(k): int(v) for k, v in flip_pairs_dict.items()}
    flip_perm = (
        build_flip_perm(int(first_ckpt.get("num_classes", cfg.model.num_classes)), flip_pairs_dict).to(device)
        if (use_tta and flip_pairs_dict)
        else None
    )
    if flip_perm is not None:
        print(f"Label-aware TTA: remapping {len(flip_pairs_dict)} class indices on flipped views.")

    # Sum softmax probabilities across (views × checkpoints) per video.
    summed_probs: torch.Tensor | None = None
    all_labels: List[torch.Tensor] = []
    for ckpt_idx, ckpt_path in enumerate(checkpoint_paths, start=1):
        if ckpt_idx > 1:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
        print(f"[{ckpt_idx}/{len(checkpoint_paths)}] Inference with {ckpt_path.name}")

        per_ckpt_probs: List[torch.Tensor] = []
        with torch.no_grad():
            for video_batch, labels in val_loader:
                video_batch = video_batch.to(device)
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    if n_views > 1:
                        logits = _multi_view_logits(
                            model, video_batch, eval_view_chunk,
                            flip_perm=flip_perm, spatial_tta=use_tta,
                        )
                    else:
                        logits = model(video_batch)
                per_ckpt_probs.append(logits.float().softmax(dim=1).cpu())
                if ckpt_idx == 1:
                    all_labels.append(labels.cpu())
        probs = torch.cat(per_ckpt_probs, dim=0)
        summed_probs = probs if summed_probs is None else summed_probs + probs

    assert summed_probs is not None
    labels = torch.cat(all_labels, dim=0)
    total = labels.numel()
    correct_top1 = int((summed_probs.argmax(1) == labels).sum().item())
    _, top5 = summed_probs.topk(5, dim=1)
    correct_top5 = int(top5.eq(labels.view(-1, 1)).any(dim=1).sum().item())

    print(f"Validation samples: {total}")
    print(f"Top-1 accuracy: {correct_top1 / max(total, 1):.4f}")
    print(f"Top-5 accuracy: {correct_top5 / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
