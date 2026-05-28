"""
Stage-1: MAE self-supervised pretraining on individual video frames.

Run from src/:
    python train_mae.py experiment=mae_pretrain

After training, the encoder weights are exported to a separate file:
    {checkpoint_path}_encoder.pt
This file is fed to Stage-2 via model.mae_pretrained_path in the experiment config.

Key differences from train.py:
  * num_frames=1 — each video contributes one randomly sampled frame per epoch
  * No class labels — reconstruction loss only
  * NO horizontal flip (direction-sensitive dataset)
  * AdamW + cosine warmup schedule
  * Saves both the full MAE checkpoint and an encoder-only file for Stage-2
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from dataset.video_dataset import collect_video_samples
from models.vit_spacetime import MAEViT
from utils import build_transforms, make_dataset, set_seed, split_train_val


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _is_main() -> bool:
    return _rank() == 0


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

class _FrameOnlyDataset(torch.utils.data.Dataset):
    """Wraps VideoFrameDataset to return only the frame tensor (drop the label)."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        frame, _ = self.base[idx]          # (1, C, H, W) for num_frames=1
        return frame.squeeze(0)            # (C, H, W)


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def _build_mae_optimizer(model: MAEViT, cfg: DictConfig) -> torch.optim.Optimizer:
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.get("weight_decay", 0.05))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_mae_scheduler(optimizer, cfg: DictConfig):
    epochs = int(cfg.training.epochs)
    warmup = int(cfg.training.get("warmup_epochs", 10))
    if warmup <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup)
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cosine], milestones=[warmup])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: MAEViT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: "torch.amp.GradScaler",
    clip_grad_norm: float,
    desc: str,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    amp_enabled = scaler is not None

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not _is_main())
    for frames in pbar:
        frames = frames.to(device, non_blocking=True)   # (B, C, H, W)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss, _, _ = model(frames)

        if amp_enabled:
            scaler.scale(loss).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        running_loss += float(loss.item()) * frames.shape[0]
        total += frames.shape[0]
        pbar.set_postfix(loss=f"{running_loss / max(total, 1):.4f}")

    return running_loss / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # ---- Distributed setup -----------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if _is_main():
        print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed) + rank)   # different seed per rank for data diversity

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        if _is_main():
            print("CUDA unavailable; using CPU.")
        device_str = "cpu"
    device = torch.device(f"cuda:{local_rank}" if device_str == "cuda" else "cpu")

    # ---- Data -------------------------------------------------------
    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)
    train_samples, _ = split_train_val(
        all_samples, val_ratio=float(cfg.dataset.val_ratio), seed=int(cfg.dataset.seed)
    )

    image_size = int(cfg.dataset.get("image_size", 224))
    # MAE augmentation: RRC + jitter, NO hflip
    transform = build_transforms(
        image_size=image_size,
        is_training=True,
        use_imagenet_norm=False,   # MAE reconstructs raw pixels
        rand_augment_ops=0,
        rand_augment_magnitude=0.0,
        horizontal_flip=False,     # direction-sensitive dataset
    )

    # num_frames=1: one randomly jittered frame per video per epoch
    base_ds = make_dataset(
        root_dir=train_dir,
        num_frames=1,
        transform=transform,
        sample_list=train_samples,
        lmdb_path=None,
        temporal_jitter=True,
        interpolate_frames=False,
    )
    frame_ds = _FrameOnlyDataset(base_ds)

    nw = int(cfg.training.num_workers)
    sampler = DistributedSampler(frame_ds, shuffle=True) if _is_distributed() else None
    loader = DataLoader(
        frame_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
        drop_last=True,
    )

    # ---- Model -------------------------------------------------------
    m = cfg.model
    model = MAEViT(
        img_size=int(cfg.dataset.get("image_size", 224)),
        patch_size=int(m.get("patch_size", 16)),
        embed_dim=int(m.get("embed_dim", 768)),
        depth=int(m.get("depth", 12)),
        num_heads=int(m.get("num_heads", 12)),
        mlp_ratio=float(m.get("mlp_ratio", 4.0)),
        decoder_embed_dim=int(m.get("decoder_embed_dim", 512)),
        decoder_depth=int(m.get("decoder_depth", 4)),
        decoder_num_heads=int(m.get("decoder_num_heads", 16)),
        mask_ratio=float(m.get("mask_ratio", 0.75)),
        dropout=float(m.get("dropout", 0.0)),
        attn_dropout=float(m.get("attn_dropout", 0.0)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if _is_main():
        print(f"MAEViT: {n_params:.1f}M parameters | world_size={world_size}", flush=True)

    if _is_distributed():
        model = DDP(model, device_ids=[local_rank])

    # ---- Optimizer / scheduler ----------------------------------------
    optimizer = _build_mae_optimizer(model, cfg)
    scheduler = _build_mae_scheduler(optimizer, cfg)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    clip_grad_norm = float(cfg.training.get("clip_grad_norm", 1.0))

    # ---- Resume -------------------------------------------------------
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    resume_path = checkpoint_path.with_stem(checkpoint_path.stem + "_last")
    start_epoch = 0

    raw_model = model.module if isinstance(model, DDP) else model

    if bool(cfg.training.get("resume", False)) and resume_path.exists():
        if _is_main():
            print(f"Resuming MAE from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"])
        if _is_main():
            print(f"  Resumed at epoch {start_epoch}", flush=True)

    # ---- Logging (rank 0 only) ------------------------------------------
    log_file = log_writer = None
    if _is_main():
        log_path = checkpoint_path.with_suffix(".csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_mode = "a" if start_epoch > 0 else "w"
        log_file = open(log_path, log_mode, newline="", buffering=1)
        log_writer = csv.writer(log_file)
        if start_epoch == 0:
            log_writer.writerow(["epoch", "lr", "train_loss"])
        print(f"MAE logging to {log_path}", flush=True)

    # ---- Training loop -------------------------------------------------
    epochs = int(cfg.training.epochs)
    best_loss = float("inf")

    for epoch in tqdm(range(start_epoch, epochs), desc="MAE epochs", dynamic_ncols=True,
                      disable=not _is_main()):
        if sampler is not None:
            sampler.set_epoch(epoch)

        loss = _train_one_epoch(
            model, loader, optimizer, device, scaler, clip_grad_norm,
            desc=f"MAE {epoch + 1}/{epochs}",
        )

        if scheduler is not None:
            scheduler.step()

        if _is_main():
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{epochs} | lr {current_lr:.2e} | loss {loss:.4f}",
                flush=True,
            )
            log_writer.writerow([epoch + 1, f"{current_lr:.2e}", f"{loss:.4f}"])

            # Save latest checkpoint (for resumption)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, resume_path)

            # Save best model (by loss)
            if loss < best_loss:
                best_loss = loss
                torch.save({
                    "model_state_dict": raw_model.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "train_loss": loss,
                }, checkpoint_path)
                print(f"  Saved best MAE checkpoint (loss={loss:.4f})", flush=True)

    if _is_main():
        log_file.close()

    # ---- Export encoder weights for Stage-2 (rank 0 only) --------------
    if _is_main():
        encoder_path = checkpoint_path.with_stem(checkpoint_path.stem + "_encoder")
        best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(best_ckpt["model_state_dict"])
        torch.save(
            {"encoder_state_dict": raw_model.encoder_state_dict()},
            encoder_path,
        )
        print(f"Encoder weights exported to {encoder_path}", flush=True)
        print(f"Done. Best MAE loss: {best_loss:.4f}", flush=True)

    if _is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
