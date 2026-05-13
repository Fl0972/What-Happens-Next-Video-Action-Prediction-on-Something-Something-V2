"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.video_dataset import collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.r2plus1d import R2Plus1D
from models.tsm_resnet import TSMResNet
from models.video_former_lite import VideoFormerLite
from utils import (
    build_transforms,
    build_weighted_sampler,
    cutmix_data,
    make_dataset,
    mixed_loss,
    mixup_data,
    set_seed,
    split_train_val,
)


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(num_classes=num_classes, pretrained=pretrained, lstm_hidden_size=int(hidden))
    if name == "tsm_resnet":
        return TSMResNet(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            backbone=str(cfg.model.get("backbone", "resnet50")),
            fold_div=int(cfg.model.get("fold_div", 8)),
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.0)),
            dropout=float(cfg.model.get("dropout", 0.0)),
        )
    if name == "r2plus1d":
        return R2Plus1D(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=float(cfg.model.get("dropout", 0.0)),
        )
    if name == "video_former_lite":
        return VideoFormerLite(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            d_model=int(cfg.model.get("d_model", 512)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            n_layers=int(cfg.model.get("n_layers", 2)),
            mlp_ratio=float(cfg.model.get("mlp_ratio", 4.0)),
            attn_dropout=float(cfg.model.get("attn_dropout", 0.0)),
            proj_dropout=float(cfg.model.get("proj_dropout", 0.0)),
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.0)),
            dropout=float(cfg.model.get("dropout", 0.0)),
        )
    raise ValueError(f"Unknown model.name: {name}")


def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Adam (default), AdamW, or SGD+momentum, optionally with weight decay."""
    name = str(cfg.training.get("optimizer", "adam")).lower()
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg.training.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=momentum > 0,
        )
    return torch.optim.Adam(model.parameters(), lr=lr)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    """Cosine annealing with optional linear warmup. Returns None if disabled.

    Supported schedulers:
      * ``cosine`` — linear warmup + plain CosineAnnealingLR.
      * ``cosine_warm_restarts`` — linear warmup + SGDR
        (CosineAnnealingWarmRestarts). Reads ``training.sgdr_t0`` (initial
        cycle length, default = (epochs - warmup) // 4) and
        ``training.sgdr_t_mult`` (cycle multiplier, default 1 = equal cycles).
    """
    name = str(cfg.training.get("scheduler", "none")).lower()
    if name == "none":
        return None
    epochs = int(cfg.training.epochs)
    warmup = int(cfg.training.get("warmup_epochs", 0))

    if name == "cosine":
        if warmup <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup)
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warm, cosine], milestones=[warmup]
        )

    if name == "cosine_warm_restarts":
        post_warmup = max(1, epochs - warmup)
        t_0 = int(cfg.training.get("sgdr_t0", max(1, post_warmup // 4)))
        t_mult = int(cfg.training.get("sgdr_t_mult", 1))
        sgdr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult
        )
        if warmup <= 0:
            return sgdr
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warm, sgdr], milestones=[warmup]
        )

    raise ValueError(f"Unknown training.scheduler: {name}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_alpha: float = 0.0,
    cutmix_prob: float = 0.0,
    epoch_desc: str = "Train",
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = scaler is not None

    pbar = tqdm(data_loader, desc=epoch_desc, leave=False, dynamic_ncols=True)
    for video_batch, labels in pbar:
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # MixUp / CutMix — chosen randomly per batch
        use_cutmix = cutmix_prob > 0 and torch.rand(1).item() < cutmix_prob
        use_mixup  = mixup_alpha > 0 and not use_cutmix

        if use_cutmix:
            video_batch, y_a, y_b, lam = cutmix_data(video_batch, labels)
        elif use_mixup:
            video_batch, y_a, y_b, lam = mixup_data(video_batch, labels, mixup_alpha)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(video_batch)
            if use_cutmix or use_mixup:
                loss = mixed_loss(loss_fn, logits, y_a, y_b, lam)
            else:
                loss = loss_fn(logits, labels)

        if use_cutmix or use_mixup:
            correct += int((logits.detach().argmax(1) == y_a).sum().item())
        else:
            correct += int((logits.detach().argmax(1) == labels).sum().item())

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)

        pbar.set_postfix(loss=f"{running_loss / max(total, 1):.4f}",
                         acc=f"{correct / max(total, 1):.4f}")

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch_desc: str = "Val",
    amp_enabled: bool = False,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(data_loader, desc=epoch_desc, leave=False, dynamic_ncols=True)
    for video_batch, labels in pbar:
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(1) == labels).sum().item())
        total += labels.size(0)

        pbar.set_postfix(loss=f"{running_loss / max(total, 1):.4f}",
                         acc=f"{correct / max(total, 1):.4f}")

    return running_loss / max(total, 1), correct / max(total, 1)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples, val_ratio=float(cfg.dataset.val_ratio), seed=int(cfg.dataset.seed)
    )

    use_imagenet_norm = bool(cfg.model.pretrained)
    image_size = int(cfg.dataset.get("image_size", 224))
    rand_aug_ops = int(cfg.training.get("rand_augment_ops", 0))
    rand_aug_mag = float(cfg.training.get("rand_augment_magnitude", 0.5))
    horizontal_flip = bool(cfg.training.get("horizontal_flip", True))
    train_transform = build_transforms(
        image_size=image_size,
        is_training=True,
        use_imagenet_norm=use_imagenet_norm,
        rand_augment_ops=rand_aug_ops,
        rand_augment_magnitude=rand_aug_mag,
        horizontal_flip=horizontal_flip,
    )
    eval_transform  = build_transforms(image_size=image_size, is_training=False, use_imagenet_norm=use_imagenet_norm)

    num_frames = int(cfg.dataset.num_frames)
    lmdb_path = cfg.dataset.get("lmdb_path")
    lmdb_path = Path(lmdb_path).resolve() if lmdb_path else None
    if lmdb_path:
        print(f"Using LMDB cache: {lmdb_path}")

    train_dataset = make_dataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=train_transform,
        sample_list=train_samples,
        lmdb_path=lmdb_path,
        temporal_jitter=True,
    )
    val_dataset = make_dataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
        lmdb_path=lmdb_path,
        temporal_jitter=False,
    )

    nw = int(cfg.training.num_workers)
    use_weighted_sampler = bool(cfg.training.get("weighted_sampling", False))
    train_sampler = None
    if use_weighted_sampler:
        sampler_power = float(cfg.training.get("sampler_power", 0.5))
        train_sampler = build_weighted_sampler(train_samples, power=sampler_power)
        print(f"Using WeightedRandomSampler (power={sampler_power}).")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
    )

    model = build_model(cfg).to(device)

    compile_enabled = bool(cfg.training.get("compile", False)) and device.type == "cuda"
    if compile_enabled:
        compile_mode = str(cfg.training.get("compile_mode", "default"))
        print(f"torch.compile enabled (mode={compile_mode}).", flush=True)
        model = torch.compile(model, mode=compile_mode)

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    focal_gamma = float(cfg.training.get("focal_gamma", 0.0))
    if focal_gamma > 0:
        class _FocalLoss(nn.Module):
            def __init__(self, gamma: float, smoothing: float) -> None:
                super().__init__()
                self.gamma = gamma
                self._ce = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction="none")
            def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                ce = self._ce(logits, labels)
                return ((1.0 - torch.exp(-ce)) ** self.gamma * ce).mean()
        loss_fn: nn.Module = _FocalLoss(focal_gamma, label_smoothing)
        print(f"Using Focal loss (gamma={focal_gamma}, label_smoothing={label_smoothing}).")
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("Mixed precision (AMP) enabled.")

    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    cutmix_prob = float(cfg.training.get("cutmix_prob", 0.0))
    reg_warmup_epochs = int(cfg.training.get("reg_warmup_epochs", 0))
    if reg_warmup_epochs > 0 and (mixup_alpha > 0 or cutmix_prob > 0):
        print(
            f"Regularization warmup: MixUp/CutMix disabled for the first "
            f"{reg_warmup_epochs} epochs, then enabled at full strength "
            f"(mixup_alpha={mixup_alpha}, cutmix_prob={cutmix_prob})."
        )

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    log_path = checkpoint_path.with_suffix(".csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", newline="", buffering=1)
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"])
    print(f"Logging metrics to {log_path}", flush=True)

    epochs = int(cfg.training.epochs)
    for epoch in tqdm(range(epochs), desc="Epochs", dynamic_ncols=True):
        in_reg_warmup = epoch < reg_warmup_epochs
        cur_mixup = 0.0 if in_reg_warmup else mixup_alpha
        cur_cutmix = 0.0 if in_reg_warmup else cutmix_prob
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            mixup_alpha=cur_mixup, cutmix_prob=cur_cutmix,
            epoch_desc=f"Train {epoch + 1}/{epochs}",
            scaler=scaler if use_amp else None,
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device,
            epoch_desc=f"Val {epoch + 1}/{epochs}",
            amp_enabled=use_amp,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | lr {current_lr:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}",
            flush=True,
        )
        log_writer.writerow([
            epoch + 1, f"{current_lr:.2e}",
            f"{train_loss:.4f}", f"{train_acc:.4f}",
            f"{val_loss:.4f}", f"{val_acc:.4f}",
            f"{best_val_accuracy:.4f}",
        ])

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            # Unwrap torch.compile so state_dict keys do not get the "_orig_mod." prefix
            uncompiled_model = getattr(model, "_orig_mod", model)
            payload: Dict[str, Any] = {
                "model_state_dict": uncompiled_model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "num_frames": num_frames,
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.model.name == "cnn_lstm":
                payload["lstm_hidden_size"] = int(cfg.model.get("lstm_hidden_size", 512))

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint_path)
            print(f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})")

    log_file.close()
    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
