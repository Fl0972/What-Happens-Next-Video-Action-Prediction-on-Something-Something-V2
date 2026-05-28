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
from typing import Any, Dict, Optional, Tuple

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
from models.tsm_gru import TSMBiGRU
from models.tsm_resnet import TSMResNet
from models.video_former_lite import VideoFormerLite
try:
    from models.coatnet import CoAtNetVideo
except ImportError:
    CoAtNetVideo = None  # type: ignore[assignment,misc]
try:
    from models.efficientformer import EfficientFormerVideo
except ImportError:
    EfficientFormerVideo = None  # type: ignore[assignment,misc]
from models.maxvit import MaxViTVideo
from models.vit_rnn import ViTBiGRU, ViTBiGRUAttn, ViTPerceiver
from models.vit_spacetime import ViTSpaceTimePerceiver
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
    if name == "tsm_gru":
        return TSMBiGRU(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            backbone=str(cfg.model.get("backbone", "resnet50")),
            fold_div=int(cfg.model.get("fold_div", 8)),
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.2)),
            gru_hidden=int(cfg.model.get("gru_hidden", 256)),
            gru_layers=int(cfg.model.get("gru_layers", 1)),
            feat_dropout=float(cfg.model.get("feat_dropout", 0.3)),
            head_dropout=float(cfg.model.get("head_dropout", 0.5)),
        )
    if name == "vit_bigru":
        return ViTBiGRU(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            gru_hidden=int(cfg.model.get("gru_hidden", 256)),
            gru_layers=int(cfg.model.get("gru_layers", 1)),
            feat_dropout=float(cfg.model.get("feat_dropout", 0.0)),
            head_dropout=float(cfg.model.get("head_dropout", 0.3)),
        )
    if name == "vit_bigru_attn":
        return ViTBiGRUAttn(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            gru_hidden=int(cfg.model.get("gru_hidden", 256)),
            gru_layers=int(cfg.model.get("gru_layers", 1)),
            feat_dropout=float(cfg.model.get("feat_dropout", 0.0)),
            head_dropout=float(cfg.model.get("head_dropout", 0.3)),
        )
    if name == "vit_perceiver":
        return ViTPerceiver(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            num_queries=int(cfg.model.get("num_queries", 16)),
            perceiver_heads=int(cfg.model.get("perceiver_heads", 8)),
            perceiver_layers=int(cfg.model.get("perceiver_layers", 2)),
            feat_dropout=float(cfg.model.get("feat_dropout", 0.0)),
            head_dropout=float(cfg.model.get("head_dropout", 0.3)),
        )
    if name == "efficientformer":
        if EfficientFormerVideo is None:
            raise RuntimeError("timm is required for efficientformer: pip install timm")
        return EfficientFormerVideo(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            variant=str(cfg.model.get("variant", "efficientformer_l3.snap_dist_in1k")),
            head_dropout=float(cfg.model.get("head_dropout", 0.3)),
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.0)),
        )
    if name == "coatnet":
        if CoAtNetVideo is None:
            raise RuntimeError("timm is required for coatnet: pip install timm")
        return CoAtNetVideo(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            variant=str(cfg.model.get("variant", "coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k")),
            head_dropout=float(cfg.model.get("head_dropout", 0.3)),
        )
    if name == "maxvit":
        return MaxViTVideo(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            head_dropout=float(cfg.model.get("head_dropout", 0.3)),
        )
    if name == "vit_spacetime":
        mae_path = cfg.model.get("mae_pretrained_path", None)
        return ViTSpaceTimePerceiver(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            img_size=int(cfg.dataset.get("image_size", 224)),
            patch_size=int(cfg.model.get("patch_size", 16)),
            embed_dim=int(cfg.model.get("embed_dim", 768)),
            depth=int(cfg.model.get("depth", 12)),
            num_heads=int(cfg.model.get("num_heads", 12)),
            mlp_ratio=float(cfg.model.get("mlp_ratio", 4.0)),
            spacetime_depth=int(cfg.model.get("spacetime_depth", 6)),
            num_queries=int(cfg.model.get("num_queries", 16)),
            perceiver_heads=int(cfg.model.get("perceiver_heads", 8)),
            perceiver_layers=int(cfg.model.get("perceiver_layers", 2)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            attn_dropout=float(cfg.model.get("attn_dropout", 0.0)),
            head_dropout=float(cfg.model.get("head_dropout", 0.3)),
            mae_pretrained_path=str(mae_path) if mae_path else None,
        )
    raise ValueError(f"Unknown model.name: {name}")


def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Adam (default), AdamW, or SGD+momentum, optionally with weight decay.

    backbone_lr_scale < 1.0 splits parameters into two groups: backbone at
    lr * scale and everything else (RNN head, classifier) at full lr.
    Requires the model to have a .backbone attribute.
    """
    name = str(cfg.training.get("optimizer", "adam")).lower()
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    backbone_lr_scale = float(cfg.training.get("backbone_lr_scale", 1.0))

    llrd_decay = float(cfg.training.get("llrd_decay", 0.0))
    if llrd_decay > 0 and hasattr(model, "param_groups_llrd"):
        param_groups = model.param_groups_llrd(lr, llrd_decay)
        lrs = sorted({g["lr"] for g in param_groups})
        print(f"LLRD (decay={llrd_decay}): {len(param_groups)} groups, lr range [{lrs[0]:.2e}, {lrs[-1]:.2e}]")
    elif backbone_lr_scale < 1.0 and hasattr(model, "backbone"):
        backbone_ids = {id(p) for p in model.backbone.parameters()}
        param_groups = [
            {"params": [p for p in model.backbone.parameters()], "lr": lr * backbone_lr_scale},
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": lr},
        ]
        print(f"Layer-wise LR: backbone={lr * backbone_lr_scale:.2e}, head={lr:.2e}")
    else:
        param_groups = list(model.parameters())

    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg.training.get("momentum", 0.9))
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=momentum > 0,
        )
    return torch.optim.Adam(param_groups, lr=lr)


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
    clip_grad_norm: float = 0.0,
    ema_model: "nn.Module | None" = None,
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

        if ema_model is not None:
            ema_model.update_parameters(model)

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


def _init_wandb(
    cfg: DictConfig,
    run_name: Optional[str] = None,
    group: Optional[str] = None,
    resume_run_id: Optional[str] = None,
):
    """Start a wandb run. Returns the run object, or None if wandb_project is unset."""
    project = cfg.training.get("wandb_project", None)
    if not project:
        return None
    import wandb
    return wandb.init(
        project=str(project),
        name=run_name or cfg.training.get("wandb_run_name", None),
        group=group,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
        id=resume_run_id,
    )


def _save_resume_ckpt(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_accuracy: float,
    num_frames: int,
    cfg: DictConfig,
    wandb_run_id: Optional[str] = None,
    **extra: Any,
) -> None:
    """Overwrite the single last-epoch checkpoint used to resume interrupted runs.

    Stores everything needed to restart: model weights, optimizer moments,
    scheduler state, AMP scaler state, current epoch, and best val accuracy.
    The saved epoch is the *next* epoch to run (epoch + 1).
    Any keyword arguments in ``extra`` are merged into the payload, allowing
    callers to persist additional state (e.g. rolling_window for rotating folds).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    uncompiled = getattr(model, "_orig_mod", model)
    payload = {
        "epoch": epoch + 1,
        "model_state_dict": uncompiled.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "num_frames": num_frames,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "wandb_run_id": wandb_run_id,
        **extra,
    }
    torch.save(payload, path)


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
    interp = bool(cfg.dataset.get("interpolate_frames", False))
    lmdb_path = cfg.dataset.get("lmdb_path")
    lmdb_path = Path(lmdb_path).resolve() if lmdb_path else None
    if lmdb_path:
        print(f"Using LMDB cache: {lmdb_path}")
    if interp:
        print(f"Frame interpolation enabled: {num_frames} real frames → {num_frames * 2} after blending.")

    train_dataset = make_dataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=train_transform,
        sample_list=train_samples,
        lmdb_path=lmdb_path,
        temporal_jitter=True,
        interpolate_frames=interp,
    )
    val_dataset = make_dataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
        lmdb_path=lmdb_path,
        temporal_jitter=False,
        interpolate_frames=interp,
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

    # EMA / SWA flags (models created here; SWALR scheduler created after optimizer below)
    use_ema = bool(cfg.training.get("ema", False))
    ema_model = None
    if use_ema:
        ema_decay = float(cfg.training.get("ema_decay", 0.9998))
        ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
        )
        print(f"EMA enabled (decay={ema_decay}).")

    use_swa = bool(cfg.training.get("swa", False))
    swa_model = None
    swa_lr_val = 0.0
    swa_start = 0
    if use_swa:
        epochs_total = int(cfg.training.epochs)
        swa_epochs = int(cfg.training.get("swa_epochs", max(1, epochs_total // 4)))
        swa_start = epochs_total - swa_epochs
        swa_lr_val = float(cfg.training.get("swa_lr", float(cfg.training.lr) * 0.05))
        swa_model = torch.optim.swa_utils.AveragedModel(model)

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

    swa_scheduler = None
    if use_swa:
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer, swa_lr=swa_lr_val,
            anneal_epochs=max(1, int(cfg.training.get("swa_epochs", 1)) // 2),
        )
        print(f"SWA enabled: starts at epoch {swa_start + 1}, swa_lr={swa_lr_val:.2e}.")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("Mixed precision (AMP) enabled.")

    # --- Resume: load before compile so state_dict has no _orig_mod. prefix ---
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    resume_path = checkpoint_path.with_stem(checkpoint_path.stem + "_last")
    start_epoch = 0
    best_val_accuracy = 0.0
    wandb_run_id: Optional[str] = None

    if bool(cfg.training.get("resume", False)) and resume_path.exists():
        print(f"Resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"])
        best_val_accuracy = float(ckpt["best_val_accuracy"])
        wandb_run_id = ckpt.get("wandb_run_id")
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_accuracy:.4f}")

    # Compile after resume so loaded weights go into the uncompiled module
    compile_enabled = bool(cfg.training.get("compile", False)) and device.type == "cuda"
    if compile_enabled:
        compile_mode = str(cfg.training.get("compile_mode", "default"))
        print(f"torch.compile enabled (mode={compile_mode}).", flush=True)
        model = torch.compile(model, mode=compile_mode)

    wandb_run = _init_wandb(cfg, resume_run_id=wandb_run_id)

    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    cutmix_prob = float(cfg.training.get("cutmix_prob", 0.0))
    clip_grad_norm = float(cfg.training.get("clip_grad_norm", 0.0))
    reg_warmup_epochs = int(cfg.training.get("reg_warmup_epochs", 0))
    if reg_warmup_epochs > 0 and (mixup_alpha > 0 or cutmix_prob > 0):
        print(
            f"Regularization warmup: MixUp/CutMix disabled for the first "
            f"{reg_warmup_epochs} epochs, then enabled at full strength "
            f"(mixup_alpha={mixup_alpha}, cutmix_prob={cutmix_prob})."
        )

    log_path = checkpoint_path.with_suffix(".csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if start_epoch > 0 else "w"
    log_file = open(log_path, log_mode, newline="", buffering=1)
    log_writer = csv.writer(log_file)
    if start_epoch == 0:
        log_writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"])
    print(f"Logging metrics to {log_path}", flush=True)

    epochs = int(cfg.training.epochs)
    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", dynamic_ncols=True):
        in_reg_warmup = epoch < reg_warmup_epochs
        cur_mixup = 0.0 if in_reg_warmup else mixup_alpha
        cur_cutmix = 0.0 if in_reg_warmup else cutmix_prob
        in_swa_phase = use_swa and epoch >= swa_start
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            mixup_alpha=cur_mixup, cutmix_prob=cur_cutmix,
            epoch_desc=f"Train {epoch + 1}/{epochs}",
            scaler=scaler if use_amp else None,
            clip_grad_norm=clip_grad_norm,
            ema_model=ema_model,
        )

        # SWA: collect model weights once per epoch during the SWA phase
        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif scheduler is not None:
            scheduler.step()

        # Choose evaluation model: EMA preferred over raw model for val accuracy
        eval_model = ema_model if (ema_model is not None) else model
        val_loss, val_acc = evaluate_epoch(
            eval_model, val_loader, loss_fn, device,
            epoch_desc=f"Val {epoch + 1}/{epochs}",
            amp_enabled=use_amp,
        )

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

        if wandb_run is not None:
            wandb_run.log({
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "lr": current_lr, "best_val_acc": best_val_accuracy,
            }, step=epoch + 1)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
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

        _save_resume_ckpt(
            resume_path, model, optimizer, scheduler, scaler,
            epoch, best_val_accuracy, num_frames, cfg,
            wandb_run_id=wandb_run.id if wandb_run is not None else None,
        )

    # SWA post-training: update batch norm stats and evaluate averaged model
    if use_swa and swa_model is not None:
        print("Updating SWA batch-norm statistics...", flush=True)
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_loss, swa_acc = evaluate_epoch(
            swa_model, val_loader, loss_fn, device, epoch_desc="SWA Val", amp_enabled=use_amp
        )
        print(f"SWA val acc={swa_acc:.4f} (best so far={best_val_accuracy:.4f})", flush=True)
        if swa_acc > best_val_accuracy:
            best_val_accuracy = swa_acc
            uncompiled_swa = getattr(swa_model, "_orig_mod", swa_model)
            payload = {
                "model_state_dict": uncompiled_swa.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "num_frames": num_frames,
                "val_accuracy": swa_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(payload, checkpoint_path)
            print(f"  SWA checkpoint saved (val acc={swa_acc:.4f})")

    log_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
