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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.video_dataset import collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.trackA_r2plus1d import R2Plus1D
from models.trackA_tsm_gru import TSMBiGRU
from models.tsm_resnet import TSMResNet
from models.trackA_video_former_lite import VideoFormerLite
try:
    from models.trackA_coatnet import CoAtNetVideo
except ImportError:
    CoAtNetVideo = None  # type: ignore[assignment,misc]
try:
    from models.trackA_efficientformer import EfficientFormerVideo
except ImportError:
    EfficientFormerVideo = None  # type: ignore[assignment,misc]
from models.trackA_maxvit import MaxViTVideo
from models.trackA_vit_rnn import ViTBiGRU, ViTBiGRUAttn, ViTPerceiver, ViTSBiGRUAttn
from models.trackA_vit_spacetime import ViTSpaceTimePerceiver
from utils import (
    build_transforms,
    build_weighted_sampler,
    cutmix_data,
    discover_flip_pairs,
    mixed_loss,
    mixup_data,
    set_seed,
    split_train_val,
)


def _ema_avg_fn(decay: float):
    def avg(avg_param: torch.Tensor, model_param: torch.Tensor, num_averaged: int) -> torch.Tensor:
        return decay * avg_param + (1.0 - decay) * model_param
    return avg


def build_model(cfg: DictConfig, class_names: Optional[List[Optional[str]]] = None) -> nn.Module:
    """Create the model described by cfg.model.name.

    ``class_names`` (index-ordered list of class folder names) is only used by
    the ``videomae`` model for head warm-start; pass ``None`` at inference time.
    """
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
    if name == "vit_s_bigru_attn":
        return ViTSBiGRUAttn(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            gru_hidden=int(cfg.model.get("gru_hidden", 192)),
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
    if name == "videomae":
        from models.videomae import VideoMAE
        warm_start = bool(cfg.model.get("warm_start_head_from_ssv2", True))
        return VideoMAE(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            checkpoint=str(cfg.model.get("checkpoint", VideoMAE.DEFAULT_CHECKPOINT)),
            gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", False)),
            class_names=class_names if warm_start else None,
        )
    if name == "vjepa":
        from models.vjepa import VJEPA2
        warm_start = bool(cfg.model.get("warm_start_head_from_ssv2", True))
        return VJEPA2(
            num_classes=num_classes,
            num_frames=int(cfg.model.num_frames),
            pretrained=pretrained,
            checkpoint=str(cfg.model.get("checkpoint", VJEPA2.DEFAULT_CHECKPOINT)),
            unfreeze_top_k=int(cfg.model.get("unfreeze_top_k", 0)),
            gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", False)),
            class_names=class_names if warm_start else None,
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

        logits = model(video_batch)

        if use_cutmix or use_mixup:
            loss = mixed_loss(loss_fn, logits, y_a, y_b, lam)
            # Accuracy tracked against the dominant label
            correct += int((logits.argmax(1) == y_a).sum().item())
        else:
            loss = loss_fn(logits, labels)
            correct += int((logits.argmax(1) == labels).sum().item())

        loss.backward()
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
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)
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

    # Index-ordered class folder names (e.g. "012_Pouring_something_into_something"),
    # used for VideoMAE head warm-start. Missing indices stay None.
    _name_by_idx: Dict[int, str] = {}
    for _video_dir, _idx in all_samples:
        _name_by_idx.setdefault(int(_idx), _video_dir.parent.name)
    class_names: List[Optional[str]] = [
        _name_by_idx.get(i) for i in range(int(cfg.model.num_classes))
    ]

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples, val_ratio=float(cfg.dataset.val_ratio), seed=int(cfg.dataset.seed)
    )

    # Optional: merge pseudo-labeled test samples into the *train* pool only
    # (val_samples stays clean so in-training val_acc remains an honest signal).
    pseudo_path = cfg.dataset.get("pseudo_labels_path")
    if pseudo_path:
        import csv as _csv
        threshold = float(cfg.dataset.get("pseudo_threshold", 0.85))
        kept: List[Tuple[Path, int]] = []
        total = 0
        with open(str(pseudo_path), newline="", encoding="utf-8") as _f:
            for row in _csv.DictReader(_f):
                total += 1
                if float(row["confidence"]) >= threshold:
                    kept.append((Path(row["video_path"]), int(row["pseudo_label"])))
        print(
            f"Pseudo-labels: kept {len(kept)}/{total} test videos at "
            f"confidence ≥ {threshold} (from {pseudo_path})."
        )
        train_samples = list(train_samples) + kept

    use_imagenet_norm = bool(cfg.model.pretrained)
    train_transform = build_transforms(is_training=True,  use_imagenet_norm=use_imagenet_norm)
    eval_transform  = build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm)

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
    loss_fn  = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr))

    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    cutmix_prob = float(cfg.training.get("cutmix_prob", 0.0))

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            mixup_alpha=mixup_alpha, cutmix_prob=cutmix_prob,
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
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
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
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

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
