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

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.tsm_resnet import TSMResNet
from utils import (
    build_transforms,
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
    raise ValueError(f"Unknown model.name: {name}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_alpha: float = 0.0,
    cutmix_prob: float = 0.0,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ema_model: AveragedModel | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # MixUp / CutMix — chosen randomly per batch
        use_cutmix = cutmix_prob > 0 and torch.rand(1).item() < cutmix_prob
        use_mixup  = mixup_alpha > 0 and not use_cutmix

        if use_cutmix:
            video_batch, y_a, y_b, lam = cutmix_data(video_batch, labels)
        elif use_mixup:
            video_batch, y_a, y_b, lam = mixup_data(video_batch, labels, mixup_alpha)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(video_batch)
            if use_cutmix or use_mixup:
                loss = mixed_loss(loss_fn, logits, y_a, y_b, lam)
            else:
                loss = loss_fn(logits, labels)

        if use_cutmix or use_mixup:
            correct += int((logits.argmax(1) == y_a).sum().item())
        else:
            correct += int((logits.argmax(1) == labels).sum().item())

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema_model is not None:
            ema_model.update_parameters(model)

        running_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(1) == labels).sum().item())
        total += labels.size(0)

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
    label_aware_flip = bool(cfg.training.get("label_aware_flip", True))
    flip_pairs = discover_flip_pairs(class_names) if label_aware_flip else {}
    if flip_pairs:
        readable = {class_names[i]: class_names[j] for i, j in flip_pairs.items()}
        print(f"Label-aware hflip enabled — {len(flip_pairs)//2} mirror pair(s):")
        for src, dst in readable.items():
            print(f"  {src}  <-flip->  {dst}")
    train_transform = build_transforms(
        is_training=True,  use_imagenet_norm=use_imagenet_norm, flip_pairs=flip_pairs,
    )
    eval_transform  = build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm)

    num_frames = int(cfg.dataset.num_frames)

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=train_transform,
        sample_list=train_samples,
        temporal_jitter=True,   # TSN-style segment sampling during training
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
        temporal_jitter=False,  # deterministic for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg, class_names=class_names).to(device)
    label_smoothing = float(cfg.training.get("label_smoothing", 0.1))
    loss_fn  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    base_lr = float(cfg.training.lr)
    base_max_lr = float(cfg.training.get("max_lr", 1e-3))
    weight_decay = float(cfg.training.get("weight_decay", 0.05))
    use_llrd = (
        cfg.model.name == "videomae"
        and bool(cfg.training.get("layerwise_lr_decay", True))
    )
    if use_llrd:
        from models.videomae import build_videomae_param_groups
        decay_rate = float(cfg.training.get("llrd_decay", 0.75))
        opt_groups, max_lrs = build_videomae_param_groups(
            model,
            base_lr=base_lr,
            base_max_lr=base_max_lr,
            weight_decay=weight_decay,
            decay_rate=decay_rate,
        )
        optimizer = torch.optim.AdamW(opt_groups)
        scheduler_max_lr = max_lrs
        n_layers = len({g["_layer_id"] for g in opt_groups})
        print(
            f"LLRD enabled (decay={decay_rate}, {n_layers} layer buckets, "
            f"top-LR {max(max_lrs):.2e}, bottom-LR {min(max_lrs):.2e})."
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay,
        )
        scheduler_max_lr = base_max_lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=scheduler_max_lr,
        steps_per_epoch=len(train_loader),
        epochs=int(cfg.training.epochs),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    use_amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    amp_dtype_str = str(cfg.training.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    if use_amp:
        print(f"AMP enabled (dtype={amp_dtype_str}).")

    ema_decay = float(cfg.training.get("ema_decay", 0.999))
    use_ema = ema_decay > 0.0
    ema_model: AveragedModel | None = None
    if use_ema:
        ema_model = AveragedModel(model, avg_fn=_ema_avg_fn(ema_decay))
        print(f"EMA enabled (decay={ema_decay}).")

    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    cutmix_prob = float(cfg.training.get("cutmix_prob", 0.0))

    top_k = max(1, int(cfg.training.get("top_k_checkpoints", 3)))
    best_val_accuracy = 0.0
    # Sorted ascending by val_acc; lowest at index 0 so we pop it first when full.
    snapshots: List[Tuple[float, Path]] = []
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume from .resume.pt if present ─────────────────────────────────
    # Reboot-resilient training: every epoch we persist optimizer / scheduler /
    # EMA / snapshot bookkeeping next to the checkpoint as <base>.resume.pt.
    # On startup we look for it and pick up where we left off. The file is
    # deleted on clean completion. Toggle off with training.resume=false.
    resume_path = checkpoint_path.with_suffix(".resume.pt")
    start_epoch = 0
    if bool(cfg.training.get("resume", True)) and resume_path.exists():
        try:
            state = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            if ema_model is not None and state.get("ema_state_dict") is not None:
                ema_model.load_state_dict(state["ema_state_dict"])
            start_epoch = int(state["epoch"])
            best_val_accuracy = float(state.get("best_val_accuracy", 0.0))
            snapshots = [(float(a), Path(p)) for a, p in state.get("snapshots", [])]
            # Validate snapshot paths still exist on disk
            snapshots = [(a, p) for a, p in snapshots if p.exists()]
            print(
                f"[resume] Picked up at epoch {start_epoch}/{cfg.training.epochs} "
                f"(best val so far {best_val_accuracy:.4f}, "
                f"{len(snapshots)} snapshots on disk)"
            )
        except Exception as e:
            print(f"[resume] WARNING: failed to load {resume_path} ({e}). "
                  f"Starting from scratch.")
            start_epoch = 0
            best_val_accuracy = 0.0
            snapshots = []

    def _save_durable(obj: Any, path: Path) -> None:
        """torch.save followed by fsync, so the bytes are committed to the
        (NFS) server before we return. Without the fsync, a subsequent
        os.replace/rename — which is a metadata op and reaches the server
        synchronously — can land *before* the still-buffered data, leaving a
        0-byte or truncated file if the process/host dies in that window.
        This is exactly the failure that zeroed resume.pt + top1.pt."""
        with open(path, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())

    def _fsync_dir(path: Path) -> None:
        """Commit a directory entry (a rename/unlink) to the NFS server.
        Best-effort: never let a dir-fsync hiccup take down training."""
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def _snapshot_path(rank: int) -> Path:
        # rank=1 -> "<base>_top1.pt", etc.
        return checkpoint_path.with_name(f"{checkpoint_path.stem}_top{rank}{checkpoint_path.suffix}")

    def _save_payload(path: Path, state_dict: Dict[str, torch.Tensor], val_acc: float, ema: bool) -> None:
        payload: Dict[str, Any] = {
            "model_state_dict": state_dict,
            "model_name": cfg.model.name,
            "num_classes": int(cfg.model.num_classes),
            "pretrained": bool(cfg.model.pretrained),
            "num_frames": num_frames,
            "val_accuracy": val_acc,
            "ema": ema,
            "config": OmegaConf.to_container(cfg, resolve=True),
            # Persist class folder names + label-aware flip pairs so inference
            # can rebuild the TTA logit permutation without re-scanning train_dir.
            "class_names": list(class_names),
            "flip_pairs": dict(flip_pairs),
        }
        if cfg.model.name == "cnn_lstm":
            payload["lstm_hidden_size"] = int(cfg.model.get("lstm_hidden_size", 512))
        _save_durable(payload, path)

    def _persist_resume_state(epoch_done: int) -> None:
        """Write the full optimizer/scheduler/EMA/snapshot state atomically."""
        state = {
            "epoch": int(epoch_done),
            "best_val_accuracy": float(best_val_accuracy),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "snapshots": [(float(a), str(p)) for a, p in snapshots],
        }
        # NB: don't use .with_suffix here — resume_path already ends in
        # ".resume.pt", and with_suffix would strip ".pt" and produce a double
        # ".resume.resume.pt.tmp" name. Just append ".tmp".
        tmp = resume_path.with_name(resume_path.name + ".tmp")
        _save_durable(state, tmp)         # data committed to NFS before rename
        tmp.replace(resume_path)          # atomic rename over the old resume file
        _fsync_dir(resume_path.parent)    # make the rename itself durable

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            mixup_alpha=mixup_alpha, cutmix_prob=cutmix_prob,
            scheduler=scheduler, ema_model=ema_model,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )

        msg = (
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if ema_model is not None:
            ema_loss, ema_acc = evaluate_epoch(
                ema_model, val_loader, loss_fn, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            msg += f" | ema val loss {ema_loss:.4f} acc {ema_acc:.4f}"
        else:
            ema_acc = -1.0

        print(msg)

        # Pick the better of (raw, EMA) for snapshot ranking
        candidate_acc = max(val_acc, ema_acc)
        use_ema_state = ema_model is not None and ema_acc >= val_acc
        candidate_state = (ema_model.module.state_dict() if use_ema_state else model.state_dict())

        if len(snapshots) < top_k or candidate_acc > snapshots[0][0]:
            # Reserve a temporary path; we re-rank and rename below.
            tmp_path = checkpoint_path.with_name(f".pending_epoch{epoch+1}{checkpoint_path.suffix}")
            _save_payload(tmp_path, {k: v.detach().cpu().clone() for k, v in candidate_state.items()},
                          candidate_acc, ema=use_ema_state)
            snapshots.append((candidate_acc, tmp_path))
            snapshots.sort(key=lambda t: t[0])  # ascending
            if len(snapshots) > top_k:
                _, drop_path = snapshots.pop(0)
                if drop_path.exists():
                    drop_path.unlink()
            # Two-phase rename so the source of one rename can't be the file
            # another iteration is about to unlink. Phase 1: every snapshot to
            # a unique staging path. Phase 2: each staging file to its final
            # _topN.pt (now safe to unlink any pre-existing target).
            staged: List[Tuple[float, Path]] = []
            for i, (acc, path) in enumerate(snapshots):
                staged_path = checkpoint_path.with_name(
                    f".staging_{i}{checkpoint_path.suffix}"
                )
                if path != staged_path:
                    if staged_path.exists():
                        staged_path.unlink()
                    path.rename(staged_path)
                staged.append((acc, staged_path))
            new_snapshots: List[Tuple[float, Path]] = []
            for rank_i, (acc, path) in enumerate(
                sorted(staged, key=lambda t: -t[0]), start=1
            ):
                target = _snapshot_path(rank_i)
                if target.exists():
                    target.unlink()
                path.rename(target)
                new_snapshots.append((acc, target))
            _fsync_dir(checkpoint_path.parent)  # persist the _topN.pt renames
            snapshots = sorted(new_snapshots, key=lambda t: t[0])  # ascending
            best_val_accuracy = max(best_val_accuracy, candidate_acc)
            print(f"  Snapshots updated. Best so far: {best_val_accuracy:.4f}")

        # End-of-epoch: persist resume state so a reboot mid-training can
        # pick up at the next epoch boundary with optimizer/scheduler/EMA intact.
        _persist_resume_state(epoch + 1)

    # Mirror the top-1 snapshot to the user-facing checkpoint_path for backwards compat.
    if snapshots:
        top1_path = _snapshot_path(1)
        if top1_path.exists():
            payload = torch.load(top1_path, map_location="cpu")
            _save_durable(payload, checkpoint_path)
            _fsync_dir(checkpoint_path.parent)
            print(f"Wrote top-1 snapshot to {checkpoint_path}")

    # Clean completion: delete the resume file so the next call to train.py
    # with the same checkpoint_path starts fresh (not from this run's final
    # epoch). Skip removal if training was interrupted (start_epoch >= epochs).
    if resume_path.exists():
        resume_path.unlink()
        print(f"[resume] Cleaned up {resume_path.name}")

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")
    if len(snapshots) > 1:
        ranked = sorted(snapshots, key=lambda t: -t[0])
        print("Top-K snapshots saved:")
        for rank_i, (acc, path) in enumerate(ranked, start=1):
            print(f"  top{rank_i} (val acc={acc:.4f}): {path}")


if __name__ == "__main__":
    main()
