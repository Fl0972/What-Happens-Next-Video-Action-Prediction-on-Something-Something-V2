"""
K-fold cross-validation training — two modes.

MODE 1 (default): k separate models
    For every fold i, trains a fresh model on the other (k-1) folds and
    validates on fold i.  Saves one checkpoint per fold.  The k checkpoints
    can be ensembled with create_ensemble_submission.py.

MODE 2 (training.rotating_folds=true): one model, rotating val fold
    Trains a single model for ``epochs`` epochs.  At each epoch the val fold
    is ``epoch % k``, so the model trains on a *different* 80% each epoch and
    validates on a *different* 20%.  No fixed val partition exists — the model
    cannot overfit a single held-out slice.

    Checkpointing uses a **rolling average** of the last k val accuracies
    (one complete rotation) so the signal is always measured over all data.

Usage::

    cd src
    # Mode 1 — train all 5 folds:
    python train_kfold.py experiment=tsm_resnet50_bigru

    # Mode 1 — train a single fold (for parallel runs on separate GPUs):
    python train_kfold.py experiment=tsm_resnet50_bigru training.fold_idx=2

    # Mode 2 — rotating folds, single model:
    python train_kfold.py experiment=tsm_resnet50_bigru training.rotating_folds=true

Mode 1 checkpoints:  {stem}_fold_{i}.pt
Mode 2 checkpoint:   {stem}_rotating.pt
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.video_dataset import collect_video_samples
from train import (
    _init_wandb,
    _save_resume_ckpt,
    build_model,
    build_optimizer,
    build_scheduler,
    evaluate_epoch,
    train_one_epoch,
)
from utils import build_transforms, build_weighted_sampler, make_dataset, set_seed


# ---------------------------------------------------------------------------
# Stratified k-fold split — no sklearn required
# ---------------------------------------------------------------------------

def stratified_kfold(
    samples: List[Tuple[Path, int]],
    n_folds: int,
    seed: int,
) -> List[Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]]:
    """Return [(train_samples, val_samples), ...] for each fold.

    Each class's samples are shuffled (with ``seed``) then divided into
    n_folds consecutive chunks. Fold i uses chunk i as val, the rest as train.
    This preserves class proportions in every fold.
    """
    rng = random.Random(seed)

    by_class: Dict[int, List[Tuple[Path, int]]] = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))

    for label in by_class:
        rng.shuffle(by_class[label])

    # Build per-class chunks
    class_chunks: Dict[int, List[List[Tuple[Path, int]]]] = {}
    for label, class_samples in by_class.items():
        n = len(class_samples)
        chunks = []
        for i in range(n_folds):
            start = (i * n) // n_folds
            end = ((i + 1) * n) // n_folds
            chunks.append(class_samples[start:end])
        class_chunks[label] = chunks

    folds = []
    for fold_idx in range(n_folds):
        val: List[Tuple[Path, int]] = []
        train: List[Tuple[Path, int]] = []
        for label, chunks in class_chunks.items():
            for i, chunk in enumerate(chunks):
                if i == fold_idx:
                    val.extend(chunk)
                else:
                    train.extend(chunk)
        rng.shuffle(train)
        folds.append((train, val))

    return folds


# ---------------------------------------------------------------------------
# Shared loader factory
# ---------------------------------------------------------------------------

def _build_fold_loaders(
    train_samples: List[Tuple[Path, int]],
    val_samples: List[Tuple[Path, int]],
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders for one (train_samples, val_samples) split."""
    train_dir = Path(cfg.dataset.train_dir).resolve()
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
    eval_transform = build_transforms(
        image_size=image_size, is_training=False, use_imagenet_norm=use_imagenet_norm
    )

    num_frames = int(cfg.dataset.num_frames)
    lmdb_path = cfg.dataset.get("lmdb_path")
    lmdb_path = Path(lmdb_path).resolve() if lmdb_path else None

    train_dataset = make_dataset(
        root_dir=train_dir, num_frames=num_frames, transform=train_transform,
        sample_list=train_samples, lmdb_path=lmdb_path, temporal_jitter=True,
    )
    val_dataset = make_dataset(
        root_dir=train_dir, num_frames=num_frames, transform=eval_transform,
        sample_list=val_samples, lmdb_path=lmdb_path, temporal_jitter=False,
    )

    nw = int(cfg.training.num_workers)
    bs = int(cfg.training.batch_size)
    pin = True  # pin_memory set below based on device availability

    use_weighted_sampler = bool(cfg.training.get("weighted_sampling", False))
    sampler = None
    if use_weighted_sampler:
        sampler_power = float(cfg.training.get("sampler_power", 0.5))
        sampler = build_weighted_sampler(train_samples, power=sampler_power)

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=(sampler is None), sampler=sampler,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(
    fold_idx: int,
    train_samples: List[Tuple[Path, int]],
    val_samples: List[Tuple[Path, int]],
    cfg: DictConfig,
    checkpoint_path: Path,
) -> float:
    """Train one fold. Returns the best val accuracy achieved."""
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx}  |  train={len(train_samples)}  val={len(val_samples)}")
    print(f"{'='*60}\n", flush=True)

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    num_frames = int(cfg.dataset.num_frames)
    train_loader, val_loader = _build_fold_loaders(train_samples, val_samples, cfg)

    model = build_model(cfg).to(device)

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
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Resume ---
    resume_path = checkpoint_path.with_stem(checkpoint_path.stem + "_last")
    start_epoch = 0
    best_val_accuracy = 0.0
    wandb_run_id: Optional[str] = None

    if bool(cfg.training.get("resume", False)) and resume_path.exists():
        print(f"  Resuming fold {fold_idx} from {resume_path}", flush=True)
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

    compile_enabled = bool(cfg.training.get("compile", False)) and device.type == "cuda"
    if compile_enabled:
        compile_mode = str(cfg.training.get("compile_mode", "default"))
        model = torch.compile(model, mode=compile_mode)

    base_name = checkpoint_path.stem.replace(f"_fold_{fold_idx}", "")
    wandb_run = _init_wandb(
        cfg,
        run_name=f"fold-{fold_idx}",
        group=f"{base_name}-kfold",
        resume_run_id=wandb_run_id,
    )

    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    cutmix_prob = float(cfg.training.get("cutmix_prob", 0.0))
    clip_grad_norm = float(cfg.training.get("clip_grad_norm", 0.0))
    reg_warmup_epochs = int(cfg.training.get("reg_warmup_epochs", 0))

    log_path = checkpoint_path.with_suffix(".csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if start_epoch > 0 else "w"
    log_file = open(log_path, log_mode, newline="", buffering=1)
    log_writer = csv.writer(log_file)
    if start_epoch == 0:
        log_writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"])

    epochs = int(cfg.training.epochs)
    for epoch in tqdm(range(start_epoch, epochs), desc=f"Fold {fold_idx}", dynamic_ncols=True):
        in_reg_warmup = epoch < reg_warmup_epochs
        cur_mixup = 0.0 if in_reg_warmup else mixup_alpha
        cur_cutmix = 0.0 if in_reg_warmup else cutmix_prob

        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            mixup_alpha=cur_mixup, cutmix_prob=cur_cutmix,
            epoch_desc=f"Fold {fold_idx} | Train {epoch + 1}/{epochs}",
            scaler=scaler if use_amp else None,
            clip_grad_norm=clip_grad_norm,
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device,
            epoch_desc=f"Fold {fold_idx} | Val {epoch + 1}/{epochs}",
            amp_enabled=use_amp,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Fold {fold_idx} | Epoch {epoch + 1}/{epochs} | lr {current_lr:.2e} | "
            f"train {train_acc:.4f} | val {val_acc:.4f}",
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
                "fold_idx": fold_idx,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint_path)
            print(f"  Saved fold {fold_idx} best → {checkpoint_path} (val={val_acc:.4f})")

        _save_resume_ckpt(
            resume_path, model, optimizer, scheduler, scaler,
            epoch, best_val_accuracy, num_frames, cfg,
            wandb_run_id=wandb_run.id if wandb_run is not None else None,
        )

    log_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    return best_val_accuracy


# ---------------------------------------------------------------------------
# Rotating-folds training (single model, fold changes every epoch)
# ---------------------------------------------------------------------------

def train_rotating_folds(
    folds: List[Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]],
    cfg: DictConfig,
    checkpoint_path: Path,
) -> float:
    """Train one model for ``epochs`` epochs, rotating the val fold each epoch.

    At epoch e, fold (e % k) is the val set; the other (k-1) folds form the
    train set.  Because the val partition changes every epoch, no fixed slice
    of the data becomes a 'tuning oracle'.

    Checkpointing uses a **rolling average** of the last k val accuracies
    (one complete rotation) so the saved model is the one that generalised
    best across *all* data, not just the luckiest val fold.

    Returns the best rolling-average val accuracy achieved.
    """
    n_folds = len(folds)

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    print(f"\n{'='*60}")
    print(f"  Rotating folds ({n_folds} folds, 1 model)")
    print(f"{'='*60}\n", flush=True)

    # Pre-build one DataLoader pair per fold so we only scan the filesystem once.
    print("Building DataLoaders for all folds…", flush=True)
    loader_pairs = [_build_fold_loaders(tr, vl, cfg) for tr, vl in folds]

    num_frames = int(cfg.dataset.num_frames)
    model = build_model(cfg).to(device)

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
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Early stopping config ---
    es_patience = int(cfg.training.get("early_stopping_patience", 0))
    es_min_epochs = int(cfg.training.get("early_stopping_min_epochs", 0))
    es_min_delta = float(cfg.training.get("early_stopping_min_delta", 1e-4))
    use_early_stopping = es_patience > 0
    if use_early_stopping:
        print(
            f"Early stopping: patience={es_patience}, min_epochs={es_min_epochs}, "
            f"min_delta={es_min_delta}",
            flush=True,
        )

    # --- Resume ---
    resume_path = checkpoint_path.with_stem(checkpoint_path.stem + "_last")
    start_epoch = 0
    best_rolling_avg = 0.0
    rolling_window: List[float] = []
    no_improve_count = 0
    wandb_run_id: Optional[str] = None

    if bool(cfg.training.get("resume", False)) and resume_path.exists():
        print(f"Resuming rotating-folds from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"])
        best_rolling_avg = float(ckpt["best_val_accuracy"])
        rolling_window = list(ckpt.get("rolling_window", []))
        no_improve_count = int(ckpt.get("no_improve_count", 0))
        wandb_run_id = ckpt.get("wandb_run_id")
        print(f"  Resumed at epoch {start_epoch}, best_rolling_avg={best_rolling_avg:.4f}, no_improve={no_improve_count}")

    compile_enabled = bool(cfg.training.get("compile", False)) and device.type == "cuda"
    if compile_enabled:
        compile_mode = str(cfg.training.get("compile_mode", "default"))
        model = torch.compile(model, mode=compile_mode)

    wandb_run = _init_wandb(cfg, run_name=cfg.training.get("wandb_run_name", "rotating"), resume_run_id=wandb_run_id)

    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    cutmix_prob = float(cfg.training.get("cutmix_prob", 0.0))
    clip_grad_norm = float(cfg.training.get("clip_grad_norm", 0.0))
    reg_warmup_epochs = int(cfg.training.get("reg_warmup_epochs", 0))
    epochs = int(cfg.training.epochs)

    log_path = checkpoint_path.with_suffix(".csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if start_epoch > 0 else "w"
    log_file = open(log_path, log_mode, newline="", buffering=1)
    log_writer = csv.writer(log_file)
    if start_epoch == 0:
        log_writer.writerow([
            "epoch", "val_fold", "lr",
            "train_loss", "train_acc", "val_loss", "val_acc",
            "rolling_avg", "best_rolling_avg",
        ])

    for epoch in tqdm(range(start_epoch, epochs), desc="Rotating folds", dynamic_ncols=True):
        fold_idx = epoch % n_folds
        train_loader, val_loader = loader_pairs[fold_idx]

        in_reg_warmup = epoch < reg_warmup_epochs
        cur_mixup = 0.0 if in_reg_warmup else mixup_alpha
        cur_cutmix = 0.0 if in_reg_warmup else cutmix_prob

        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            mixup_alpha=cur_mixup, cutmix_prob=cur_cutmix,
            epoch_desc=f"Train {epoch + 1}/{epochs} [val_fold={fold_idx}]",
            scaler=scaler if use_amp else None,
            clip_grad_norm=clip_grad_norm,
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device,
            epoch_desc=f"Val   {epoch + 1}/{epochs} [fold={fold_idx}]",
            amp_enabled=use_amp,
        )

        if scheduler is not None:
            scheduler.step()

        rolling_window.append(val_acc)
        if len(rolling_window) > n_folds:
            rolling_window.pop(0)
        rolling_avg = sum(rolling_window) / len(rolling_window)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} [val_fold={fold_idx}] | lr {current_lr:.2e} | "
            f"train {train_acc:.4f} | val {val_acc:.4f} | rolling_avg {rolling_avg:.4f}",
            flush=True,
        )
        log_writer.writerow([
            epoch + 1, fold_idx, f"{current_lr:.2e}",
            f"{train_loss:.4f}", f"{train_acc:.4f}",
            f"{val_loss:.4f}", f"{val_acc:.4f}",
            f"{rolling_avg:.4f}", f"{best_rolling_avg:.4f}",
        ])

        if wandb_run is not None:
            wandb_run.log({
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "rolling_avg": rolling_avg, "val_fold": fold_idx,
                "lr": current_lr, "best_rolling_avg": best_rolling_avg,
            }, step=epoch + 1)

        if len(rolling_window) == n_folds:
            if rolling_avg > best_rolling_avg + es_min_delta:
                best_rolling_avg = rolling_avg
                no_improve_count = 0
                uncompiled_model = getattr(model, "_orig_mod", model)
                payload: Dict[str, Any] = {
                    "model_state_dict": uncompiled_model.state_dict(),
                    "model_name": cfg.model.name,
                    "num_classes": int(cfg.model.num_classes),
                    "pretrained": bool(cfg.model.pretrained),
                    "num_frames": num_frames,
                    "val_accuracy": rolling_avg,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_path)
                print(f"  Saved best → {checkpoint_path} (rolling_avg={rolling_avg:.4f})")
            else:
                no_improve_count += 1

        _save_resume_ckpt(
            resume_path, model, optimizer, scheduler, scaler,
            epoch, best_rolling_avg, num_frames, cfg,
            wandb_run_id=wandb_run.id if wandb_run is not None else None,
            rolling_window=list(rolling_window),
            no_improve_count=no_improve_count,
        )

        if (
            use_early_stopping
            and epoch + 1 >= es_min_epochs
            and len(rolling_window) == n_folds
            and no_improve_count >= es_patience
        ):
            print(
                f"Early stopping triggered at epoch {epoch + 1}: "
                f"no improvement for {no_improve_count} epochs "
                f"(best_rolling_avg={best_rolling_avg:.4f})",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({"early_stopped_epoch": epoch + 1}, step=epoch + 1)
            break

    log_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    return best_rolling_avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    n_folds = int(cfg.training.get("n_folds", 5))
    rotating = bool(cfg.training.get("rotating_folds", False))

    folds = stratified_kfold(all_samples, n_folds=n_folds, seed=int(cfg.dataset.seed))
    print(f"Stratified {n_folds}-fold split over {len(all_samples)} samples.")
    for i, (tr, vl) in enumerate(folds):
        print(f"  Fold {i}: train={len(tr)}  val={len(vl)}")

    base_path = Path(cfg.training.checkpoint_path).resolve()

    if rotating:
        # Mode 2: one model, val fold rotates every epoch
        set_seed(int(cfg.dataset.seed))
        checkpoint = base_path.with_stem(f"{base_path.stem}_rotating")
        best_acc = train_rotating_folds(folds, cfg, checkpoint)
        print(f"\nDone. Best rolling-avg val accuracy: {best_acc:.4f}")
        print(f"Checkpoint: {checkpoint}")
    else:
        # Mode 1: k independent models, one per fold
        fold_idx: Optional[int] = cfg.training.get("fold_idx", None)
        if fold_idx is not None:
            fold_idx = int(fold_idx)
        target_folds = [fold_idx] if fold_idx is not None else list(range(n_folds))

        fold_results: List[Tuple[int, float]] = []
        for i in target_folds:
            set_seed(int(cfg.dataset.seed))
            fold_checkpoint = base_path.with_stem(f"{base_path.stem}_fold_{i}")
            train_samples, val_samples = folds[i]
            best = train_fold(i, train_samples, val_samples, cfg, fold_checkpoint)
            fold_results.append((i, best))

        if len(fold_results) > 1:
            accs = [acc for _, acc in fold_results]
            mean_acc = sum(accs) / len(accs)
            std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
            print("\n" + "=" * 60)
            print("K-FOLD SUMMARY")
            for i, acc in fold_results:
                print(f"  Fold {i}: {acc:.4f}")
            print(f"  Mean: {mean_acc:.4f} ± {std_acc:.4f}")
            print("=" * 60)
            print(
                f"\nEnsemble submission command:\n"
                f"  python create_ensemble_submission.py \\\n"
                f"    \"+checkpoints=[{','.join(str(base_path.with_stem(f'{base_path.stem}_fold_{i}')) for i in range(n_folds))}]\" \\\n"
                f"    \"dataset.tta=true\""
            )


if __name__ == "__main__":
    main()
