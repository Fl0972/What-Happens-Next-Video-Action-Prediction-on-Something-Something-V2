#!/usr/bin/env python3
"""
Model analysis on the val set for the final ensemble.

Runs each model's val-set inference once, caches the logits to .npy under
../models/val_logits/<name>.npy, then computes (offline):

  * Per-model top-1 accuracy
  * Ensemble top-1 accuracy
  * Ablation study (leave-one-out from the ensemble)
  * Confusion matrix + per-class precision / recall / F1
  * Top confused class pairs

Usage (from src/):

    python analyze_ensemble.py
    python analyze_ensemble.py dataset.tta=true   # slower; matches submission

Output:
    ../docs/analysis_results.json   (machine-readable metrics)
    ../docs/analysis_summary.txt    (human-readable summary)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from create_submission import build_model_from_checkpoint
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import build_transforms, set_seed


# The 3 models in the final ensemble (weight, name, ckpt_path)
ENSEMBLE = [
    ("tsm_ultra_v2", 0.50, "../models/tsm_ultra_v2.pt"),
    ("tsm_ultra",    0.25, "../models/tsm_ultra.pt"),
    ("video_former_lite_ultra", 0.25, "../models/video_former_lite_ultra.pt"),
]


@torch.no_grad()
def compute_val_logits(ckpt_path: Path, cfg: DictConfig, device: torch.device,
                       use_tta: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Run val-set inference once, return (logits, labels) as numpy arrays."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(ckpt.get("pretrained", False))
    transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=transform,
        sample_list=val_samples,
        tta=use_tta,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    all_logits, all_labels = [], []
    n_batches = len(loader)
    log_every = max(1, n_batches // 5)
    for i, (vb, lb) in enumerate(loader, 1):
        vb = vb.to(device)
        if use_tta:
            B, N, T, C, H, W = vb.shape
            logits = model(vb.view(B * N, T, C, H, W)).view(B, N, -1).mean(dim=1)
        else:
            logits = model(vb)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(lb.numpy())
        if i % log_every == 0 or i == n_batches:
            print(f"    batch {i}/{n_batches}", flush=True)

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def per_class_metrics(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> Dict[int, Dict[str, float]]:
    """Per-class precision / recall / F1 / support."""
    out = {}
    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        support = int((labels == c).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support, "tp": tp}
    return out


def top_confused_pairs(preds: np.ndarray, labels: np.ndarray, n_classes: int,
                       top_n: int = 10) -> List[Tuple[int, int, int]]:
    """Return the top_n (true, predicted, count) off-diagonal confusion-matrix entries."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    mask = np.eye(n_classes, dtype=bool)
    cm_off = cm.copy()
    cm_off[mask] = 0
    flat = cm_off.flatten()
    idx = np.argsort(-flat)[:top_n]
    pairs = []
    for k in idx:
        if flat[k] == 0:
            break
        t, p = divmod(int(k), n_classes)
        pairs.append((t, p, int(flat[k])))
    return pairs


def get_class_names(val_dir: Path) -> List[str]:
    names = []
    for d in sorted(val_dir.iterdir()):
        if d.is_dir():
            names.append(d.name)
    return names


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.dataset.seed))
    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    use_tta = bool(cfg.dataset.get("tta", False))

    cache_dir = (Path(__file__).parent / "../models/val_logits").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tta_tag = "_tta" if use_tta else ""

    # ----------------------- inference (cached) -----------------------------
    logits_by_name: Dict[str, np.ndarray] = {}
    labels: np.ndarray = None
    for name, _w, ckpt_rel in ENSEMBLE:
        ckpt_path = (Path(__file__).parent / ckpt_rel).resolve()
        logits_npy = cache_dir / f"{name}{tta_tag}.npy"
        labels_npy = cache_dir / f"labels.npy"
        if logits_npy.exists() and labels_npy.exists():
            print(f"[cache] {name}{tta_tag}: loading {logits_npy}", flush=True)
            logits_by_name[name] = np.load(logits_npy)
            labels = np.load(labels_npy)
        else:
            print(f"[run]   {name}{tta_tag}: computing val logits ...", flush=True)
            lg, lb = compute_val_logits(ckpt_path, cfg, device, use_tta)
            np.save(logits_npy, lg)
            if labels is None:
                np.save(labels_npy, lb)
                labels = lb
            logits_by_name[name] = lg

    n_classes = int(cfg.num_classes)
    val_dir = Path(cfg.dataset.val_dir).resolve()
    class_names = get_class_names(val_dir)

    # --------------------- per-model accuracy --------------------------------
    per_model = {}
    for name, _w, _p in ENSEMBLE:
        lg = logits_by_name[name]
        per_model[name] = {
            "top1": topk_accuracy(lg, labels, 1),
            "top5": topk_accuracy(lg, labels, 5),
        }
        print(f"  {name}: top1={per_model[name]['top1']:.4f}  top5={per_model[name]['top5']:.4f}")

    # ---------------------- ensemble accuracy --------------------------------
    weights = {name: w for name, w, _ in ENSEMBLE}
    probs = sum(weights[n] * softmax(logits_by_name[n]) for n, _, _ in ENSEMBLE)
    ens_preds = probs.argmax(axis=1)
    ens_top1 = float((ens_preds == labels).mean())
    ens_top5 = topk_accuracy(probs, labels, 5)
    print(f"\nENSEMBLE (full)  top1={ens_top1:.4f}  top5={ens_top5:.4f}")

    # ------------------------- ablation study --------------------------------
    print("\nAblation (leave-one-out):")
    ablation = {}
    for name_drop, _w, _p in ENSEMBLE:
        sub = [(n, w) for n, w, _ in ENSEMBLE if n != name_drop]
        ws = sum(w for _, w in sub)
        sub_probs = sum((w / ws) * softmax(logits_by_name[n]) for n, w in sub)
        acc = float((sub_probs.argmax(axis=1) == labels).mean())
        ablation[f"-{name_drop}"] = acc
        delta = acc - ens_top1
        print(f"  remove {name_drop}: top1={acc:.4f}  (Δ={delta:+.4f})")

    # ------------------ per-class precision/recall/F1 ------------------------
    pcm = per_class_metrics(ens_preds, labels, n_classes)
    macro_p = float(np.mean([pcm[c]["precision"] for c in pcm]))
    macro_r = float(np.mean([pcm[c]["recall"] for c in pcm]))
    macro_f1 = float(np.mean([pcm[c]["f1"] for c in pcm]))
    print(f"\nMacro-avg  precision={macro_p:.4f}  recall={macro_r:.4f}  F1={macro_f1:.4f}")

    classes_sorted_by_f1 = sorted(pcm.items(), key=lambda kv: kv[1]["f1"])
    print("\n5 worst classes by F1 (ensemble):")
    for c, m in classes_sorted_by_f1[:5]:
        cname = class_names[c] if c < len(class_names) else f"class_{c}"
        print(f"  {c:2d} {cname:50s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")
    print("\n5 best classes by F1 (ensemble):")
    for c, m in classes_sorted_by_f1[-5:]:
        cname = class_names[c] if c < len(class_names) else f"class_{c}"
        print(f"  {c:2d} {cname:50s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")

    # ------------------------ top confused pairs -----------------------------
    pairs = top_confused_pairs(ens_preds, labels, n_classes, top_n=10)
    print("\nTop 10 confused (true → predicted) pairs:")
    for t, p, cnt in pairs:
        tn = class_names[t] if t < len(class_names) else f"class_{t}"
        pn = class_names[p] if p < len(class_names) else f"class_{p}"
        print(f"  {cnt:3d}x   {tn}  →  {pn}")

    # ----------------------------- dump --------------------------------------
    out_dir = (Path(__file__).parent / "../docs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "tta": use_tta,
        "n_val": int(labels.shape[0]),
        "n_classes": n_classes,
        "per_model": per_model,
        "ensemble": {"top1": ens_top1, "top5": ens_top5},
        "ablation": ablation,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_class": {
            (class_names[c] if c < len(class_names) else f"class_{c}"): pcm[c]
            for c in pcm
        },
        "confused_pairs": [
            {
                "true":  class_names[t] if t < len(class_names) else f"class_{t}",
                "pred":  class_names[p] if p < len(class_names) else f"class_{p}",
                "count": cnt,
            }
            for t, p, cnt in pairs
        ],
    }
    out_json = out_dir / f"analysis_results{tta_tag}.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
