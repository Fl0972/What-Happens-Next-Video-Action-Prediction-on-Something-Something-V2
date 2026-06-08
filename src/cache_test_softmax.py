#!/usr/bin/env python3
"""Run per-checkpoint TTA inference on the test set and save the softmax to
the same cache files ensemble_per_class.py uses, **without** the unnecessary
val pass or per-class submission writing.

Lets us cache test softmax for a new model (e.g. V-JEPA-ft) so we can combine
it with existing per-model caches (k400, tsm) in honest_ensemble.py without
recomputing what's already cached.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from create_submission import (
    _resolve_checkpoint_paths,
    build_model_from_checkpoint,
    discover_all_test_videos,
    run_inference_logits,
)
from dataset.video_dataset import VideoFrameDataset, _list_frame_paths
from ensemble_per_class import _cache_key
from utils import build_flip_perm, build_transforms, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))
    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    ckpt_paths = _resolve_checkpoint_paths(cfg)
    missing = [p for p in ckpt_paths if not p.is_file()]
    if missing:
        raise SystemExit(f"Checkpoint(s) not found: {missing}")

    cache_dir_cfg = cfg.training.get("softmax_cache_dir")
    if cache_dir_cfg is None:
        cache_dir_cfg = Path(ckpt_paths[0]).parent / "_softmax_cache"
    cache_dir = Path(str(cache_dir_cfg)).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"cache_dir: {cache_dir}")

    first_ckpt: Dict[str, Any] = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    num_classes = int(first_ckpt.get("num_classes", cfg.model.num_classes))
    num_frames = int(first_ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(first_ckpt.get("pretrained", cfg.model.pretrained))
    image_size = int(first_ckpt.get("image_size", cfg.dataset.get("image_size", 224)))
    eval_transform = build_transforms(image_size=image_size, is_training=False, use_imagenet_norm=pretrained)

    use_tta = bool(cfg.dataset.get("tta", True))
    n_clips = max(1, int(cfg.dataset.get("n_clips", 1)))
    n_views = (10 if use_tta else 1) * n_clips
    multi_view = n_views > 1
    use_amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(cfg.training.get("amp_dtype", "bfloat16")).lower() == "bfloat16" else torch.float16
    eval_view_chunk = max(1, int(cfg.training.get("eval_view_chunk", 4)))

    flip_pairs_dict = {int(k): int(v) for k, v in (first_ckpt.get("flip_pairs") or {}).items()}
    flip_perm = (
        build_flip_perm(num_classes, flip_pairs_dict).to(device)
        if (use_tta and flip_pairs_dict) else None
    )

    test_root = Path(cfg.dataset.test_dir).resolve()
    test_names, test_dirs = discover_all_test_videos(test_root)
    valid_test_dirs = [p for n, p in zip(test_names, test_dirs) if _list_frame_paths(p)]
    if not valid_test_dirs:
        raise SystemExit(f"No test videos with frames under {test_root}")
    test_sample_list = [(p, 0) for p in valid_test_dirs]
    test_dataset = VideoFrameDataset(
        root_dir=test_root, num_frames=num_frames, transform=eval_transform,
        sample_list=test_sample_list, tta=use_tta, n_clips=n_clips,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.num_workers), pin_memory=(device.type == "cuda"),
    )
    print(f"test: {len(test_dataset)} clips | num_frames={num_frames} image_size={image_size} TTA={use_tta}")

    for i, ckpt_path in enumerate(ckpt_paths, start=1):
        key = _cache_key(ckpt_path, test_root, n_clips, use_tta, num_frames)
        cache_file = cache_dir / f"{ckpt_path.stem}_test_{key}.pt"
        if cache_file.exists():
            print(f"[{i}/{len(ckpt_paths)}] cache HIT: {cache_file.name}")
            continue
        ckpt = first_ckpt if i == 1 else torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model_from_checkpoint(ckpt).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[{i}/{len(ckpt_paths)}] inferring {ckpt_path.name} ...", flush=True)
        probs = run_inference_logits(
            model, test_loader, device, total_videos=len(test_dataset),
            multi_view=multi_view, use_amp=use_amp, amp_dtype=amp_dtype,
            eval_view_chunk=eval_view_chunk, flip_perm=flip_perm, spatial_tta=use_tta,
        )
        torch.save(probs, cache_file)
        print(f"  saved {cache_file.name} shape={tuple(probs.shape)}")


if __name__ == "__main__":
    main()
