"""
VideoFrameDataset: loads a fixed number of RGB frames per video folder.

Expected layout under root_dir::

    root_dir/
      000_SomeClassName/
        video_12345/
          frame_000.jpg
          frame_001.jpg
          ...
      001_AnotherClass/
        ...

Class index is parsed from the leading number in the class folder name (000, 001, ...).
Each __getitem__ returns:
    video_tensor: float tensor of shape (T, C, H, W)  or (N, T, C, H, W) when tta=True
    label: int64 scalar class index
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


def _list_frame_paths(video_dir: Path) -> List[Path]:
    """All image files in a video folder, sorted by name."""
    paths: List[Path] = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(sorted(video_dir.glob(extension)))
    return sorted(paths, key=lambda p: p.name)


def _parse_class_index(class_dir_name: str) -> Optional[int]:
    match = re.match(r"^(\d+)_", class_dir_name)
    return int(match.group(1)) if match else None


def collect_video_samples(root_dir: Path) -> List[Tuple[Path, int]]:
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    samples: List[Tuple[Path, int]] = []
    class_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]
    fallback_index = {p.name: i for i, p in enumerate(class_dirs)}

    for class_dir in class_dirs:
        parsed = _parse_class_index(class_dir.name)
        class_index = parsed if parsed is not None else fallback_index[class_dir.name]

        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            if _list_frame_paths(video_dir):
                samples.append((video_dir, class_index))

    if not samples:
        raise RuntimeError(f"No video folders with frames under {root_dir}")
    return samples


def _pick_frame_indices(num_available: int, num_frames: int) -> List[int]:
    """Uniform linspace sampling (deterministic — used for val/test)."""
    if num_available <= 0:
        raise ValueError("Video has no frames.")
    if num_available == 1:
        return [0] * num_frames
    positions = torch.linspace(0, num_available - 1, steps=num_frames)
    return [int(round(float(x))) for x in positions]


def _pick_frame_indices_tsn(num_available: int, num_frames: int) -> List[int]:
    """
    TSN-style temporal jitter: divide video into T equal segments, sample
    one frame uniformly at random from each segment. Each epoch the model
    sees a different temporal slice of every video.
    """
    seg = num_available / num_frames
    indices = []
    for i in range(num_frames):
        start = int(i * seg)
        end   = max(start, int((i + 1) * seg) - 1)
        end   = min(end, num_available - 1)
        indices.append(random.randint(start, end))
    return indices


def _pick_frame_indices_multi(
    num_available: int, num_frames: int, clip_idx: int, n_clips: int
) -> List[int]:
    """
    Multi-clip uniform sampling: produces ``n_clips`` deterministic but
    distinct frame samplings. Within each segment of length ``seg``, the
    k-th clip uses an offset of ``(k+0.5) * seg / n_clips`` — so the clips
    cover non-overlapping sub-positions within each segment.
    """
    if num_available <= 0:
        raise ValueError("Video has no frames.")
    if num_available == 1:
        return [0] * num_frames
    seg = num_available / num_frames
    offset = (clip_idx + 0.5) * seg / max(n_clips, 1)
    indices: List[int] = []
    for i in range(num_frames):
        idx = int(round(i * seg + offset))
        idx = max(0, min(idx, num_available - 1))
        indices.append(idx)
    return indices


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int,
        transform: Callable,
        sample_list: Optional[List[Tuple[Path, int]]] = None,
        temporal_jitter: bool = False,
        tta: bool = False,
        n_clips: int = 1,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.temporal_jitter = temporal_jitter
        self.tta = tta
        self.n_clips = max(1, int(n_clips))
        self.samples = list(sample_list) if sample_list is not None else collect_video_samples(self.root_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_dir, label = self.samples[index]
        frame_paths = _list_frame_paths(video_dir)
        n_avail = len(frame_paths)

        views: List[torch.Tensor] = []
        # Label may get remapped by the transform (label-aware hflip on
        # direction-encoded SSv2 classes). Only the n_clips=1, training-mode
        # path actually mutates it; TTA / multi-clip eval keeps it as-is.
        out_label = int(label)
        for clip_idx in range(self.n_clips):
            if self.temporal_jitter:
                indices = _pick_frame_indices_tsn(n_avail, self.num_frames)
            elif self.n_clips > 1:
                indices = _pick_frame_indices_multi(n_avail, self.num_frames, clip_idx, self.n_clips)
            else:
                indices = _pick_frame_indices(n_avail, self.num_frames)

            pil_frames: List[Image.Image] = []
            for fi in indices:
                with Image.open(frame_paths[fi]) as img:
                    pil_frames.append(img.convert("RGB"))

            if self.tta:
                views.append(self.transform.tta(pil_frames))    # (10, T, C, H, W)
            else:
                result = self.transform(pil_frames, label=out_label)
                if isinstance(result, tuple):
                    view, out_label = result
                else:
                    view = result
                views.append(view)                               # (T, C, H, W)

        if self.n_clips == 1:
            return views[0], torch.tensor(out_label, dtype=torch.long)

        # Multi-clip: stack along view dim, then flatten any spatial-TTA dim
        # into the same view axis so downstream code only sees (N, T, C, H, W).
        stacked = torch.stack(views, dim=0)
        if stacked.dim() == 6:
            # (n_clips, 10, T, C, H, W) -> (n_clips*10, T, C, H, W)
            stacked = stacked.view(stacked.shape[0] * stacked.shape[1], *stacked.shape[2:])
        return stacked, torch.tensor(label, dtype=torch.long)
