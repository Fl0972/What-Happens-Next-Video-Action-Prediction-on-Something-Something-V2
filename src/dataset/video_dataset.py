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

import numpy as np
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


def _interp_frames(frames: List[Image.Image]) -> List[Image.Image]:
    """Double frame count by inserting a linear blend between each adjacent pair.

    4 frames → [f0, blend(f0,f1), f1, blend(f1,f2), f2, blend(f2,f3), f3, f3] = 8
    Works for any even number of input frames; the last real frame is repeated
    as padding so the output length is always 2*len(frames).
    """
    out: List[Image.Image] = []
    for i in range(len(frames) - 1):
        out.append(frames[i])
        a = np.array(frames[i], dtype=np.float32)
        b = np.array(frames[i + 1], dtype=np.float32)
        blend = Image.fromarray(((a + b) * 0.5).clip(0, 255).astype(np.uint8))
        out.append(blend)
    out.append(frames[-1])
    out.append(frames[-1])  # pad last slot so len == 2 * original
    return out


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
        interpolate_frames: bool = False,
        n_clips: int = 1,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.temporal_jitter = temporal_jitter
        self.tta = tta
        self.interpolate_frames = interpolate_frames
        self.n_clips = max(1, int(n_clips))
        self.samples = list(sample_list) if sample_list is not None else collect_video_samples(self.root_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_dir, label = self.samples[index]
        frame_paths = _list_frame_paths(video_dir)
        n_avail = len(frame_paths)

        # Guard: when the video has fewer source frames than we sample, the
        # different "temporal clips" of multi-clip TTA collapse to the same
        # rounded indices — wasted compute. Silently downgrade to 1 clip in
        # that case so the eval/submission path doesn't burn 3× the inference
        # time for no signal.
        effective_n_clips = self.n_clips if n_avail >= self.num_frames else 1

        views: List[torch.Tensor] = []
        # Label may get remapped by the transform (label-aware hflip on
        # direction-encoded SSv2 classes). Only the n_clips=1, training-mode
        # path actually mutates it; TTA / multi-clip eval keeps it as-is.
        out_label = int(label)
        for clip_idx in range(effective_n_clips):
            if self.temporal_jitter:
                indices = _pick_frame_indices_tsn(n_avail, self.num_frames)
            elif effective_n_clips > 1:
                indices = _pick_frame_indices_multi(
                    n_avail, self.num_frames, clip_idx, effective_n_clips,
                )
            else:
                indices = _pick_frame_indices(n_avail, self.num_frames)

            pil_frames: List[Image.Image] = []
            for fi in indices:
                with Image.open(frame_paths[fi]) as img:
                    pil_frames.append(img.convert("RGB"))

            if self.interpolate_frames:
                pil_frames = _interp_frames(pil_frames)

            if self.tta:
                views.append(self.transform.tta(pil_frames))    # (10, T, C, H, W)
            else:
                result = self.transform(pil_frames, label=out_label)
                if isinstance(result, tuple):
                    view, out_label = result
                else:
                    view = result
                views.append(view)                               # (T, C, H, W)

        if effective_n_clips == 1:
            return views[0], torch.tensor(out_label, dtype=torch.long)

        # Multi-clip: stack along view dim, then flatten any spatial-TTA dim
        # into the same view axis so downstream code only sees (N, T, C, H, W).
        stacked = torch.stack(views, dim=0)
        if stacked.dim() == 6:
            # (n_clips, 10, T, C, H, W) -> (n_clips*10, T, C, H, W)
            stacked = stacked.view(stacked.shape[0] * stacked.shape[1], *stacked.shape[2:])
        return stacked, torch.tensor(label, dtype=torch.long)
