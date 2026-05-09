"""
LMDBVideoDataset: drop-in replacement for VideoFrameDataset backed by LMDB.

LMDB layout (produced by pack_lmdb.py):
  b"__samples__"            -> pickle of [(rel_path: str, label: int, n_frames: int), ...]
  b"{idx:08d}_{fi:04d}"     -> raw JPEG bytes for sample #idx, frame #fi
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset.video_dataset import _pick_frame_indices, _pick_frame_indices_tsn


def load_lmdb_samples(lmdb_path: str | Path, root_dir: str | Path) -> List[Tuple[Path, int]]:
    """Read sample metadata from LMDB without touching the original file tree."""
    root = Path(root_dir).resolve()
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        meta: List[Tuple[str, int, int]] = pickle.loads(txn.get(b"__samples__"))
    env.close()
    return [(root / rel, label) for rel, label, _ in meta]


class LMDBVideoDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str | Path,
        root_dir: str | Path,
        num_frames: int,
        transform: Callable,
        sample_list: Optional[List[Tuple[Path, int]]] = None,
        temporal_jitter: bool = False,
        tta: bool = False,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.root_dir = Path(root_dir).resolve()
        self.num_frames = num_frames
        self.transform = transform
        self.temporal_jitter = temporal_jitter
        self.tta = tta
        self._env = None

        tmp = lmdb.open(self.lmdb_path, readonly=True, lock=False, subdir=True)
        with tmp.begin() as txn:
            all_meta: List[Tuple[str, int, int]] = pickle.loads(txn.get(b"__samples__"))
        tmp.close()

        if sample_list is None:
            self.items: List[Tuple[int, int, int]] = [
                (i, label, n_frames) for i, (_, label, n_frames) in enumerate(all_meta)
            ]
        else:
            rel_to_meta = {rel: (i, nf) for i, (rel, _, nf) in enumerate(all_meta)}
            self.items = []
            for video_dir, label in sample_list:
                rel = str(Path(video_dir).resolve().relative_to(self.root_dir))
                lmdb_idx, n_frames = rel_to_meta[rel]
                self.items.append((lmdb_idx, label, n_frames))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def _env_handle(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=False, subdir=True
            )
        return self._env

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lmdb_idx, label, n_frames = self.items[index]

        pick = _pick_frame_indices_tsn if self.temporal_jitter else _pick_frame_indices
        frame_indices = pick(n_frames, self.num_frames)

        env = self._env_handle()
        pil_frames: List[Image.Image] = []
        with env.begin() as txn:
            for fi in frame_indices:
                key = f"{lmdb_idx:08d}_{fi:04d}".encode()
                jpeg_bytes = txn.get(key)
                pil_frames.append(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))

        if self.tta:
            video_tensor = self.transform.tta(pil_frames)   # (10, T, C, H, W)
        else:
            video_tensor = self.transform(pil_frames)        # (T, C, H, W)

        return video_tensor, torch.tensor(label, dtype=torch.long)
