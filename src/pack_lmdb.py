"""
Pack a processed_data split into a single LMDB file for fast NFS I/O.

One-time step (run from src/):

    python pack_lmdb.py --root /Data/.../processed_data/train \\
                        --out  /Data/.../train.lmdb

Then set dataset.lmdb_path in the config (or pass on the CLI):

    python train.py dataset.lmdb_path=/Data/.../train.lmdb

LMDB layout (matches LMDBVideoDataset in dataset/lmdb_dataset.py):
  b"__samples__"          -> pickle[(rel_path, label, n_frames), ...]
  b"{idx:08d}_{fi:04d}"  -> raw JPEG bytes
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lmdb
from tqdm import tqdm

from dataset.video_dataset import _list_frame_paths, collect_video_samples


def pack(root: Path, out: Path, map_size_gb: int = 100) -> None:
    root = root.resolve()
    samples = collect_video_samples(root)
    print(f"Found {len(samples)} videos under {root}")

    out.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out), map_size=map_size_gb * 1024 ** 3, subdir=True)

    meta = []
    with env.begin(write=True) as txn:
        for idx, (video_dir, label) in enumerate(tqdm(samples, desc="Packing LMDB")):
            frame_paths = _list_frame_paths(video_dir)
            n_frames = len(frame_paths)
            rel = str(video_dir.resolve().relative_to(root))
            meta.append((rel, label, n_frames))
            for fi, fp in enumerate(frame_paths):
                key = f"{idx:08d}_{fi:04d}".encode()
                txn.put(key, fp.read_bytes())

        txn.put(b"__samples__", pickle.dumps(meta))

    env.close()
    print(f"Done — {len(samples)} videos packed into {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Dataset split root (e.g. processed_data/train)")
    parser.add_argument("--out",  required=True, type=Path, help="Output LMDB path")
    parser.add_argument("--map_size_gb", type=int, default=100, help="LMDB map size in GB (default 100)")
    args = parser.parse_args()
    pack(args.root, args.out, args.map_size_gb)
