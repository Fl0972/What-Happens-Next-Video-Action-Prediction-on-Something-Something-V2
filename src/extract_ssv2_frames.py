#!/usr/bin/env python3
"""
Re-extract N frames per clip from the source Something-Something-v2 videos,
**strictly within the provided (non-privileged) temporal window**.

Why window-capping
------------------
The competition serves only the *first portion* of each clip (the shipped 4
frames span ~0..40% of the source video) and asks you to *anticipate* the
action. Sampling the full source video would feed the model the withheld
ending — "use of privileged information / data leakage", which the rules
prohibit in both tracks. So for each clip we anchor the sampling window to that
clip's own shipped frames: we locate where the LAST shipped frame (frame_003)
sits in the source video and sample our N frames only within `[0, that index]`.
The search for that index is restricted to the first 70% of the source so a
noisy match can never land in the withheld tail.

This gives higher temporal resolution of the *allowed* early window (legitimate
in Track B, which permits external data) without ever using a withheld frame.

Modes
-----
  --verify : report, over a sample of clips, the window fraction (i_end / n)
             and the endpoint match error vs the shipped frames. Window
             fractions should cluster well below 1.0 (≈0.4 here); low endpoint
             error confirms the window is temporally aligned.
  (default): extract N frames per clip within its window into a parallel tree
             (val2 -> val2_win16), preserving structure. Resumable.

Source layout expected: SRC/<id>.webm  (also tries .mp4/.avi/.mkv).
Frames are written frame_000.jpg .. frame_{N-1}.jpg at <size>x<size>.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

VIDEO_EXTS = (".webm", ".mp4", ".avi", ".mkv")
MATCH_SIZE = 112          # low-res size for robust temporal matching
SEARCH_FRAC = 0.50        # window end is searched only in the first half — the true
                          # frame_003 sits at ~0.40 (max ~0.52), and the withheld
                          # outcome is at 0.6+, so this can never reach the tail.


def find_source(src_dir: Path, vid_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        p = src_dir / f"{vid_id}{ext}"
        if p.is_file():
            return p
    return None


def _decode_all(video: Path, tmp: Path, size: int) -> List[Path]:
    """Decode every frame of `video` to tmp at <size>x<size> (resize+centre-crop)."""
    vf = f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
         "-vf", vf, "-q:v", "2", str(tmp / "f_%05d.jpg")],
        check=True,
    )
    return sorted(tmp.glob("f_*.jpg"))


def _lowres(path_or_img, sz: int = MATCH_SIZE) -> np.ndarray:
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    return np.asarray(img.convert("RGB").resize((sz, sz)), dtype=np.float32)


def _last_shipped_frame(clip_dir: Path) -> Optional[Path]:
    fs = sorted(clip_dir.glob("frame_*.jpg"))
    return fs[-1] if fs else None


def _window_end(frames: List[Path], shipped_last: Path) -> int:
    """Index in `frames` of the best match to the last shipped frame,
    searched only within the first SEARCH_FRAC of the clip (can't reach tail)."""
    n = len(frames)
    hi = max(2, int(round(SEARCH_FRAC * n)))
    target = _lowres(shipped_last)
    best_i, best_e = 0, float("inf")
    for i in range(hi):
        e = float(np.abs(_lowres(frames[i]) - target).mean())
        if e < best_e:
            best_e, best_i = e, i
    return best_i


def _window_indices(i_end: int, n_want: int) -> List[int]:
    """n_want indices uniformly in [0, i_end] (repeats allowed if window short)."""
    if i_end <= 0:
        return [0] * n_want
    return list(np.linspace(0, i_end, n_want).round().astype(int))


def extract_one(clip_dir: Path, vid_id: str, src_dir: Path, out_dir: Path,
                n_frames: int, size: int) -> Tuple[str, str]:
    done_marker = out_dir / f"frame_{n_frames - 1:03d}.jpg"
    if done_marker.is_file():
        return vid_id, "skip"
    shipped_last = _last_shipped_frame(clip_dir)
    if shipped_last is None:
        return vid_id, "no_shipped"
    video = find_source(src_dir, vid_id)
    if video is None:
        return vid_id, "missing_source"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"ssv2_{vid_id}_"))
    try:
        frames = _decode_all(video, tmp, size)
        if not frames:
            return vid_id, "decode_empty"
        i_end = _window_end(frames, shipped_last)
        for out_i, src_i in enumerate(_window_indices(i_end, n_frames)):
            shutil.copyfile(frames[src_i], out_dir / f"frame_{out_i:03d}.jpg")
        return vid_id, "ok"
    except subprocess.CalledProcessError:
        return vid_id, "ffmpeg_error"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _list_clip_dirs(in_root: Path) -> List[Path]:
    return sorted(in_root.rglob("video_*"))


def _job(args):
    clip, in_root, out_root, n_frames, size, src_dir = args
    vid_id = clip.name.replace("video_", "")
    out_dir = out_root / clip.relative_to(in_root)
    return extract_one(clip, vid_id, src_dir, out_dir, n_frames, size)


def run_extract(in_root: Path, out_root: Path, src_dir: Path,
                n_frames: int, size: int, workers: int) -> None:
    clips = _list_clip_dirs(in_root)
    print(f"[extract] {in_root} -> {out_root}: {len(clips)} clips, N={n_frames} "
          f"(window-capped to shipped frames), {workers} workers", flush=True)
    jobs = [(c, in_root, out_root, n_frames, size, src_dir) for c in clips]
    counts: dict = {}
    fails: List[str] = []
    with mp.Pool(workers) as pool:
        for i, (vid_id, status) in enumerate(pool.imap_unordered(_job, jobs, chunksize=8), 1):
            counts[status] = counts.get(status, 0) + 1
            if status in ("missing_source", "decode_empty", "ffmpeg_error", "no_shipped"):
                fails.append(f"{vid_id}:{status}")
            if i % 2000 == 0:
                print(f"  {i}/{len(clips)}  {counts}", flush=True)
    print(f"[extract] done: {counts}", flush=True)
    if fails:
        log = out_root / "_extract_failures.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(fails))
        print(f"[extract] {len(fails)} failures logged to {log}", flush=True)


def run_verify(in_root: Path, src_dir: Path, size: int, sample: int) -> None:
    clips = [c for c in _list_clip_dirs(in_root) if _last_shipped_frame(c)]
    rng = np.random.default_rng(0)
    pick = [clips[i] for i in rng.choice(len(clips), min(sample, len(clips)), replace=False)]
    fracs, endpoint_err, checked = [], [], 0
    for clip in pick:
        vid_id = clip.name.replace("video_", "")
        video = find_source(src_dir, vid_id)
        if video is None:
            continue
        tmp = Path(tempfile.mkdtemp(prefix="verify_"))
        try:
            frames = _decode_all(video, tmp, size)
            if not frames:
                continue
            i_end = _window_end(frames, _last_shipped_frame(clip))
            fracs.append(i_end / max(1, len(frames) - 1))
            # endpoint alignment: our last sampled frame vs shipped last frame
            our_last = _lowres(frames[_window_indices(i_end, 16)[-1]])
            endpoint_err.append(float(np.abs(our_last - _lowres(_last_shipped_frame(clip))).mean()))
            checked += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    fr = np.array(fracs)
    print(f"[verify] clips checked: {checked}")
    print(f"[verify] window-end fraction (i_end/n): median={np.median(fr):.2f} "
          f"p90={np.percentile(fr,90):.2f} max={fr.max():.2f}  (must stay well below 1.0)")
    print(f"[verify] endpoint match error vs shipped last frame: median={np.median(endpoint_err):.1f}")
    print("[verify] OK if fractions cluster ~0.4 and never approach 1.0 — no withheld frames used.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="Dir of source SSv2 videos (<id>.webm)")
    ap.add_argument("--in-root", type=Path, default=Path("/Data/florian.guillaumey/val2"),
                    help="Challenge tree (its shipped frames define each clip's window)")
    ap.add_argument("--out-root", type=Path, default=Path("/Data/florian.guillaumey/val2_win16"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)  # 256 serves V-JEPA; VideoMAE downscales to 224
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-sample", type=int, default=60)
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"--src not found: {args.src}")

    if args.verify:
        run_verify(args.in_root / "train", args.src, args.size, args.verify_sample)
        return

    for split in args.splits:
        in_root = args.in_root / split
        if not in_root.is_dir():
            print(f"[skip] {in_root} missing", flush=True)
            continue
        run_extract(in_root, args.out_root / split, args.src, args.n_frames, args.size, args.workers)


if __name__ == "__main__":
    main()
