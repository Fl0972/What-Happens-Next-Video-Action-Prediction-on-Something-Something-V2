"""
Small helpers: reproducibility, image transforms, and metric computation.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Temporally-consistent RandAugment-style ops
# ---------------------------------------------------------------------------
# Each entry is (name, op_factory). op_factory(magnitude_in_[0,1]) returns a
# PIL.Image -> PIL.Image callable. Sampled once per clip and applied to every
# frame, so the temporal trajectory is preserved.

def _rand_aug_ops() -> List[Tuple[str, Callable[[float], Callable]]]:
    return [
        ("AutoContrast", lambda _m: ImageOps.autocontrast),
        ("Equalize",     lambda _m: ImageOps.equalize),
        ("Posterize",    lambda m: (lambda img: ImageOps.posterize(img, max(4, 8 - int(round(m * 4)))))),
        ("Solarize",     lambda m: (lambda img: ImageOps.solarize(img, int(round(256 - m * 256))))),
        ("Sharpness",    lambda m: (lambda img: ImageEnhance.Sharpness(img).enhance(1.0 + m * 0.9 * random.choice((-1, 1))))),
        ("ShearX",       lambda m: (lambda img: img.transform(img.size, Image.AFFINE, (1, m * 0.3 * random.choice((-1, 1)), 0, 0, 1, 0), resample=Image.BILINEAR))),
        ("ShearY",       lambda m: (lambda img: img.transform(img.size, Image.AFFINE, (1, 0, 0, m * 0.3 * random.choice((-1, 1)), 1, 0), resample=Image.BILINEAR))),
        ("TranslateX",   lambda m: (lambda img: img.transform(img.size, Image.AFFINE, (1, 0, m * img.size[0] * 0.2 * random.choice((-1, 1)), 0, 1, 0), resample=Image.BILINEAR))),
        ("TranslateY",   lambda m: (lambda img: img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, m * img.size[1] * 0.2 * random.choice((-1, 1))), resample=Image.BILINEAR))),
    ]


def _sample_rand_augment(num_ops: int, magnitude: float) -> List[Callable]:
    """Sample ``num_ops`` ops once (per clip). Returns ready-to-call PIL transforms."""
    if num_ops <= 0:
        return []
    pool = _rand_aug_ops()
    chosen = random.sample(pool, k=min(num_ops, len(pool)))
    return [factory(magnitude) for _name, factory in chosen]


def set_seed(seed: int) -> None:
    """Make runs reproducible (as far as CUDA allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VideoTransform:
    """
    Temporally-consistent augmentation: random parameters are sampled once per
    video and applied identically to every frame.

    transform(frames: List[PIL.Image]) -> Tensor (T, C, H, W)
    transform.tta(frames)             -> Tensor (10, T, C, H, W)  [5 crops × 2 flips]
    """

    def __init__(
        self,
        image_size: int = 224,
        is_training: bool = True,
        use_imagenet_norm: bool = True,
        rand_augment_ops: int = 0,
        rand_augment_magnitude: float = 0.5,
        horizontal_flip: bool = True,
    ) -> None:
        self.image_size = image_size
        self.is_training = is_training
        self.rand_augment_ops = int(rand_augment_ops)
        self.rand_augment_magnitude = float(rand_augment_magnitude)
        self.horizontal_flip = bool(horizontal_flip)
        if use_imagenet_norm:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        if self.is_training:
            # All random params sampled once — same for every frame in the clip
            i, j, h, w = self._crop_params(frames[0])
            # hflip is gated by ``horizontal_flip`` because several action classes
            # are direction-sensitive (e.g. "Pulling left/right"); flipping them
            # silently inverts the label.
            do_flip   = self.horizontal_flip and (random.random() < 0.5)
            brightness = random.uniform(0.6, 1.4)
            contrast   = random.uniform(0.6, 1.4)
            saturation = random.uniform(0.6, 1.4)
            hue        = random.uniform(-0.1, 0.1)
            do_gray    = random.random() < 0.1
            do_blur    = random.random() < 0.2
            sigma      = random.uniform(0.1, 2.0)
            do_erase   = random.random() < 0.3
            # RandAugment-style: pick ops once per clip, apply identically to every frame
            rand_aug = _sample_rand_augment(self.rand_augment_ops, self.rand_augment_magnitude)

        tensors: List[torch.Tensor] = []
        for img in frames:
            if self.is_training:
                img = F.resized_crop(img, i, j, h, w, [self.image_size, self.image_size])
                if do_flip:
                    img = F.hflip(img)
                img = F.adjust_brightness(img, brightness)
                img = F.adjust_contrast(img, contrast)
                img = F.adjust_saturation(img, saturation)
                img = F.adjust_hue(img, hue)
                if do_gray:
                    img = F.rgb_to_grayscale(img, num_output_channels=3)
                if do_blur:
                    img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
                for op in rand_aug:
                    img = op(img)
            else:
                img = F.resize(img, [self.image_size + 32, self.image_size + 32])
                img = F.center_crop(img, [self.image_size, self.image_size])

            t = F.to_tensor(img)
            t = F.normalize(t, self.mean, self.std)
            tensors.append(t)

        video = torch.stack(tensors, dim=0)  # (T, C, H, W)

        # Random Erasing: zero out the same rectangle on all frames
        if self.is_training and do_erase:
            video = self._random_erase(video)

        return video

    def tta(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        Test-time augmentation: 5 crops (center + 4 corners) × 2 (original + hflip)
        = 10 views. Returns (10, T, C, H, W).
        """
        S = self.image_size
        large = S + 32  # resize to 256 then crop 224

        resized = [F.resize(img, [large, large]) for img in frames]
        flipped = [F.hflip(img) for img in resized]

        half = (large - S) // 2
        offsets = [
            (0,    0),           # top-left
            (0,    large - S),   # top-right
            (large - S, 0),      # bottom-left
            (large - S, large - S),  # bottom-right
            (half, half),        # center
        ]

        views: List[torch.Tensor] = []
        for top, left in offsets:
            views.append(self._encode_crop(resized, top, left, S))
            views.append(self._encode_crop(flipped, top, left, S))

        return torch.stack(views, dim=0)  # (10, T, C, H, W)

    # ------------------------------------------------------------------
    def _encode_crop(self, imgs: List[Image.Image], top: int, left: int, size: int) -> torch.Tensor:
        tensors = []
        for img in imgs:
            t = F.to_tensor(F.crop(img, top, left, size, size))
            t = F.normalize(t, self.mean, self.std)
            tensors.append(t)
        return torch.stack(tensors, dim=0)  # (T, C, H, W)

    def _crop_params(self, img: Image.Image) -> Tuple[int, int, int, int]:
        width, height = img.size
        scale  = random.uniform(0.5, 1.0)
        ratio  = random.uniform(3 / 4, 4 / 3)
        crop_h = min(int(round(height * scale)), height)
        crop_w = min(int(round(crop_h * ratio)), width)
        top    = random.randint(0, height - crop_h)
        left   = random.randint(0, width  - crop_w)
        return top, left, crop_h, crop_w

    def _random_erase(self, video: torch.Tensor) -> torch.Tensor:
        T, C, H, W = video.shape
        erase_h = random.randint(int(H * 0.02), int(H * 0.33))
        erase_w = random.randint(int(W * 0.02), int(W * 0.33))
        top  = random.randint(0, H - erase_h)
        left = random.randint(0, W - erase_w)
        video = video.clone()
        video[:, :, top:top + erase_h, left:left + erase_w] = 0.0
        return video


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
    rand_augment_ops: int = 0,
    rand_augment_magnitude: float = 0.5,
    horizontal_flip: bool = True,
) -> VideoTransform:
    return VideoTransform(
        image_size=image_size,
        is_training=is_training,
        use_imagenet_norm=use_imagenet_norm,
        rand_augment_ops=rand_augment_ops,
        rand_augment_magnitude=rand_augment_magnitude,
        horizontal_flip=horizontal_flip,
    )


# ---------------------------------------------------------------------------
# MixUp / CutMix
# ---------------------------------------------------------------------------

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Returns (mixed_x, y_a, y_b, lambda)."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix: paste a rectangle from another clip. Returns (mixed_x, y_a, y_b, lambda)."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    B, T, C, H, W = x.shape
    cut_h = max(1, int(H * (1 - lam) ** 0.5))
    cut_w = max(1, int(W * (1 - lam) ** 0.5))
    cy = random.randint(0, H); cx = random.randint(0, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    mixed = x.clone()
    mixed[:, :, :, y1:y2, x1:x2] = x[idx, :, :, y1:y2, x1:x2]
    actual_lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, y, y[idx], actual_lam


def mixed_loss(
    criterion: Callable,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def make_dataset(
    root_dir: Path,
    num_frames: int,
    transform: VideoTransform,
    sample_list=None,
    lmdb_path: Optional[Path] = None,
    temporal_jitter: bool = False,
    tta: bool = False,
    interpolate_frames: bool = False,
) -> Dataset:
    """Return LMDBVideoDataset if lmdb_path exists, else VideoFrameDataset."""
    kwargs = dict(
        num_frames=num_frames,
        transform=transform,
        sample_list=sample_list,
        temporal_jitter=temporal_jitter,
        tta=tta,
        interpolate_frames=interpolate_frames,
    )
    if lmdb_path is not None and Path(lmdb_path).exists():
        from dataset.lmdb_dataset import LMDBVideoDataset
        return LMDBVideoDataset(lmdb_path, root_dir, **kwargs)
    from dataset.video_dataset import VideoFrameDataset
    return VideoFrameDataset(root_dir, **kwargs)


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    max_k = max(topk)
    batch_size = targets.size(0)
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))
    return tuple(correct[:k].reshape(-1).float().sum() / batch_size for k in topk)


def build_weighted_sampler(
    samples: Sequence[Tuple[Path, int]],
    power: float = 0.5,
) -> WeightedRandomSampler:
    """
    WeightedRandomSampler with per-sample weight proportional to ``1 / count^power``.

    power=0.0  -> uniform sampling (no rebalancing).
    power=0.5  -> sqrt-frequency reweighting (recommended default; soft balance).
    power=1.0  -> full inverse-frequency reweighting (every class drawn equally often).

    ``num_samples`` is set to ``len(samples)`` so one "epoch" still sees the same
    number of clips as before (just with a class-balanced distribution).
    """
    counts = Counter(c for _, c in samples)
    weights = [1.0 / (counts[c] ** float(power)) for _, c in samples]
    weights_t = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_t, num_samples=len(samples), replacement=True)


def split_train_val(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_ratio <= 0.0:
        return shuffled, []

    n_val = int(round(len(shuffled) * val_ratio))
    n_val = max(1, n_val) if len(shuffled) > 1 else 0
    val_samples   = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    if len(train_samples) == 0:
        train_samples = val_samples[:-1]
        val_samples   = val_samples[-1:]
    return train_samples, val_samples
