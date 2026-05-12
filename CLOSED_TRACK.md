# Closed Track — From-Scratch Recipe

This note describes the changes added to make the closed track (no pretrained weights) competitive, and how to reproduce / test them.

The classic-DL toolkit applied: stochastic depth, classifier dropout, label smoothing, AdamW + cosine LR with linear warmup, RandAugment-style augmentation on top of MixUp/CutMix, and a second 3D-CNN architecture (R(2+1)D-18) for ensembling and ablation.

---

## 1. What changed

### 1.1 TSM-ResNet — [src/models/tsm_resnet.py](src/models/tsm_resnet.py)

Two new constructor arguments:

| Arg | Default | Effect |
|---|---|---|
| `drop_path_rate` | `0.0` | Stochastic depth (Huang et al., ECCV'16) on the **residual branch only** — identity skip is preserved. Drop probability scales linearly across blocks (0 at the first, `drop_path_rate` at the last). |
| `dropout` | `0.0` | Dropout right before the final `Linear` classifier head. |

Implementation notes:

- `_drop_path` is per-sample; survivors are rescaled by `1/keep_prob`.
- `_ResBlockWithDropPath` wraps a torchvision `BasicBlock` or `Bottleneck`, replays its forward but applies `_drop_path` on the residual just before the identity addition.
- When `drop_path_rate == 0`, blocks are **not** wrapped, so the module structure is identical to before — old `tsm_pretrained_18_tta3.pt` checkpoints still load.
- `dropout == 0` uses `nn.Identity()` (no parameters), so the `state_dict` is also unchanged.

### 1.2 R(2+1)D-18 — [src/models/r2plus1d.py](src/models/r2plus1d.py)

A new model that factorizes each 3D conv into a spatial `(1, k, k)` conv followed by a temporal `(k, 1, 1)` conv (Tran et al., CVPR'18). Different inductive bias from TSM — useful for ensembling and as an architectural ablation.

- Built on torchvision's `r2plus1d_18(weights=None)` (architecture only — no pretrained weights, closed-track legal).
- Permutes input from `(B, T, C, H, W)` (codebase convention) to `(B, C, T, H, W)` (torchvision convention).
- Same `dropout` knob before the head.

Registered in `build_model` in [src/train.py](src/train.py) under `model.name == "r2plus1d"` and exported from [src/models/__init__.py](src/models/__init__.py).

### 1.3 Temporally-consistent RandAugment — [src/utils.py](src/utils.py)

A pool of nine PIL ops: AutoContrast, Equalize, Posterize, Solarize, Sharpness, ShearX/Y, TranslateX/Y. For each clip:

1. Sample `rand_augment_ops` ops once.
2. Apply the **same** ops with the **same** magnitudes to every frame.

This preserves temporal trajectories (a panning frame doesn't suddenly stop panning mid-clip). Sits on top of the existing per-clip-consistent augmentation pipeline (random resized crop, hflip, rotation, color jitter, gaussian blur, random erasing).

`build_transforms` and `VideoTransform.__init__` accept `rand_augment_ops` and `rand_augment_magnitude`.

### 1.4 Training recipe — [src/train.py](src/train.py)

Two new helper functions plus wiring in `main`:

- `build_optimizer(model, cfg)` → Adam (default, backward compatible), AdamW, or SGD with Nesterov momentum, all with optional `weight_decay`.
- `build_scheduler(optimizer, cfg)` → `none` or `cosine`. Cosine optionally chains a `LinearLR` warmup (`start_factor=0.01`, `total_iters=warmup_epochs`) before `CosineAnnealingLR(T_max=epochs - warmup_epochs)` via `SequentialLR`.
- `CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)` — composes correctly with the existing `mixed_loss` used by MixUp/CutMix.
- Per-epoch log now includes the current learning rate.

### 1.5 Config knobs — [src/configs/train/default.yaml](src/configs/train/default.yaml)

New keys (all default to behavior-preserving values, so the existing `tsm_pretrained` experiment is unaffected):

```yaml
optimizer: adam              # adam | adamw | sgd
weight_decay: 0.0
momentum: 0.9                # only used by sgd
scheduler: none              # none | cosine
warmup_epochs: 0
label_smoothing: 0.0
rand_augment_ops: 0          # 0 disables
rand_augment_magnitude: 0.5
```

### 1.6 New experiment configs

- [src/configs/model/r2plus1d.yaml](src/configs/model/r2plus1d.yaml) — model definition.
- [src/configs/experiment/tsm_closed.yaml](src/configs/experiment/tsm_closed.yaml) — closed-track TSM-ResNet18 recipe.
- [src/configs/experiment/r2plus1d_closed.yaml](src/configs/experiment/r2plus1d_closed.yaml) — closed-track R(2+1)D-18 recipe.

Both closed configs use:

| Setting | Value |
|---|---|
| `num_frames` | 16 |
| `epochs` | 80 |
| `optimizer` | AdamW |
| `lr` | 1e-3 |
| `weight_decay` | 5e-4 |
| `scheduler` | cosine, 5-epoch linear warmup |
| `label_smoothing` | 0.1 |
| `dropout` | 0.5 |
| `mixup_alpha / cutmix_prob` | 0.4 / 0.5 |
| `rand_augment_ops / magnitude` | 2 / 0.5 |
| TSM only — `drop_path_rate` | 0.2 (linearly scaled across 8 blocks) |

---

## 2. How to test it

All commands assume you are in the `src/` directory (Hydra needs to find `configs/`).

### 2.1 Smoke test (1–2 minutes on CPU)

Verify the new pipeline runs end-to-end on a tiny subset before launching a long run:

```bash
cd src
python train.py experiment=tsm_closed \
    dataset.max_samples=64 \
    training.epochs=2 \
    training.batch_size=4 \
    training.num_workers=0 \
    training.device=cpu \
    training.checkpoint_path=$PWD/../models/_smoke_tsm.pt
```

Expect to see:

- The full merged config printed at the top, with `optimizer: adamw`, `scheduler: cosine`, `label_smoothing: 0.1`, `drop_path_rate: 0.2`, etc.
- Two epochs of `train loss / acc / val loss / acc` lines, each prefixed with the current learning rate (warmup brings it up from `1e-5` toward `1e-3`).
- A checkpoint written to `models/_smoke_tsm.pt` if val accuracy improves.

Repeat for R(2+1)D:

```bash
python train.py experiment=r2plus1d_closed \
    dataset.max_samples=64 \
    training.epochs=2 \
    training.batch_size=4 \
    training.num_workers=0 \
    training.device=cpu \
    training.checkpoint_path=$PWD/../models/_smoke_r2plus1d.pt
```

### 2.2 Full closed-track training (overnight, GPU)

```bash
cd src
python train.py experiment=tsm_closed
python train.py experiment=r2plus1d_closed
```

Best checkpoints land in `../models/tsm_closed.pt` and `../models/r2plus1d_closed.pt` (paths set in the experiment YAMLs). Each checkpoint embeds the full Hydra config, so evaluation and submission scripts reload the exact architecture without further flags.

You can override anything from the CLI without editing files — useful for sweeps:

```bash
# Try SGD instead of AdamW
python train.py experiment=tsm_closed \
    training.optimizer=sgd training.lr=0.05 training.momentum=0.9

# Sweep LR
for lr in 3e-4 1e-3 3e-3; do
  python train.py experiment=tsm_closed training.lr=$lr \
      training.checkpoint_path=$PWD/../models/tsm_closed_lr${lr}.pt
done
```

### 2.3 Evaluate on the full validation split

```bash
cd src
python evaluate.py training.checkpoint_path=$PWD/../models/tsm_closed.pt
python evaluate.py training.checkpoint_path=$PWD/../models/tsm_closed.pt dataset.tta=true
```

`dataset.tta=true` enables 10-crop test-time augmentation (5 spatial positions x 2 flips, averaged). It typically adds ~0.5–1 point of top-1.

### 2.4 Generate a submission CSV

```bash
cd src
python create_submission.py \
    training.checkpoint_path=$PWD/../models/tsm_closed.pt \
    dataset.tta=true \
    dataset.submission_output=$PWD/../submissions/tsm_closed_tta.csv
```

The CSV format is `video_name,predicted_class` and matches the expected Kaggle layout.

### 2.5 Quick unit checks (no data needed)

Sanity-check that the new modules build and forward shapes are correct:

```bash
cd src
python - <<'PY'
import torch
from models.tsm_resnet import TSMResNet
from models.r2plus1d import R2Plus1D

x = torch.randn(2, 16, 3, 224, 224)  # (B, T, C, H, W)

m = TSMResNet(num_classes=33, num_frames=16, pretrained=False,
              backbone="resnet18", drop_path_rate=0.2, dropout=0.5)
print("TSM logits:", m(x).shape)  # expect torch.Size([2, 33])

m = R2Plus1D(num_classes=33, pretrained=False, dropout=0.5)
print("R(2+1)D logits:", m(x).shape)  # expect torch.Size([2, 33])
PY
```

---

## 3. Backward compatibility

- Old `tsm_pretrained_18_tta3.pt` checkpoints still load: defaults of `drop_path_rate=0` and `dropout=0` keep the module structure identical to the original.
- The `tsm_pretrained` experiment is untouched and still runs the original Adam-only recipe — the new optimizer/scheduler/label-smoothing knobs default to the previous behavior.
- All new code paths are gated behind config flags; nothing changes unless an experiment YAML opts in.
