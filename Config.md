# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

Video classification challenge (CSC_43M04_EP — Modal d'informatique). The task is action recognition over 33 classes on the "What_Happens_Next?" dataset. Each video is a folder of extracted JPEG frames; models consume a fixed number of frames per clip and output class logits.

## Setup

```bash
uv sync
```

All scripts must be run from `src/` so Hydra resolves `configs/`:

```bash
cd src
python train.py                          # uses defaults from configs/config.yaml
python train.py experiment=cnn_lstm      # override experiment
python train.py training.epochs=10 training.lr=0.001
```

## Key Commands

**Train:**
```bash
cd src && python train.py experiment=tsm_ultra_v2
```

**Evaluate on full val set:**
```bash
cd src && python evaluate.py
cd src && python evaluate.py dataset.tta=true   # test-time augmentation (10 crops)
```

**Generate submission CSV:**
```bash
cd src && python create_submission.py
cd src && python create_submission.py training.checkpoint_path=/path/to/model.pt dataset.tta=true
```

## Data Layout

```
processed_data/           # must exist at project root (next to src/)
  train/
    000_ClassName/
      video_12345/
        frame_000.jpg
        frame_001.jpg
    001_AnotherClass/
      ...
  val/                    # same structure; used only by evaluate.py
  test/                   # video_* folders without class subfolders; used by create_submission.py
```

Class index is parsed from the leading digits in the class folder name (`000_...` → class 0).

## Architecture

### Config System (Hydra)
`configs/config.yaml` is the root; it composes `model/`, `data/`, `train/`, and `experiment/` groups. An **experiment** YAML (`configs/experiment/`) is the recommended entry point — it overrides model and training settings without touching Python.

Current experiments (all closed-track, trained from scratch): `baseline_from_scratch`, `tsm_from_scratch`, `tsm_ultra_v2`, `video_former_lite_ultra`.

### Models
All models share the same interface: input `(B, T, C, H, W)`, output logits `(B, num_classes)`.

- **CNNBaseline** (`models/cnn_baseline.py`): ResNet18 backbone, average pool over time frames.
- **CNNLSTM** (`models/cnn_lstm.py`): ResNet18 features per frame fed into an LSTM.
- **TSMResNet** (`models/tsm_resnet.py`): Best-performing model. Wraps every ResNet residual block with `TemporalShift`, which shifts 1/8 of channels backward and 1/8 forward in time — zero added parameters. Defaults: ResNet18 backbone, `fold_div=8`, `num_frames` from dataset config.

`build_model()` in `train.py` is the single factory; add new models there and register a `configs/model/your_model.yaml`.

### Checkpoint Format
`.pt` files saved by `train.py` contain:
```python
{
  "model_state_dict": ...,
  "config": <full Hydra config as dict>,   # used by evaluate.py and create_submission.py
  "val_accuracy": float,
  "num_frames": int,
  "pretrained": bool,
  ...
}
```
`evaluate.py` and `create_submission.py` rebuild the model entirely from the saved `config` — no need to specify architecture flags at eval time.

### Data Pipeline
- `VideoFrameDataset` (`dataset/video_dataset.py`): samples `num_frames` frames per clip.
  - Training: TSN-style temporal jitter (random frame per segment).
  - Val/test: uniform linspace sampling (deterministic).
- `VideoTransform` (`utils.py`): temporally-consistent augmentation — random parameters sampled once and applied identically to all frames in a clip. Has a `.tta()` method for 10-crop test-time augmentation (5 spatial positions × 2 flips).
- Optional LMDB cache: `make_dataset()` in `utils.py` auto-switches to `LMDBVideoDataset` if an `lmdb_path` is provided.

### Training Details
- Default optimizer: Adam, lr=1e-4.
- MixUp (`mixup_alpha=0.4`) and CutMix (`cutmix_prob=0.5`) are both active by default; one is chosen randomly per batch (CutMix takes priority when triggered).
- Best checkpoint is saved by val accuracy on an internal split of `train_dir` (controlled by `dataset.val_ratio`). The full `val_dir` is only used by `evaluate.py`.
- `training.device=cuda` (falls back to CPU if CUDA unavailable).

## Adding a New Model

1. Implement `nn.Module` in `src/models/your_model.py` with input `(B, T, C, H, W)` → output `(B, num_classes)`.
2. Add a branch in `build_model()` in `train.py`.
3. Add `src/configs/model/your_model.yaml` (use `# @package _global_` header).
4. Add `src/configs/experiment/your_experiment.yaml` with `- override /model: your_model`.
5. Train with `python train.py experiment=your_experiment`.

`evaluate.py` and `create_submission.py` need no changes.
