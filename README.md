# What Happens Next? — Video Action Recognition

**CSC_43M04_EP — Modal d'informatique, École Polytechnique (2026)**  
**Team 8 — Florian Guillaumey & Andrea Signoretti**

Temporal action prediction on a 33-class subset of Something-Something V2 (SSv2).
Each clip is represented as **four JPEG frames** covering the first ~40% of the original video;
the model must predict the action outcome from motion onset alone.

---

## Tracks

| Track | Constraint | Best result | Details |
|---|---|---|---|
| **A — Closed World** | Train from scratch, no pretrained weights | **47.85% Private Kaggle top-1 score** (val-dir) | [docs/TRACK_A.md](docs/TRACK_A.md) |
| **B — Open World** | External data and pretrained models allowed | **0.6586 Private Kaggle top-1** (16th overall) | [docs/TRACK_B.md](docs/TRACK_B.md) |

A digest of all results across both tracks is in [RESULTS.md](RESULTS.md).  
The full academic report is at [Experiment_report.pdf](Experiment_report.pdf).

---

## Repository Layout

```
src/                        # All Python source code
  train.py                  # Main training entry point (Hydra)
  evaluate.py               # Evaluation on full val set
  create_submission.py      # Generate Kaggle submission CSV
  train_kfold.py            # K-fold / rotating-fold training
  models/                   # Model architectures
    trackA_video_former_lite.py  # VideoFormer-Lite (Track A)
    videomae.py             # VideoMAE wrapper (Track B)
    vjepa.py                # V-JEPA 2 wrapper (Track B)
    ...
  dataset/
    video_dataset.py        # VideoFrameDataset, TSN jitter, multi-clip TTA
    lmdb_dataset.py         # Optional LMDB-backed dataset
  configs/                  # Hydra YAML configs
    experiment/             # Per-experiment presets (recommended entry point)
    model/                  # Model-specific configs
    train/                  # Training recipe defaults
  honest_ensemble.py        # Cross-preprocessing uniform ensemble (Track B)
  ensemble_gradient.py      # Learned-weight gradient ensemble
  ensemble_per_class.py     # Per-class accuracy-weighted ensemble
  cache_test_softmax.py     # Cache softmax outputs for ensemble
  pseudo_label.py           # Pseudo-labelling pipeline (Track B)
  extract_ssv2_frames.py    # Window-capped source-frame re-extraction (Track B)
scripts/                    # Training and ablation shell scripts
submissions/                # All Kaggle submission CSVs
models/                     # Val-accuracy CSVs for every training run
logs/                       # Done-markers for reboot-resilient pipeline
docs/
  TRACK_A.md                # Track A: method, experiments, ablations, results
  TRACK_B.md                # Track B: method, integrity story, results
  figures/
    track_a/
    track_b/
  reports/
    figures/
    report.pdf
    report.tex
    ...

```

---

## Dataset

1. Download the prepared dataset from Google Drive: [frames.zip](https://drive.google.com/file/d/1SlRJBD6cyXMr5772kOKe5xXAU9Scu5vR/view?usp=sharing)
2. Unzip so that `processed_data/` sits at the repository root, containing:
   - `processed_data/train/` — class subfolders `000_ClassName/video_<id>/frame_*.jpg`
   - `processed_data/val/` — same layout
   - `processed_data/test/` — video folders without class prefix

---

## Environment

Python 3.10+, managed with `uv`:

```bash
uv sync
```

All training and evaluation commands must be run from `src/` so Hydra resolves `configs/`:

```bash
cd src
```

---

## Quick Start — Track A (Closed World)

```bash
cd src

# Train the best single model (TSM-ResNet18, T=4, focal loss, rotating folds)
python train.py experiment=trackA_tsm_ultra_v2_rotating

# Evaluate on full val set
python evaluate.py training.checkpoint_path=../models/tsm_ultra_v2_rotating.pt

# Generate a submission CSV
python create_submission.py training.checkpoint_path=../models/tsm_ultra_v2_rotating.pt

# Reproduce the best ensemble (log-softmax avg of 3 models)
# First generate val logits for each checkpoint, then:
python ensemble_benchmark.py
```

Experiment configs for Track A live in `src/configs/experiment/trackA_*.yaml`.

---

## Quick Start — Track B (Open World)

Track B requires HuggingFace checkpoints (~1–3 GB each). Set `HF_HOME` to a
location with sufficient disk space before running.

```bash
cd src

# Fine-tune V-JEPA 2 (three-stage progressive unfreezing)
bash run_vjepa_ft.sh

# Cache softmax outputs for ensemble
python cache_test_softmax.py training.checkpoint_path=../models/vjepa_ft_s2.pt

# Build the honest ensemble from cached softmax tensors
python honest_ensemble.py

# Generate the ensemble submission CSV
python create_submission.py  # uses ensemble_honest_v*.csv internally
```

For the full reproducible pipeline with reboot-resilience, see `src/run_ovn2.sh`.

---

## Adding a New Model

1. Implement `torch.nn.Module` in `src/models/your_model.py` — input `(B, T, C, H, W)`, output `(B, num_classes)`.
2. Register it in `build_model()` in `src/train.py`.
3. Add `src/configs/model/your_model.yaml` (use `# @package _global_` header).
4. Add `src/configs/experiment/your_experiment.yaml` that overrides `/model: your_model`.
5. Train: `python train.py experiment=your_experiment`

`evaluate.py` and `create_submission.py` need no changes — they reconstruct the model entirely from the saved checkpoint config.

---

## Citing

If you build on this work, please cite the challenge report:

```
Florian Guillaumey and Andrea Signoretti.
"What Happens Next? Closed- and Open-World Video Action Recognition
on a 33-Class Something-Something V2 Subset."
CSC_43M04_EP — Modal d'informatique, École Polytechnique, 2026.
```
