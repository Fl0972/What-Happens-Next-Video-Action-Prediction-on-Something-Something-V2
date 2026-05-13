# VideoFormer-Lite — Closed Track

A hybrid **2D-CNN + temporal Transformer** for short-clip action recognition on
the *What_Happens_Next?* dataset. Trained from scratch (no pretrained weights),
designed to fit a single GPU and finish in well under 16 hours.

> **Why a new model?** The previous R(2+1)D-18 closed-track run reached
> 6.52 % top-1 / 24.80 % top-5 on the 6,745-clip val set. The architecture
> wasn't wrong, but the recipe was thin (Adam @ 1e-4, no scheduler, no weight
> decay, no label smoothing, RandAugment disabled, and `hflip(p=0.5)` enabled
> despite ~6 direction-sensitive classes). VideoFormer-Lite combines a smaller,
> faster backbone with a deliberate regularization stack tuned for from-scratch
> training on a 45 k-clip, 33-class dataset.

---

## 1. Architecture

```
Input  : (B, T=4, C=3, H=224, W=224)

per-frame ResNet18 (random init, fc removed)        # spatial reasoning
        -> reshape (B*T, 3, 224, 224)
        -> conv stack ... GAP                       # (B*T, 512)
        -> reshape (B, T, 512)                      # 4 frame tokens

[CLS] (1, 512)  ||  4 frame tokens                  # (B, 5, 512)
 +  learnable temporal positional embedding (1, 5, 512)

Transformer encoder x 2 (pre-LN)                    # temporal reasoning
   block_l(x) = x + DropPath(MHSA(LN(x)))
   block_l(x) = x + DropPath(MLP(LN(x)))
   d_model=512, n_heads=8, d_ff=2048, GELU
   drop_path linearly 0.0 -> 0.1

LayerNorm( [CLS] )                                  # (B, 512)
Dropout(p=0.2)
Linear(512, 33)                                     # logits
```

| Component                  |   Params |
|----------------------------|---------:|
| ResNet18 backbone (no fc)  |  ~11.2 M |
| Temporal Transformer (×2)  |   ~6.3 M |
| `[CLS]` + positional embed |   ~3.1 k |
| LayerNorm + classifier     |   ~18 k  |
| **Total**                  | **~17.5 M** |

### Why this architecture for **4-frame** clips

Long-clip models (3D CNNs, TSM with 8-16 frames) waste parameters and FLOPs on
temporal patterns we don't have here — at T=4 there's no point growing the
temporal receptive field with stacked shifts or factored convs. The Transformer
sees all 4 tokens at once: any pairwise frame interaction (1↔2, 1↔4, etc.) is
*one* attention step, not many shifted convolutions. The CNN keeps doing what it
does best (cheap spatial feature extraction), and the Transformer handles the
short, dense temporal reasoning.

---

## 2. Compute budget

| Metric                      | Estimate                          |
|-----------------------------|-----------------------------------|
| FLOPs / clip (fwd)          | ~7.2 GFLOPs (CNN ≈ 99 %)          |
| Params                      | ~17.5 M                           |
| VRAM @ B=32, AMP            | ~2.5–3.5 GB                       |
| Step time (1 GPU, AMP)      | ~0.10–0.15 s / step               |
| ~35 k samples → ~1100 steps |                                   |
| Time / epoch                | ~3 min                            |
| **60 epochs**               | **~3 h** (well under the 16h cap) |

Where the budget goes: ResNet18 forward dominates at ~1.8 GFLOPs/frame × 4 frames.
The Transformer is essentially free (~0.03 GFLOPs/clip) — 5 tokens × d=512 is
tiny compared to the spatial backbone.

---

## 3. Lecture cross-references

The architecture and training recipe are anchored in the course material:

| Idea                                   | Lecture / slide        |
|----------------------------------------|------------------------|
| Scaled-dot-product attention, MHSA     | Lecture 6 — self-attention |
| Learnable positional encoding          | Lecture 6 — positional encoding |
| `[CLS]` token classifier               | Lecture 6 — ViT slides |
| Pre-LN encoder block                   | Lecture 6 — Transformer encoder |
| Hybrid CNN + self-attention            | Lecture 6, slide 27 ("CNN with self-attention") |
| Bias-variance — small model + strong regularization on a modest dataset | Lecture 7 |
| Stochastic depth (DropPath)            | Lecture 7 — depth regularization |
| Label smoothing, MixUp, CutMix         | Lecture 7 — output / input regularization |

---

## 4. Training recipe (`configs/experiment/video_former_lite_closed.yaml`)

### Optimization
- **AdamW**, lr **5 × 10⁻⁴**, weight_decay **5 × 10⁻²**.
  - Higher weight decay than R(2+1)D (5e-4) because the Transformer half is
    much more prone to overfitting on a 45 k-clip dataset (bias-variance,
    Lecture 7).
- **Cosine annealing** + **5-epoch linear warmup** (lr from 0.01·lr → lr).
  - Warmup prevents the Transformer LayerNorms from blowing up in the first
    few steps when activations are uncalibrated.
- **AMP** (mixed precision) enabled — already wired in `train.py`.

### Regularization stack (deliberate, not piled on)

| Knob                | Value          | Why                                            |
|---------------------|----------------|------------------------------------------------|
| Label smoothing     | 0.1            | Softer targets — known to help with class-imbalanced multi-class classification. |
| MixUp α             | 0.2            | Gentler than the existing 0.4 — clips are short, very strong blends destroy temporal signal. |
| CutMix p            | 0.5            | When triggered, copy a spatial rectangle from another clip across all 4 frames — temporally consistent. |
| RandAugment         | 2 ops, mag 0.5 | Enabled (was off in `default.yaml`). Same op sampled once per clip and applied to every frame. |
| Random Erasing      | p=0.25         | Already in `VideoTransform`. Same erased rectangle across all frames. |
| DropPath            | 0.0 → 0.1      | Stochastic depth on Transformer residual branches, linear schedule. Keeps the network exposed during training. |
| Dropout (head)      | 0.2            | Right before the classifier. |
| Weight decay        | 0.05           | AdamW weight decay. |

### Data-side choices

- **Horizontal flip DISABLED.** Several classes are direction-sensitive
  (`Pulling left` ↔ `Pulling right`, `Pouring into` ↔ `Pouring out of`,
  `Moving up` ↔ `Moving down`, …). `p=0.5` hflip silently inverts the label for
  these clips — a label-corrupting bug, not augmentation. Disabling it removes
  noise the model was being asked to fit through. Gated by
  `training.horizontal_flip: false` in the experiment yaml.
- **WeightedRandomSampler**, weights ∝ 1/√(class_count). The dataset is
  imbalanced ~20× (162 → 3,170 clips/class). Pure inverse-frequency (power=1)
  over-samples the rare classes too aggressively and hurts overall accuracy;
  sqrt-frequency (power=0.5) is the empirically robust middle ground.
- **TSN-style temporal jitter** (existing). Training picks one frame at random
  from each of T uniform temporal segments — every epoch the model sees a
  slightly different temporal slice of every clip.
- **`num_frames: 4`** for both training and inference. Submission CSV is built
  with the same setting via `create_submission.py`.

### Loop / data
- **batch_size 32**, **60 epochs**, **8 workers**, persistent + prefetch_factor=2.
- Internal train/val split: 80/20 over `dataset.train_dir`. The dedicated
  `dataset.val_dir` is only used by `evaluate.py` for final reporting.

### What I deliberately **did not** add
- **EMA / SWA**: the win is small at ~60 epochs and it adds an extra moving
  copy of the weights. Skipped to keep the recipe honest.
- **Early stopping**: with cosine annealing the last few epochs are where the
  best accuracy usually appears — early stopping would discard them.
- **Frame-token attention masking**: with only 4 tokens (5 with CLS) full
  attention is trivially cheap; masking complicates the code without changing
  results.
- **Pretrained backbone init**: forbidden by the closed-track rules.

---

## 5. How to run

```bash
# Training (from src/)
cd src
python train.py experiment=video_former_lite_closed

# Quick smoke test — small subset, 2 epochs
cd src
python train.py experiment=video_former_lite_closed \
    dataset.max_samples=2000 training.epochs=2 training.batch_size=16

# Evaluate the saved checkpoint on the full val_dir
cd src
python evaluate.py training.checkpoint_path=$(pwd)/../models/video_former_lite_closed.pt

# Build the submission CSV
cd src
python create_submission.py training.checkpoint_path=$(pwd)/../models/video_former_lite_closed.pt
```

The checkpoint saved by `train.py` contains the full Hydra config — both
`evaluate.py` and `create_submission.py` rebuild the model from it, so no
additional flags are needed at eval time.

---

## 6. Expected behavior

- First couple of epochs: train accuracy hovers around 5–10 % (33 classes, MixUp
  + CutMix active). This is normal — heavy mixing makes the *raw* accuracy a
  poor metric early on. Watch validation accuracy instead.
- Around epoch 10–15: clear validation signal (~20–25 % top-1 typically) once
  the warmup phase ends and the Transformer attention starts shaping up.
- Final 60-epoch accuracy: target **~40 % top-1** on the internal val split.

If validation accuracy stalls < 15 % after epoch 20, the most likely culprits
are: (a) the LMDB cache being stale (rebuild with `pack_lmdb.py`), (b) a frame
sampling bug on very short clips, or (c) gradient explosion from a stray
`lr` override — check the first epoch's lr printout in the CSV log.

---

# TSM_ultra_50 — Closed Track (scaled-up TSM)

A **ResNet50 + Temporal Shift Module** recipe targeting the ~46 % leaderboard
tier. Designed as the natural scale-up of TSM_ultra: same architecture family,
more spatial capacity, more temporal context, and a longer cyclic LR schedule.

> **Why a new model?** TSM_ultra (ResNet18, 5 frames, 80 epochs) reached
> 53.32 % internal val / 34.46 % leaderboard. The training curve plateaued at
> epoch 68 — the bottleneck was spatial capacity (R18 ≈ 11 M params) and
> temporal context (5 frames is below the literature standard for SSv2). The
> ensemble with VFL added ≈ 3 pp on val. To clear the next tier we need a
> stronger *single* model; TSM_ultra_50 is that model.

---

## 1. Architecture

```
Input  : (B, T=8, C=3, H=224, W=224)

reshape (B*T, 3, 224, 224)
ResNet50 with TemporalShift wrapping every residual block
  - fold_div=8  (1/8 channels shift back, 1/8 shift forward, 6/8 unchanged)
  - drop_path linearly 0.0 → 0.2 over the 16 bottleneck blocks
  - random init (closed track)
GAP                                          # (B*T, 2048)
reshape (B, T, 2048) → mean over T           # (B, 2048)
Dropout(p=0.5)
Linear(2048, 33)                             # logits
```

| Component                        |   Params |
|----------------------------------|---------:|
| ResNet50 backbone (no fc)        |  ~23.5 M |
| TemporalShift (parameter-free)   |       0  |
| Classifier head                  |   ~67 k  |
| **Total**                        | **~23.6 M** |

### Why ResNet50 (vs ResNet18 in TSM_ultra)

- ResNet18 plateaued at 53.32 % internal — extra epochs stopped helping after
  ~68. That's the signature of insufficient capacity, not insufficient training.
- ResNet50's bottleneck blocks (1×1 → 3×3 → 1×1) carry richer per-frame features
  (2048-d vs 512-d after GAP). On SSv2 many class pairs are *visually* similar
  with motion as the only differentiator — better spatial features feed cleaner
  motion signal into the temporal shift.
- TSM remains parameter-free: every extra parameter goes to per-frame spatial
  reasoning, where it is most useful given the closed-track no-pretraining rule.

### Why 8 frames (vs 5 in TSM_ultra)

- The original TSM paper (Lin et al., ICCV 2019) benchmarked SSv2 at 8 and 16
  frames; 5 was a compute compromise, not a principled choice.
- "What happens next?" labels are defined by the *arc* of motion (start →
  intermediate → end). 8 frames gives ~2× the temporal density of 5, which
  helps disambiguate trajectories like "pushing right" vs "pushing-then-stopping".
- With 8 frames the standard `fold_div=8` is sufficient (1/8 each direction
  participates per layer); the aggressive `fold_div=4` from TSM_ultra was a
  compensation for very short clips.

---

## 2. Compute budget

| Metric                          | Estimate                                |
|---------------------------------|-----------------------------------------|
| FLOPs / clip (fwd)              | ~33 GFLOPs (R50 ≈ 4.1 GF × 8 frames)    |
| Params                          | ~23.6 M                                 |
| VRAM @ B=16, AMP                | ~10–12 GB                               |
| Step time (1 GPU, AMP)          | ~0.7–1.0 s / step                       |
| ~35 k samples → ~2200 steps     |                                         |
| Time / epoch                    | ~10–15 min                              |
| **120 epochs**                  | **~12–18 h**                            |

If wall-clock is tight, the cheapest knobs are `batch_size: 12` (slightly less
stable BN but ~25 % less memory) or `epochs: 96` (drops one SGDR cycle —
finishes at epoch 92's local minimum).

---

## 3. Training recipe (`configs/experiment/tsm_ultra_50.yaml`)

### Optimization
- **AdamW**, lr **1 × 10⁻³**, weight_decay **5 × 10⁻⁴**.
  - Same lr and weight decay as TSM_ultra — they were already well-calibrated
    for the TSM family; the architecture change does not justify retuning.
- **SGDR (cosine warm restarts)** with `T_0 = 28`, `T_mult = 1`.
  - 4 equal cycles of 28 epochs after the 8-epoch linear warmup:
    `8 + 4 × 28 = 120 epochs`.
  - Restarts at epochs **36, 64, 92** — the LR jumps back to its peak then
    anneals to near-zero before the next restart. Each cycle ends in a
    near-zero LR (a "save point") that often outperforms a single long cosine.
  - Why SGDR vs monotonic cosine: with 120 epochs of a heavy from-scratch
    model, monotonic cosine can converge to a sharp local minimum. SGDR's
    restarts let the optimizer escape and explore wider basins (Loshchilov &
    Hutter, ICLR 2017).
- **Linear warmup**, 8 epochs (lr 0.01·lr → lr) before SGDR kicks in.
  - Prevents BN statistics from being shaped by tiny minibatches at full LR
    on a randomly initialised network.
- **AMP** enabled — keeps R50 + 8 frames at batch 16 within the 16–20 GB
  budget on a single GPU.

### Regularization stack

| Knob                | Value          | Why                                            |
|---------------------|----------------|------------------------------------------------|
| reg_warmup_epochs   | **10**         | MixUp/CutMix disabled epochs 0–9. Non-negotiable; the +36 pp lesson from VFL applies to every from-scratch model. |
| Label smoothing     | **0.15**       | Bumped from 0.1 — SSv2 has many close class neighbours; softer targets help calibration. |
| MixUp α             | 0.2            | Gentle blend (same as TSM_ultra). |
| CutMix p            | 0.25           | Reduced from 0.5 — patch erasure across all 8 frames destroys motion signal on SSv2. |
| RandAugment         | 2 ops, mag 0.5 | Temporally consistent (sampled once per clip). |
| Random Erasing      | p=0.3          | Same erased rectangle across frames. |
| **DropPath**        | **0.2**        | Increased from 0.1 — R50 has 2× the parameters of R18, proportionally more depth regularization. |
| **Dropout (head)**  | **0.5**        | Increased from 0.3 — wider 2048-d feature vector needs more output regularization. |
| Weight decay        | 5 × 10⁻⁴       | AdamW weight decay (same as TSM_ultra). |

### Data-side choices (same as TSM_ultra)
- **Horizontal flip DISABLED** — direction-sensitive SSv2 labels.
- **WeightedRandomSampler**, power 0.5 — soft inverse-sqrt rebalancing for
  the 20× class imbalance.
- **TSN-style temporal jitter** — one frame per segment, sampled randomly
  during training.
- **`num_frames: 8`** at train and inference time. The submission script
  reads `num_frames` from the checkpoint, so no flag is needed at inference.

### What is deliberately NOT in this recipe
- **Attention temporal pooling**: would replace `mean(T)` with a learned
  weighted sum. Likely +0.5–1 pp but adds a new code path. Keep the recipe
  pure for now; revisit only if TSM_ultra_50 stagnates below target.
- **Optical flow**: +3–8 pp but requires a separate preprocessing pipeline
  and a second model. Defer to a later experiment.
- **Multi-clip inference**: sampling 3–5 temporal windows per video at test
  time is a known SSv2 trick (+1–2 pp). Inference-only, can be bolted on the
  ensemble script without re-training — promote to Tier 1 after TSM_ultra_50.

---

## 4. Lecture cross-references (additions over TSM_ultra)

| Idea                                   | Lecture / slide                            |
|----------------------------------------|--------------------------------------------|
| SGDR / cosine warm restarts            | Lecture 4 — optimization (learning rate schedules) |
| Bottleneck residual blocks (ResNet50)  | Lecture 5 — modern CNNs                    |
| Capacity vs regularization trade-off   | Lecture 7 — bias-variance                  |

---

## 5. How to run

```bash
cd src

# Full training (~12–18 h)
uv run python train.py experiment=tsm_ultra_50

# Smoke test — small subset, 2 epochs, smaller batch
uv run python train.py experiment=tsm_ultra_50 \
    dataset.max_samples=2000 training.epochs=2 training.batch_size=8

# Evaluate the saved checkpoint on the official val_dir
uv run python evaluate.py training.checkpoint_path=$(pwd)/../models/tsm_ultra_50.pt
uv run python evaluate.py training.checkpoint_path=$(pwd)/../models/tsm_ultra_50.pt dataset.tta=true

# Build the submission CSV (always submit with TTA)
uv run python create_submission.py \
    training.checkpoint_path=$(pwd)/../models/tsm_ultra_50.pt \
    dataset.tta=true \
    dataset.submission_output=../submissions/tsm_ultra_50_tta.csv

# Ensemble with TSM_ultra and VFL
uv run python create_ensemble_submission.py \
    "+checkpoints=[../models/tsm_ultra_50.pt,../models/tsm_ultra.pt,../models/video_former_lite_closed.pt]" \
    dataset.submission_output=../submissions/ensemble_v2_tta.csv \
    dataset.tta=true \
    training.batch_size=4
```

The checkpoint includes the full Hydra config, so `evaluate.py`,
`create_submission.py`, and `create_ensemble_submission.py` rebuild the model
without architecture flags.

---

## 6. Expected behaviour

- **Epochs 1–10 (warmup, no MixUp/CutMix):** loss falls ~3.4 → 2.7;
  val climbs to ~25–30 %. R50 from scratch is slower to take off than R18
  for the first few epochs (more parameters to settle).
- **Epoch 10 (regularization kicks in):** train accuracy dips, val keeps
  climbing. Same shape as TSM_ultra.
- **Epochs 11–36 (first SGDR cycle):** rapid gain; expect ~45–50 % val by
  epoch 36 when the LR hits the bottom of the first cosine.
- **Epoch 36 (first restart):** LR jumps back to peak. Loss may spike briefly
  before annealing again. Val accuracy should re-stabilise within 2–3 epochs.
- **Subsequent cycles (37–120):** each cycle adds 1–3 pp at its trough.
  Final target: **57–60 % internal val** (a +5–7 pp lift over TSM_ultra).
  Translating to the leaderboard via the same gap factor as TSM_ultra
  (≈ 0.65) gives a 37–39 % LB single-model estimate; combined with
  TTA + ensemble, the **46 % leaderboard tier becomes plausible**.

**Red flags during training:**
- Val accuracy diverges down after a restart and stays down → SGDR cycle too
  long, model is leaving a good minimum without recovering. Drop `T_0` to 20.
- Train accuracy matches val within 2 pp → not enough regularization,
  bump `drop_path_rate` to 0.3.
- Train accuracy 30+ pp above val → too much regularization; drop dropout
  to 0.3 and/or reduce `cutmix_prob` to 0.15.

