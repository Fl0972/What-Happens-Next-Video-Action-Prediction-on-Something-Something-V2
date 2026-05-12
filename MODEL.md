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
