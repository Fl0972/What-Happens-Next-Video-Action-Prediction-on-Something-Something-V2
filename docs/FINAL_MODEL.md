# Final Model — Action Recognition on "What Happens Next?"

**Track:** Closed World (no ImageNet pretraining)
**Dataset:** 33-class subset of Something-Something V2 — labels are *motion-defined* over 4 JPEG frames per clip.
**Final approach:** Weighted late-fusion ensemble of 3 complementary architectures + 10-crop Test-Time Augmentation.

---

## 1. Iterative Path to the Final Model

| Step | Model | Key idea | Internal val acc |
|------|-------|----------|------------------|
| 1 | **CNN Baseline** | ResNet18 + temporal average pool | low baseline |
| 2 | **CNN-LSTM** | Frame features → LSTM | marginal gain |
| 3 | **VideoFormer-Lite** (VFL) | ResNet18 features → 2-layer Transformer over frame tokens with `[CLS]` | 45.78% (60 ep.) |
| 4 | **TSM-ResNet** | Temporal Shift Module — zero-parameter temporal mixing inside every ResNet block | strong baseline |
| 5 | **`tsm_ultra`** | TSM + 5 frames, `fold_div=4`, full reg recipe | **53.32%** |
| 6 | **`video_former_lite_ultra`** | VFL + deeper transformer, lr tuned | **53.37%** |
| 7 | **`tsm_ultra_v2`** | TSM with **4 frames** (matches the true clip length) + **focal loss** + 100 epochs | **57.73%** |
| 8 | **Final Ensemble + TTA** | Weighted late fusion of the 3 best + 10-crop TTA | *Best submission* |

> **Key insight (step 7):** Every clip contains exactly 4 frames. Using `num_frames > 4` duplicates frames via interpolation, corrupting the temporal-shift signal. Setting `num_frames = 4` unlocked a **+4.4 pp** jump.

---

## 2. The Three Building Blocks of the Final Model

### Block A — `tsm_ultra_v2` (TSM ResNet18, 4 frames, focal loss) — **57.73%**

- **Architecture:** ResNet18 backbone where every residual block is wrapped with `TemporalShift` — shifts 1/4 of channels backward and 1/4 forward in time (`fold_div=4`). Adds **zero parameters**.
- **Why it wins:** SSv2 labels are motion-defined (e.g. *"Pouring X into Y"*). TSM was specifically designed for this dataset family.
- **Loss:** Focal loss (`γ=2`) on top of label-smoothed CE → concentrates gradient on hard within-group pairs (e.g. *"Poking so it falls"* vs *"Poking so it slightly moves"*).
- **Schedule:** AdamW, 100 epochs, cosine annealing with 8-epoch linear warmup.

### Block B — `tsm_ultra` (TSM ResNet18, 5 frames) — **53.32%**

- Same architecture as Block A but **5 frames** and **standard CE**.
- **Role in the ensemble:** error decorrelation w.r.t. Block A — different temporal sampling resolution, different loss landscape.

### Block C — `video_former_lite_ultra` (CNN + Transformer) — **53.37%**

- **Architecture:** Per-frame ResNet18 → `[CLS]` token + learnable temporal positional embedding → 2 pre-LN Transformer encoder blocks (MHSA + MLP) → classifier on `[CLS]`.
- **Why include it:** Radically different inductive bias from TSM — *global* temporal attention rather than *local* shifts. Decorrelates errors maximally.

---

## 3. Regularization Recipe (shared across all blocks)

All three models share the same regularization stack — empirically the highest-impact knob in the experiment series.

| Technique | Setting | Rationale |
|-----------|---------|-----------|
| **Regularization warm-up** | MixUp / CutMix **disabled for first 10 epochs** | Cold-start backbone must learn basic per-frame features before label-mixing kicks in. Single biggest fix in the project (**+36 pp** on VFL: 9.45% → 45.78%). |
| **Label smoothing** | 0.1 | Softens targets — many SSv2 classes are visually adjacent. |
| **MixUp** | α = 0.2 | Convex combinations of (input, label) pairs. |
| **CutMix** | p = 0.25 (low) | A cut patch crosses all 4 frames and frequently deletes the hand/object motion that *is* the label → kept low. |
| **RandAugment** | 2 ops @ magnitude 0.5 | Temporally consistent (same params across all frames of a clip). |
| **Random Erasing** | p = 0.25 | Occlusion robustness. |
| **DropPath (stochastic depth)** | 0.1–0.15 | Regularizes deeper residual branches. |
| **Dropout** | 0.2–0.3 before classifier | Standard head regularization. |
| **Weight decay** | 5e-4 (TSM) / 5e-2 (VFL) | AdamW decoupled decay. |
| **Horizontal flip** | **DISABLED** | Direction-sensitive classes (*"pulling left/right"*) → hflip silently corrupts labels. |
| **Weighted sampling** | `1/sqrt(class_count)` | Soft rebalancing of the 20× class imbalance (162 → 3170 clips/class). |
| **TSN-style temporal jitter** | training only | Random frame per segment → temporal augmentation diversity. |

---

## 4. Final Inference Pipeline — Weighted Ensemble + TTA

### 4.1 Late-Fusion Ensemble

Each model runs its own forward pass with its own `num_frames` and preprocessing. Logits are averaged with weights proportional to validation accuracy.

```text
P_final(class) = softmax( 0.50·logits_A + 0.25·logits_B + 0.25·logits_C )
```

| Block | Weight | Why this weight |
|-------|--------|------------------|
| `tsm_ultra_v2` (A) | **0.50** | Clearly dominant on val (+4.4 pp) |
| `tsm_ultra` (B) | **0.25** | Complementary frame count |
| `video_former_lite_ultra` (C) | **0.25** | Architectural diversity |

`tsm_ultra_50` (ResNet50, 8 frames, 45.26%) and `vfl_ultra_focal` (53.17%) are **excluded** — they would drag the ensemble down or duplicate the signal of stronger models.

### 4.2 Test-Time Augmentation (10-crop TTA)

For each test clip, run inference on **10 augmented views**:
- **5 spatial crops:** center + 4 corners
- **× 2 flips:** original + horizontal flip (re-enabled only at test time, where label semantics no longer matter for the ensemble vote)

Per-model logits are averaged across the 10 views before fusion. Typical gain: **+0.5 to +2 pp**.

### 4.3 Reproducing the Final Submission

```bash
cd src
python create_ensemble_submission.py \
  "+checkpoints=[../models/tsm_ultra_v2.pt,../models/tsm_ultra.pt,../models/video_former_lite_ultra.pt]" \
  "+weights=[0.5,0.25,0.25]" \
  "dataset.tta=true" \
  "dataset.submission_output=../submissions/ensemble_final.csv"
```

---

## 5. Lessons Learned

1. **Match `num_frames` to the actual clip length.** Duplicating frames silently corrupts temporal reasoning — the single largest accuracy unlock.
2. **Regularization warm-up is non-negotiable on cold-start models.** Strong label mixing before basic features are learned destroys training.
3. **Domain-aware augmentation choices matter more than recipe transfer.** Disabling horizontal flip and keeping CutMix low was specific to SSv2 motion semantics.
4. **Ensemble diversity beats raw capacity.** Two complementary architectures at 53% beat one stronger ResNet50 at 45%.
5. **Verify the data before scaling the model.** `tsm_ultra_50` (ResNet50, 8 frames) underperformed `tsm_ultra_v2` (ResNet18, 4 frames) — scaling capacity without fixing the frame-count bug was wasted compute.
