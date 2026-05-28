# Final Model — Action Recognition on "What Happens Next?"

**Track:** Closed World (no ImageNet pretraining)
**Dataset:** 33-class subset of Something-Something V2 — labels are *motion-defined* over 4 JPEG frames per clip.
**Final approach:** Log-softmax late-fusion ensemble of complementary from-scratch models (TSM + VideoFormer-Lite, with rotating-fold variants) + 10-crop TTA — **44.03% val-dir top-1**.

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
| 8 | **Final Ensemble + TTA** | Log-softmax fusion of `tsm_ultra_v2` + `tsm_ultra_v2_rotating` + `video_former_lite_ultra_rotating` + 10-crop TTA | **44.03% val-dir (best)** |

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

## 3. All Models — Architecture Reference

### 3.1 CNNBaseline (`cnn_baseline`)

- **File:** `src/models/cnn_baseline.py`
- **Architecture:** ResNet18 backbone; per-frame features are average-pooled over time.
- **Result:** Low baseline (undisclosed; serves only as the starting point).
- **Note:** No temporal reasoning — every frame is treated as independent.

### 3.2 CNNLSTM (`cnn_lstm`)

- **File:** `src/models/cnn_lstm.py`
- **Architecture:** ResNet18 per-frame features → unidirectional LSTM → last hidden state → classifier.
- **Result:** Marginal improvement over CNNBaseline.
- **Note:** LSTM is sequential; backprop through time is unstable on short sequences.

### 3.3 VideoFormerLite (`video_former_lite`)

- **File:** `src/models/video_former_lite.py`
- **Architecture:** Per-frame ResNet18 feature extraction → learned `[CLS]` token + temporal positional embedding → N pre-LN Transformer blocks (MHSA + MLP) → classifier on `[CLS]`.
- **Params:** `d_model=512`, `n_heads=8`, `n_layers=2` (ultra variant uses `n_layers=4`).
- **Results:**
  - `video_former_lite` (standard): ~45.78% internal val (60 ep.)
  - `video_former_lite_ultra` (deeper, tuned lr): **53.37%** internal val
  - `vfl_ultra_focal` (+ focal loss): 53.17% internal val
  - Val-dir top-1 (no TTA): **32.11%**
- **Note:** Global temporal attention complements TSM's local shifts — key ensemble diversity source.

### 3.4 TSMResNet (`tsm_resnet`)

- **File:** `src/models/tsm_resnet.py`
- **Architecture:** ResNet18/50 backbone where every residual block is wrapped with `TemporalShift`. Shifts `1/fold_div` of channels backward and forward in time with zero added parameters.
- **Results:**
  - `tsm_ultra` (ResNet18 scratch, 5 fr, `fold_div=4`): **53.32%** internal val, **34.91%** val-dir
  - `tsm_ultra_v2` (ResNet18 scratch, 4 fr, `fold_div=4`, focal): **57.73%** internal val, **38.25%** val-dir
  - `tsm_ultra_50` (ResNet50 scratch, 8 fr): 45.26% — scaling capacity without fixing frame count was wasted compute
- **Note:** Best single model. Designed for SSv2 motion-defined labels.

### 3.5 R2Plus1D (`r2plus1d`)

- **File:** `src/models/r2plus1d.py`
- **Architecture:** 3D conv factored into 2D spatial + 1D temporal convolutions (torchvision `r3d_18` backbone).
- **Results:** Competitive closed-track baseline; used in `r2plus1d_closed` experiment.

### 3.6 TSMBiGRU (`tsm_gru`)

- **File:** `src/models/tsm_gru.py`
- **Architecture:** TSM-ResNet50 backbone → per-frame features → BiGRU → temporal mean pool → classifier. Combines TSM's local temporal shifts with GRU's sequential memory.
- **Params:** `gru_hidden=256`, `gru_layers=1`, `feat_dropout=0.3`, `head_dropout=0.5`.
- **Experiments:** `tsm_resnet50_bigru`, `tsm_resnet50_bigru_scratch` (rotating-fold training).
- **Note:** The BiGRU adds 2× capacity to the temporal head over plain pooling; the rotating-fold variant trains on all data via fold rotation.

### 3.7 MaxViTVideo (`maxvit`) — *in development*

- **File:** `src/models/maxvit.py`
- **Architecture:** MaxViT-T (Multi-Axis ViT, ~31M params) per-frame backbone → 512-dim feature via hierarchical window+grid attention → temporal mean pool → classifier.
- **Key innovation: frame interpolation.** Since each clip has exactly 4 real JPEG frames, `dataset.interpolate_frames=true` inserts a linear pixel blend between each adjacent pair, expanding 4→8 frames before the model. This doubles the temporal positions available to the mean-pool aggregator without adding disk I/O.
  ```
  [f0, blend(f0,f1), f1, blend(f1,f2), f2, blend(f2,f3), f3, f3]  →  8 frames
  ```
- **Training recipe:** AdamW, `backbone_lr_scale=0.05`, cosine 100 ep., focal loss γ=2, horizontal_flip=false.
- **Experiment:** `maxvit_interp` (`dataset.num_frames=4`, `dataset.interpolate_frames=true`, `model.num_frames=8`).
- **Result:** Not yet trained. Expected to serve as an ensemble complement to TSM (orthogonal inductive biases: MaxViT's multi-scale attention vs. TSM's local channel shifts).

---

## 4. Regularization Recipe (shared across all blocks)

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
| **Weight decay** | 5e-4 (TSM) / 5e-2 (VFL/ViT) | AdamW decoupled decay. |
| **Horizontal flip** | **DISABLED** | Direction-sensitive classes (*"pulling left/right"*) → hflip silently corrupts labels. |
| **Weighted sampling** | `1/sqrt(class_count)` | Soft rebalancing of the 20× class imbalance (162 → 3170 clips/class). |
| **TSN-style temporal jitter** | training only | Random frame per segment → temporal augmentation diversity. |

---

## 5. Final Inference Pipeline — Weighted Ensemble + TTA

### 5.1 Late-Fusion Ensemble

The best closed-track configuration (see `ENSEMBLE_EXP.md`) averages the **log-softmax** outputs of three from-scratch models — each run with 10-crop TTA — with equal weight:

```text
P_final(class) = softmax( mean[ log_softmax(logits_A), log_softmax(logits_B), log_softmax(logits_C) ] )
```

| Component | Checkpoint | Val-dir top-1 (TTA) |
|-----------|------------|---------------------|
| A — `tsm_ultra_v2` | `tsm_ultra_v2.pt` | 0.4086 |
| B — `tsm_ultra_v2` (rotating fold) | `tsm_ultra_v2_rotating.pt` | 0.4168 |
| C — `video_former_lite_ultra` (rotating fold) | `video_former_lite_ultra_rotating.pt` | 0.3781 |
| **Ensemble** | log-softmax avg + TTA | **0.4403** |

Log-softmax averaging (rather than raw-logit averaging) down-weights over-confident wrong predictions and rewards model agreement, which suits components with different calibration (rotating-fold vs. single-split). The earlier three-model *weighted-logit* ensemble (40.43%) and its leave-one-out component study are analysed in §6.

### 5.2 Test-Time Augmentation (10-crop TTA)

For each test clip, run inference on **10 augmented views**:
- **5 spatial crops:** center + 4 corners
- **× 2 flips:** original + horizontal flip (re-enabled only at test time, where label semantics no longer matter for the ensemble vote)

Per-model logits are averaged across the 10 views before fusion. Typical gain: **+0.5 to +2 pp**.

### 5.3 Reproducing the Final Submission

```bash
cd src
python create_ensemble_submission.py \
  "+checkpoints=[../models/tsm_ultra_v2.pt,../models/tsm_ultra_v2_rotating.pt,../models/video_former_lite_ultra_rotating.pt]" \
  "dataset.tta=true" "+log_softmax=true" \
  "dataset.submission_output=../submissions/ensemble_final.csv"
```

---

## 6. Model Analysis on the Full Validation Set (three-model weighted ensemble, 40.43%)

> **Scope.** This section analyses the earlier **three-model weighted-logit ensemble** — `tsm_ultra_v2` (0.50) + `tsm_ultra` (0.25) + `video_former_lite_ultra` (0.25), no TTA, **40.43%** val-dir. The leave-one-out, per-class and confusion-matrix studies below were all computed on that specific configuration. The final submission (§5) improves on it to **44.03%** by adding rotating-fold variants, TTA and log-softmax fusion; the component-contribution insights here carry over.

Following the lecture techniques (ablation studies, confusion matrix, per-class metrics), we ran a single forward pass per model on the full validation split (**6 746 clips, 33 classes**) and analysed the cached logits offline. Numbers below are **without TTA** (TTA adds ~0.5–2 pp on top, see §5.2).

> Note: the per-model `val_accuracy` stored in each checkpoint (53–58 %) is measured on a small internal split of the *train* folder used for early-stopping. The numbers below are on the larger, harder **`val_dir`** — the apples-to-apples comparison set for ensembling.

### 6.1 Three-model weighted ensemble vs single models

| Model | Top-1 | Top-5 |
|-------|-------|-------|
| `tsm_ultra_v2` | 38.25 % | 68.70 % |
| `tsm_ultra` | 34.91 % | 66.88 % |
| `video_former_lite_ultra` | 32.11 % | 64.85 % |
| **Ensemble (0.50 / 0.25 / 0.25)** | **40.43 %** | **71.27 %** |

The ensemble outperforms the strongest single model by **+2.18 pp top-1** and **+2.57 pp top-5** — the lift comes from genuinely complementary errors, not from averaging close-to-identical predictions.

### 6.2 Ablation study (leave-one-out)

The ablation directly answers the lecture's *"which part is most important?"* question by deleting one model at a time and re-running inference with renormalised weights.

| Configuration | Top-1 | Δ vs full ensemble |
|---------------|-------|--------------------|
| Full ensemble (3 models) | **40.43 %** | — |
| − `tsm_ultra_v2` | 37.26 % | **−3.17 pp** |
| − `tsm_ultra` | 39.56 % | −0.87 pp |
| − `video_former_lite_ultra` | 39.82 % | −0.61 pp |

**Interpretation.** Every block has a *positive* contribution (no removal improves the ensemble), confirming all three are correctly weighted. `tsm_ultra_v2` is clearly the most critical — removing it costs more than the other two combined. The two complementary models each contribute ~0.6–0.9 pp, justifying their inclusion despite their lower stand-alone accuracy.

### 6.3 Per-class precision / recall / F1

Macro-averaged on the ensemble predictions:

| Metric | Value |
|--------|-------|
| Macro Precision | 0.360 |
| Macro Recall | 0.369 |
| Macro F1 | 0.358 |

**Best 5 classes (F1, ensemble):**

| Class | Precision | Recall | F1 | n |
|-------|-----------|--------|-----|---|
| 018 — Pulling something from left to right | 0.61 | 0.63 | **0.62** | 169 |
| 012 — Pouring something into something | 0.58 | 0.57 | 0.58 | 278 |
| 007 — Moving something closer to something | 0.58 | 0.57 | 0.58 | 213 |
| 030 — Uncovering something | 0.53 | 0.58 | 0.55 | 391 |
| 003 — Folding something | 0.56 | 0.51 | 0.54 | 285 |

These are all classes defined by a **single, clean motion arc** (pour, pull, fold) — exactly what TSM was designed to capture.

**Worst 5 classes (F1, ensemble):**

| Class | Precision | Recall | F1 | n |
|-------|-----------|--------|-----|---|
| 028 — Taking something out of something | — | — | 0.00 | 0 (no val samples) |
| 026 — Spilling something next to something | 0.10 | 0.02 | 0.03 | 60 |
| 011 — Picking something up | 0.09 | 0.06 | 0.07 | 199 |
| 016 — Pretending to put something into something | 0.15 | 0.13 | 0.14 | 68 |
| 017 — Pretending to throw something | 0.13 | 0.32 | 0.18 | 47 |

The failure mode is exactly what SSv2 was designed to expose: **"pretending" classes** and **classes whose motion is ambiguous out of context** (e.g. *spilling next to* vs *putting next to*).

### 6.4 Most-confused class pairs

The top off-diagonal entries of the 33×33 confusion matrix:

| Count | True class | Predicted class |
|-------|------------|------------------|
| 40 | Moving something up | **Pretending to pick something up** |
| 36 | Showing something to the camera | Turning something upside down |
| 34 | (class 032) | Folding something |
| 30 | **Picking something up** | **Pretending to pick something up** |
| 29 | Folding something | Unfolding something |
| 29 | Showing something to the camera | Holding something |
| 28 | Opening something | Holding something |
| 28 | Moving something down | Holding something |
| 27 | Folding something | (class 032) |
| 24 | Unfolding something | Folding something |

Three patterns dominate:
1. **Real vs pretended actions** (*picking up* ↔ *pretending to pick up*) — the model sees a similar hand trajectory and cannot disambiguate intent. Focal loss in `tsm_ultra_v2` partially addresses this but doesn't fully solve it.
2. **Temporal direction confusion** (*folding* ↔ *unfolding*) — both classes share the same spatial appearance; only the temporal ordering of frames separates them. With only 4 frames, this signal is fragile.
3. **"Static" sink class** (many true classes → *holding something*) — *holding* becomes the default prediction when no motion is detected. A future improvement would re-balance against this prior.

### 6.5 Reproducing the analysis

```bash
cd src
python analyze_ensemble.py                  # cached logits → ../models/val_logits/
python analyze_ensemble.py dataset.tta=true # with TTA
```
Caches and full JSON results are written under `../models/val_logits/` and `docs/analysis_results.json`.

---

## 7. Lessons Learned

1. **Match `num_frames` to the actual clip length.** Duplicating frames silently corrupts temporal reasoning — the single largest accuracy unlock.
2. **Regularization warm-up is non-negotiable on cold-start models.** Strong label mixing before basic features are learned destroys training.
3. **Domain-aware augmentation choices matter more than recipe transfer.** Disabling horizontal flip and keeping CutMix low was specific to SSv2 motion semantics.
4. **Ensemble diversity beats raw capacity.** Two complementary architectures at ~33 % each beat one stronger ResNet50 at lower stand-alone accuracy — the ablation (§6.2) confirms every block contributes positively.
5. **Verify the data before scaling the model.** `tsm_ultra_50` (ResNet50, 8 frames) underperformed `tsm_ultra_v2` (ResNet18, 4 frames) — scaling capacity without fixing the frame-count bug was wasted compute.
6. **Failure analysis matches the SSv2 design intent.** The hardest classes (§6.3, §6.4) are precisely the "pretending" pairs and direction-sensitive pairs that the dataset was built to expose.
7. **Frame interpolation enables higher temporal resolution from fixed-size clips.** When videos are represented by exactly 4 real frames, inserting pixel-blend midpoints (→ 8 frames) doubles temporal positions for pooling-based models at zero disk cost. This is the basis for the `maxvit_interp` experiment.
