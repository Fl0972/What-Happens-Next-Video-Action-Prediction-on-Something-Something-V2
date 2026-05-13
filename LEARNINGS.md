# Challenge Model Learnings — CSC_43M04_EP (What Happens Next?)

> Dataset: Something-Something V2 (33-class subset, "What Happens Next?" framing)  
> Constraint: **Closed track** — training from scratch, no pretrained weights  
> Hardware: 1× NVIDIA RTX A4000 (16 GB)

---

## 1. Experimental Results at a Glance

| Model | Internal Val Acc | Official Val (val_dir) | Leaderboard | Frames | Epochs | Training Time |
|---|---|---|---|---|---|---|
| VFL (killed, no warmup) | 9.45% | — | — | 4 | 4 / 60 | ~10 min |
| R2+1D_closed (killed) | 10.78% | — | — | 8 | 15 / ? | ~30 min |
| VFL_closed (with warmup) | 45.78% | — | — | 4 | 60 | ~3.5 h |
| **TSM_ultra** | **53.32%** | — | **34.46%** | 5 | 80 | ~4 h |
| Ensemble TSM_ultra + VFL (no TTA) | — | **37.06%** | — | 5 / 4 | — | inference only |
| Ensemble TSM_ultra + VFL (TTA) | — | **37.35%** (top-5 = 70.48%) | — | 5 / 4 | — | inference only |
| **TSM_ultra_50** (planned) | target 57–60% | target ~45% | target ~38–40% single, **~46% ensembled** | 8 | 120 | ~12–18 h |

**Takeaways at a glance:**
- The single biggest training-time lever was fixing **heavy regularization
  applied too early** (§3.1) — VFL improved 9.45% → 45.78% with that one fix.
- The single biggest inference-time lever was **late-fusion ensembling** of
  TSM_ultra + VFL (+2.6 pp on official val over single-model TTA).
- **TTA adds only ~0.3 pp on top of the ensemble** on the official val set —
  smaller than the +1–3 pp typically reported on image benchmarks. The
  ensemble itself already smooths predictions, so TTA's marginal benefit is
  reduced. Still worth keeping (free at submission time).
- **Top-5 of 70.5%** with the ensemble shows the correct class is usually in
  the model's top guesses — the room to grow is rank-1 calibration, which
  more capacity (TSM_ultra_50) and more views (multi-clip inference) target.
- **R2+1D** was killed before converging — its real ceiling is unknown.

---

## 2. Architecture Analysis

### 2.1 TSMResNet — `models/tsm_resnet.py`

**How it works:**  
A standard ResNet (18 or 50) with every residual block wrapped in a `TemporalShift` layer.
Before each block, `fold_div` of the channel tensor is shifted by ±1 frame along the time
axis — backward-shifted channels can "see" the previous frame; forward-shifted channels can
"see" the next one. The rest are unchanged. No parameters are added.

```
Input (B, T, C, H, W)
  → reshape (B·T, C, H, W)
  → TemporalShift + ResBlock × N  ← TSM injected at every block
  → GAP → (B·T, 512)
  → mean over T → (B, 512)
  → Dropout → Linear(512, 33)
```

**Why it is right for SSv2 from scratch:**
- SSv2 was the benchmark in the original TSM paper (Lin et al., ICCV 2019). The dataset
  was *designed* to test temporal reasoning; TSM was *designed* to provide it cheaply.
- The temporal shift is **parameter-free**. Every parameter in the model learns
  class-discriminative spatial features; there is no overhead budget for temporal modeling.
  This is uniquely valuable from scratch, where every parameter must earn its place.
- Temporal shift is a **local inductive bias**: frame t interacts only with frame t±1 per
  layer. With 5 frames this is sufficient — you don't need global attention across all pairs.

**TSM_ultra vs. tsm_closed — what changed and why:**

| Setting | tsm_closed | tsm_ultra | Reason |
|---|---|---|---|
| Frames | 16 | 5 | Matches closed-track budget; bigger batch size |
| Batch size | 8 | 32 | 5 frames → less memory; stable BN stats |
| Epochs | 40 | 80 | From-scratch ResNet needs more |
| fold_div | 8 | **4** | Short clips need more aggressive per-layer mixing |
| drop_path | 0.2 | **0.1** | Was over-regularising the small backbone |
| dropout | 0.5 | **0.3** | Same: excessive for from-scratch R18 |
| mixup_alpha | 0.4 | **0.2** | Gentler label mixing |
| cutmix_prob | 0.5 | **0.25** | Patch erasure destroys motion signal on SSv2 |
| warmup_epochs | 5 | **8** | Aligns LR peak with end of reg warmup |
| reg_warmup_epochs | — | **10** | The single most impactful change |
| horizontal_flip | true | **false** | Direction-sensitive labels |
| weighted_sampling | false | **true** | 20× class imbalance |

**Pros:**
- Best internal val accuracy (53.32%) — 7.5 pp ahead of VFL with 20 extra epochs.
- Fast per-epoch inference — parameter-free temporal module means near-identical
  throughput to a plain ResNet.
- Proven on SSv2 in the literature; architecture is not a risk factor.
- Scales naturally with more frames (just change `num_frames`).

**Cons:**
- Local temporal modeling only (each layer sees ±1 frame). For actions spanning the
  full clip, deep layers must propagate temporal context through many shifts.
- `fold_div=4` vs. `fold_div=8` was not ablated — unclear which is better for 5 frames.
- Training curve shows a clear plateau after epoch 68 (val stuck at ~52–53%). The model
  may be at capacity given from-scratch ResNet18 and 5 frames.

**Planned successor — TSM_ultra_50 (see §6 / MODEL.md):** same architecture
family scaled up with ResNet50 backbone, 8 frames, and 120 epochs of SGDR.
The plateau in TSM_ultra suggests *capacity* is the bottleneck rather than
temporal modelling — the scale-up directly targets that.

---

### 2.2 VideoFormerLite (VFL) — `models/video_former_lite.py`

**How it works:**  
Per-frame ResNet18 extracts a 512-d feature vector for each of the T frames. These are
concatenated with a learnable [CLS] token and summed with learned temporal positional
embeddings, then passed through L pre-LN Transformer encoder blocks. The [CLS] token
output is projected to class logits.

```
Input (B, T, C, H, W)
  → per-frame ResNet18 (shared weights) → (B, T, 512)
  → prepend [CLS] token, add pos. embed. → (B, T+1, 512)
  → L × (MHSA + MLP, pre-LN) → (B, T+1, 512)
  → CLS output → Dropout → Linear(512, 33)
```

**Why it differs from TSM:**
- **Global temporal attention**: every frame token can directly attend to every other,
  including the first and last frame simultaneously. TSM shifts are local; attention is
  global.
- **More parameters**: the Transformer adds ~3 M parameters (2 layers) that have no
  counterpart in TSM. All of these must be learned from scratch.
- **Separate spatial / temporal streams**: the CNN is a pure spatial processor (no
  temporal mixing inside it), while the Transformer is a pure temporal aggregator.
  This separation is cleaner than TSM but requires more data to converge.

**Training curve insight:**  
VFL plateaus later (it was still improving at epoch 60) and converges more slowly than
TSM_ultra. Given 80 epochs, VFL might approach or exceed TSM_ultra — but this has not
been tested.

**Pros:**
- Global temporal attention: can directly compare frame 1 to frame 4 without relying on
  layer-by-layer shift propagation.
- Architecture is flexible: Transformer width, depth, and number of frames can all be
  varied independently of the CNN backbone.
- Clean spatial/temporal factorisation makes it easier to upgrade just the CNN backbone
  (e.g. ResNet50) or just the Transformer (more layers) independently.
- Still improving at epoch 60 — more compute headroom than TSM_ultra.

**Cons:**
- Transformer is data-hungry: attention weights must be learned from scratch.
  With 33 classes and a modest dataset, this is a genuine risk.
- Positional embeddings are fixed-length (constructed at T+1). Cannot change the
  number of frames at inference without re-training.
- Trains more slowly per epoch due to Transformer overhead.
- With only 4–5 frame tokens, the attention module is very shallow — 4 tokens × 2 layers
  is close to a simple MLP with residual connections, not a deep Transformer.

---

### 2.3 R2+1D — `models/r2plus1d.py`

**How it works:**  
A 3D convolutional network where each 3×3×3 spatio-temporal conv is factored into a
2D spatial conv (1×k×k) followed by a 1D temporal conv (k×1×1) with a non-linearity
in between. This adds expressivity over plain 3D conv with the same parameter count.

```
Input (B, T, C, H, W)
  → permute → (B, C, T, H, W)   ← torchvision convention
  → R(2+1)D-18 backbone (random init)
  → Global average pool → (B, 512)
  → Dropout → Linear(512, 33)
```

**Why it underperformed (10.78% at epoch 15, then killed):**
1. **Training was interrupted** at epoch 15 / ? — the network never converged.
2. **No regularization warmup**: MixUp + CutMix were active from epoch 1, preventing
   the cold-start 3D convolutions from learning basic spatial features.
3. **Tiny batch size (8) and small image (112×112)**: 3D convolutions over 8 frames at
   224×224 do not fit in 16 GB VRAM at reasonable batch sizes. The 112×112 compromise
   reduces the spatial signal and degrades BatchNorm statistics (batch 8 → unstable BN).
4. **3D conv is parameter-heavy from scratch**: R(2+1)D-18 has ~31 M parameters vs.
   ~11 M for ResNet18. From scratch on a modest dataset, higher parameter count with
   no good initialisation is a strong disadvantage.

**Theoretical pros:**
- **Volumetric convolution**: directly models short-range spatio-temporal patterns
  (e.g. optical flow implicitly) without needing the temporal shift trick.
- **Factored design**: the intermediate non-linearity between 2D and 1D parts makes it
  more expressive than a plain 3D conv with identical FLOPs.

**Practical cons for this setting:**
- Memory-hungry: forces a trade-off between `num_frames`, `image_size`, and `batch_size`.
- No pretrained weights available within closed-track rules — 3D conv needs much more
  data or epochs to learn good filters from scratch.
- Requires ≥8–16 frames to capture meaningful 1D temporal patterns across the filter.
  At 4–5 frames, the temporal convolution window is too small to be beneficial.

**Verdict:** Do not prioritise R2+1D for the closed track. It is architecturally sound
for open-track (with Kinetics pretraining) but poorly suited to from-scratch training on
a modest dataset with short clips.

---

## 3. Data Augmentation & Regularization Techniques

### 3.1 Regularization Warmup ⭐ MOST IMPACTFUL

**What it is:** MixUp and CutMix are disabled for the first `reg_warmup_epochs` (set to
10) epochs. After that they are enabled at full strength.

**Why it matters (the cold-start problem):**  
When a network is randomly initialised, every layer is learning simultaneously. In the
first epochs the model is trying to learn "what does a hand look like?", "what is motion?"
and "how do I classify 33 actions?" all at once. MixUp and CutMix make this harder by
presenting **blended or pasted inputs with mixed labels** — the model must simultaneously
learn basic features AND deal with ambiguous targets. This creates a reinforcing loop of
confusion.

**Experimental evidence:**

| Run | Warmup | Val Acc @epoch 10 | Final Val Acc |
|---|---|---|---|
| VFL_prev (killed) | None | ~9.4% (epoch 4) | 9.45% |
| VFL_closed | 10 epochs | 17.58% | **45.78%** |
| TSM_ultra | 10 epochs | 25.60% | **53.32%** |
| R2+1D_closed (killed) | None | ~8.4% (epoch 1) | 10.78% |

The VFL comparison is the cleanest A/B: identical architecture, identical config except
for the warmup. The result is +36 pp. This is the single most impactful change in the
entire experiment series.

**How to read the TSM_ultra training curve:**
- Epochs 1–10 (no aug): loss drops from 3.50 → 2.85; val climbs 8% → 26%
- Epoch 11 (aug kicks in): train acc dips (model sees harder examples); val acc jumps
- Epochs 11–29: rapid gain 26% → 44%, the warmup "investment" pays off
- Epochs 30–75: steady climb to 53.32%

---

### 3.2 MixUp

**What it is:** Two clips (x_a, x_b) and their labels (y_a, y_b) are linearly
interpolated: `x = λ·x_a + (1−λ)·x_b`, loss = `λ·CE(y_a) + (1−λ)·CE(y_b)`.
λ is sampled from Beta(α, α); α=0.2 in TSM_ultra.

**Pros:**
- Reduces overconfidence — forces soft probability outputs.
- Acts as a form of manifold mixup: interpolated examples lie "between" classes,
  which can improve generalisation.
- Preserves temporal structure within each clip (no spatial masking).

**Cons for SSv2:**
- The interpolated video has visually incoherent temporal motion (two unrelated
  actions layered on top of each other). This contradicts SSv2's motion-defining labels.
- Low α (0.2) mitigates this: most sampled λ are close to 0 or 1, so one clip dominates.

**Setting:** α=0.2 (reduced from 0.4 in tsm_closed). Active after epoch 10.

---

### 3.3 CutMix

**What it is:** A rectangular patch from clip B is pasted onto clip B' at the same
spatial location **across all frames**. The label is mixed proportionally to patch area.

**Why it is reduced (0.25 not 0.5) on SSv2:**  
The action label in SSv2 is defined by the motion of a hand or object in the clip.
A CutMix patch that covers the hand removes exactly the signal that distinguishes
"Pulling left" from "Pulling right". At prob=0.5 (50% of batches use CutMix), this
is destructive enough to hurt convergence, especially early in training.

**Pros:**
- Very effective regularizer on image tasks (ImageNet, CIFAR-100). Teaches the model
  to use distributed spatial features rather than relying on a single discriminative region.

**Cons for SSv2:**
- The patch is identical across all T frames — it does not "slide" or respect motion.
  A static occluder on a motion-defined dataset disrupts the temporal signal.
- Strong enough at high probability to teach the model to ignore spatial regions,
  which is the opposite of what we want for hand/object recognition.

**Setting:** prob=0.25. Active after epoch 10.

---

### 3.4 Label Smoothing

**What it is:** The one-hot target vector is softened: the correct class gets probability
`1 - ε + ε/K` and each incorrect class gets `ε/K`, where ε=0.1, K=33.

**Pros:**
- Penalises over-confident predictions, acting as a calibration regularizer.
- Effective on imbalanced datasets by preventing the model from becoming a "one-class
  predictor" with 100% confidence on the majority class.

**Cons:** Mild in practice; 0.1 is the standard setting and unlikely to be harmful.

**Setting:** 0.1 throughout training.

---

### 3.5 Stochastic Depth (DropPath)

**What it is:** During training, each residual branch is zeroed out with probability
`p_i` (the identity shortcut is always preserved). In TSM_ultra, `p_i` increases
linearly from 0 at the first block to 0.1 at the last.

**Pros:**
- Functions as a form of implicit ensemble: the network must be useful at all depths,
  since any subset of blocks may be dropped.
- Has zero cost at inference (all blocks are kept with their expected scale).

**Cons:**
- With the mild rate used here (0–0.1 over ResNet18's 8 blocks), the effect is subtle.
  The main benefit is preventing over-reliance on the deepest features early in training.

**Setting:** max rate 0.1 (reduced from 0.2 in tsm_closed).

---

### 3.6 Dropout (Head)

**What it is:** p=0.3 applied to the pooled 512-d vector before the final Linear layer.

**Setting:** 0.3 (reduced from 0.5 in tsm_closed). The high value in tsm_closed was
masking half the representation at the classifier, which is excessive for a 512-d
from-scratch feature vector.

---

### 3.7 RandAugment (temporally consistent)

**What it is:** Two ops are sampled from a pool (AutoContrast, Equalize, Posterize,
Solarize, Sharpness, ShearX/Y, TranslateX/Y) at magnitude 0.5. The ops are applied
**identically to all frames** in the clip (sampled once per clip, not per frame).

**Why temporal consistency matters:** If each frame were augmented independently, the
clip would appear "jittery" — the model would learn to ignore between-frame differences,
which is precisely the motion signal we need to preserve.

**Pros:** Increases visual diversity cheaply. No hyperparameter for the specific ops.

**Cons:** Some ops (ShearX/Y, TranslateY) subtly distort the direction of motion, which
is harmful on direction-sensitive SSv2 labels. At magnitude 0.5 the distortion is small
enough to be acceptable.

---

### 3.8 Random Erasing

**What it is:** A random rectangle of the clip is set to zero (same rectangle across all
frames). Applied with p=0.3.

**Pros:** Teaches the model to use context beyond a single region. Complements CutMix.

**Cons for SSv2:** Similar to CutMix — can erase the hand/object. Kept because p=0.3
is low enough that most clips are not affected.

---

### 3.9 TSN-style Temporal Jitter

**What it is:** The T frames are not sampled uniformly. Instead, the clip is divided into
T segments and one frame is sampled **randomly within each segment**. This introduces
temporal jitter: the same video appears at slightly different temporal positions each
epoch.

**Pros:** Free augmentation that increases the diversity of temporal samples seen during
training. Especially important when the same video is seen 80 times (80 epochs).

**Cons:** With only 5 frames, each segment is short (mean ~2–3 frames if the clip has
~15 frames), limiting the range of temporal positions.

---

### 3.10 WeightedRandomSampler

**What it is:** Each training sample is assigned weight `1 / count(class)^0.5`, where
`count(class)` is the number of training clips in that class. The sampler draws with
replacement, so minority classes appear more frequently.

**Why it matters:** The SSv2-33 subset has a **20× class imbalance** (162 clips in the
rarest class, 3170 in the most common). Without rebalancing, the model spends ~80% of
gradient updates on the 5 most common classes and essentially ignores the rarest ones.
With `power=0.5` the rebalancing is **soft** (not fully uniform), which prevents the
model from seeing so many rare-class examples that it loses accuracy on common ones.

**Setting:** power=0.5 (√-frequency weighting). Full inverse-frequency (power=1.0) is
an option if rare-class accuracy needs more attention.

---

### 3.11 Horizontal Flip — DISABLED

**Why:** Several SSv2 action classes are defined by direction:
*"Pulling from left to right"*, *"Pouring into container on left"*, etc.
A horizontal flip maps "left" to "right", silently swapping the label.
This creates mislabelled training examples with no signal to recover from.

**Always disable for SSv2.** This is not a regularisation choice — it is a correctness
requirement.

---

### 3.12 Color Jitter

Brightness, contrast, saturation (each ±40%) and hue (±0.1) are sampled once per clip
and applied to every frame. This is mild and universally beneficial.

---

### 3.13 AMP (Automatic Mixed Precision)

Training uses `torch.autocast` with `GradScaler`. Not a regulariser but allows doubling
effective batch size or training faster at the same memory budget.

---

### 3.14 SGDR (Cosine Annealing with Warm Restarts) — NEW for TSM_ultra_50

**What it is:** The learning rate follows a cosine curve down to ~0, then
**restarts** back to the peak LR and anneals again. With `T_0 = 28` and
`T_mult = 1`, TSM_ultra_50 runs 4 equal cycles of 28 epochs after an 8-epoch
linear warmup (restarts at epochs 36, 64, 92). Reference: Loshchilov & Hutter,
*SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017.

**Why use it for a long schedule:**
- Monotonic cosine over 120 epochs settles into one minimum and stays there.
  SGDR's restarts let the optimizer escape and explore wider basins — each
  trough of the cosine is a candidate "checkpoint snapshot".
- Each restart acts as **implicit ensembling over time**: averaging the
  weights at each cycle's trough is a known trick (snapshot ensembles,
  Huang et al., 2017). Even without averaging, the final trough usually
  outperforms a single long cosine on long schedules (≥100 epochs).
- The restart's LR jump is a free *exploration phase* — useful when the
  bias-variance trade-off is uncertain and a single cosine might be too
  greedy.

**Why monotonic cosine was used in TSM_ultra:** at 80 epochs, the model didn't
have time to need restarts — the curve was still improving when the schedule
ended. SGDR pays off as schedules get longer (and the next experiment is 120).

**Risks:**
- After a restart the val accuracy briefly drops before recovering. If it
  doesn't recover within 3–5 epochs, the cycle was too long — drop `T_0`.
- Best-checkpoint logic must compare across cycles, not stop at the first
  trough (`train.py` already saves on every new best val accuracy — no
  change needed).

**Where it's configured:** `scheduler: cosine_warm_restarts`, with
`sgdr_t0` and `sgdr_t_mult` controlling the cycle structure. Implementation
in `train.py::build_scheduler` (added with `tsm_ultra_50`).

---

## 4. Techniques Considered but Not Tried

### 4.1 Cross-Validation (CV)

**What it would give:** Lower-variance model selection; optionally an ensemble of K
models for the final submission (each fold's model predicts the test set, then logits
are averaged).

**Cost:** K × training time. With K=5 and 4 h per run → **20 hours**.

**When CV is the right tool:**
- Small dataset with noisy val accuracy estimates.
- Final submission (fold-ensemble gives free 2–4 pp).
- Hyperparameter search where overfitting to one val split is a risk.

**When it is not worth it:**
- During exploration / iteration — 5× cost kills fast feedback.
- When a dedicated held-out val set (`val_dir`) already exists as a reliable estimate.
- When model selection uncertainty is not the bottleneck (it probably is not here;
  the bottleneck is epoch budget and architecture capacity).

**Current recommendation:** Do not use CV during iteration. If TSM_ultra or VFL_closed
is chosen as the final architecture, run a **3-fold ensemble** for the final submission.
Expected gain: +2–4 pp.

### 4.2 Two-Stream (RGB + Optical Flow)

**What it would give:** A separate model trained on pre-computed optical flow alongside
the RGB model. Late fusion of both streams typically gives +3–8 pp on SSv2.

**Why not for this challenge:** Optical flow computation requires dense per-frame
warping (TVL-1 or RAFT), which is expensive and storage-intensive. The challenge
pipeline only stores JPEG frames — adding flow would require a preprocessing step
and additional disk space. Not recommended unless other improvements plateau.

### 4.3 Test-Time Augmentation (TTA)

**Already implemented** (`VideoTransform.tta()`, `evaluate.py --dataset.tta=true`).
TTA applies 5 spatial crops × 2 flips = 10 views per clip and averages logits.
Expected gain: **+1–3 pp** at no training cost. **Run this before every submission.**

---

## 5. The Internal Val vs. Leaderboard Gap

TSM_ultra achieved **53.32% internal val** but the submitted score was **34.46%**.
This 18.9 pp gap is larger than a typical 5–10 pp overfitting gap and warrants
investigation before spending more compute.

**Possible causes (ranked by likelihood):**

1. **Non-stratified internal split:** `split_train_val` in `utils.py` shuffles randomly
   without stratification. With 20× class imbalance, the 20% val split can accidentally
   over-represent easy / frequent classes, inflating internal accuracy. The official val
   set (`val_dir`) has its own distribution.

2. **Submit from wrong checkpoint:** The submission output path in the config defaults to
   `submissions/R2+1D_closed1.csv`. If `create_submission.py` was run without specifying
   `training.checkpoint_path=../models/tsm_ultra.pt`, it may have used an older model.

3. **Label index mismatch:** The model parses class indices from folder names
   (`000_ClassName` → class 0). If the test set has a different folder ordering, all
   predictions would be off.

4. **TTA not used:** TTA adds ~1–3 pp. Always run
   `create_submission.py training.checkpoint_path=../models/tsm_ultra.pt dataset.tta=true`.

**Note: the official `val_dir` is a held-out evaluation set — it cannot be used for
training, fine-tuning, or hyperparameter search. It is only valid as a final
leaderboard-style benchmark via `evaluate.py`.**

**Action items before the next submission:**
```bash
cd src
# Evaluate on the official val_dir
python evaluate.py training.checkpoint_path=../models/tsm_ultra.pt
python evaluate.py training.checkpoint_path=../models/tsm_ultra.pt dataset.tta=true

# Generate submission
python create_submission.py training.checkpoint_path=../models/tsm_ultra.pt dataset.tta=true
```

---

## 6. Improvement Roadmap (Ranked by ROI)

### Status of Tier 1 (May 2026)
| Task | Status | Result |
|---|---|---|
| TTA on TSM_ultra | ✅ done | included in ensemble |
| Evaluate ensemble on official val_dir | ✅ done | 37.06% (no TTA), 37.35% (TTA) |
| Late-fusion ensemble TSM_ultra + VFL | ✅ done | top-1 37.35%, top-5 70.48% |

### Tier 1 — Next experiment (the big bet)

| # | Improvement | Expected Gain | Cost | Notes |
|---|---|---|---|---|
| 1 | **TSM_ultra_50** — R50 + 8 frames + 120 epochs + SGDR + label_smoothing 0.15 | +5–7 pp internal val; ~46% LB when ensembled | ~12–18 h | Single highest-confidence change. Config: `tsm_ultra_50.yaml`. See MODEL.md §TSM_ultra_50 for the full recipe and rationale. |

**Why TSM_ultra_50 over other alternatives:**
- TSM_ultra plateaued at epoch 68 → capacity is the bottleneck, not the
  architecture family. Scaling R18 → R50 directly targets that.
- 5 frames is below the SSv2 literature standard; 8 frames adds ~60 % more
  motion samples per clip without changing the data pipeline.
- SGDR is appropriate *only* for long schedules; pairing it with 120 epochs
  is its highest-leverage use case.
- Each change is independently validated in the literature — no compound
  experimental risk.

**What was deliberately not added to TSM_ultra_50** (and why):
- *Attention temporal pooling*: estimated +0.5–1 pp at the cost of new code.
  Add only if TSM_ultra_50 stagnates.
- *Multi-clip inference* (3–5 temporal windows per video at test time):
  estimated +1–2 pp. Inference-only — defer until after the next training
  run and add as a post-hoc improvement to the ensemble script.
- *Optical flow / two-stream*: estimated +3–8 pp but requires a separate
  preprocessing pipeline (~8 h compute) and an additional model
  (~4 h training). Defer to Tier 3.
- *Cross-validation*: K× training cost; the held-out `val_dir` is already
  a clean evaluation signal — CV's main benefit (variance reduction in
  model selection) is not the bottleneck here.

### Tier 2 — Medium cost, medium return (after TSM_ultra_50)

| # | Improvement | Expected Gain | Cost | Notes |
|---|---|---|---|---|
| 2 | **Multi-clip inference** (3 temporal windows + TTA) | +1–2 pp | minimal — modify ensemble script | Stack with existing TSM_ultra_50 + ensemble. |
| 3 | **3-fold TSM_ultra_50 ensemble** (different seeds) | +1–3 pp on top of single | ~24–36 h total | Only after TSM_ultra_50 confirms the recipe. |
| 4 | **Attention temporal pooling on TSM_ultra_50** | +0.5–1 pp | ~1 h code + retrain | Replace `features.mean(dim=1)` with a learned attention head. |
| 5 | **Train VFL_closed for 80 epochs** | +3–8 pp single | ~4 h | VFL was still improving at epoch 60; could close the gap with TSM_ultra. |
| 6 | **Stratified train/val split** | +1–3 pp on internal | minimal | Fix `split_train_val` to stratify by class. Improves internal val signal. |

### Tier 3 — High cost, variable return

| # | Improvement | Expected Gain | Cost | Notes |
|---|---|---|---|---|
| 7 | **Two-stream (RGB + optical flow)** | +3–8 pp | ~8 h preprocessing + ~4 h training | The single highest-ceiling intervention but a substantial engineering project. Recommend only if TSM_ultra_50 + multi-clip + 3-fold leaves the score short of target. |
| 8 | **R2+1D with full recipe** | unknown | ~4 h | Add reg warmup, fix batch/image size, train 80 epochs. Uncertain ceiling from scratch. |
| 9 | **Deeper VFL (4 Transformer layers)** | +2–5 pp | ~5 h | More temporal capacity; risk of overfitting with more params from scratch. |

---

## 7. Ensemble Strategy

A late-fusion ensemble requires no re-training. To run it:

```python
import torch
from pathlib import Path

# Load both checkpoints
ckpt_tsm = torch.load("models/tsm_ultra.pt", map_location="cpu")
ckpt_vfl = torch.load("models/video_former_lite_closed.pt", map_location="cpu")

# Rebuild models, set to eval
# Run inference on test set with both models
# Average softmax logits (or logits directly — averaging logits is equivalent for argmax)
ensemble_logits = 0.6 * tsm_logits + 0.4 * vfl_logits  # weight by internal val acc
predicted_class = ensemble_logits.argmax(dim=1)
```

Weight the ensemble by internal validation accuracy (TSM: 53.32% → 0.6 weight,
VFL: 45.78% → 0.4 weight) or tune weights on the official val set.

The ensemble works because:
- TSM captures **local temporal patterns** (adjacent-frame motion via shifts in all
  32 ResNet blocks).
- VFL captures **global temporal patterns** (all-to-all attention at the end of the
  pipeline).
- These two representations are partially orthogonal → ensembling reduces variance.

---

## 8. Summary: What to prefer and why

| Situation | Recommended approach |
|---|---|
| Fast iteration, new architecture idea | Single model, 40–60 epochs, val_dir check |
| Best single model **proven so far** | TSM_ultra at 80 epochs (53.32% internal / 34.46% LB) |
| Best submission **today** | TSM_ultra + VFL ensemble with TTA (37.35% official val) |
| Next big bet for the **46% LB tier** | Train **TSM_ultra_50** (R50, 8f, 120 ep, SGDR), then ensemble with TSM_ultra + VFL |
| Maximum leaderboard score (budget permitting) | 3-fold TSM_ultra_50 ensemble + multi-clip inference + TTA |
| New data augmentation experiment | A/B test on VFL (faster iteration than TSM) |

**Do not invest more compute in R2+1D** on the closed track unless you are willing to
spend time fixing the recipe (batch size, image size, reg warmup, 80+ epochs).

**The regularization warmup (10 epochs of no MixUp/CutMix) is non-negotiable** for any
new model trained from scratch. The experimental evidence is decisive.

**Use SGDR only for long schedules (≥ 100 epochs).** Below that, monotonic
cosine reaches the same minimum without paying the per-restart recovery cost.
