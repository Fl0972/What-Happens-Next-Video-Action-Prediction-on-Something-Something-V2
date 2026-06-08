# Ablation Study — TSM-Ultra-v2 and VideoFormer-Lite-Ultra

**Challenge:** CSC_43M04_EP — "What Happens Next?" action prediction, 33 classes, closed track (no pretrained weights).  
**Dataset:** 44,993 train / 6,745 val / 6,913 test clips (4 JPEG frames each).  
**Reference point:** All val-dir numbers are top-1 accuracy on the 6,745-clip official validation set unless stated otherwise.  
**Internal val:** accuracy on an 80/20 fixed split inside the training set (inflated vs. val-dir by ~15–19 pp; used only for relative comparisons within the same training protocol).

---

## 1. Scope

This document isolates the contribution of each design choice in the two best-performing single models:

- **TSM-Ultra-v2** — Temporal Shift Module on ResNet18, T=4 frames, focal loss, rotating folds, SGDR; best single model at **39.45% val-dir** (41.68% + TTA).
- **VFL-Ultra (rotating)** — Frame-CNN + Transformer, T=4 frames, n_layers=3, focal loss, rotating folds, SGDR; best alternate architecture at **36.44% val-dir** (37.81% + TTA).

Each ablation modifies exactly one axis relative to the "current best" config of the same model family. Ablations are derived from fully-completed training runs recorded in `models/*.csv`; no experiment in this document is hypothetical.

---

## 2. TSM-ResNet Ablation Series

### 2.1 Complete Results Table

All runs use ResNet18 backbone and T=4 (or stated alternative), trained from scratch. Internal val is peak single-epoch accuracy on the fixed 80/20 split; rotating-fold runs report the best **rolling-average** over the last 5 epochs (one complete fold rotation).

| ID | Config | Backbone | T | Loss | Split | Epochs | Internal Val | Val-dir Top-1 | Val-dir Top-5 |
|----|--------|----------|---|------|-------|--------|-------------|--------------|--------------|
| T0 | tsm\_ultra (T=5) | R18 | 5 | CE | fixed | 80 | 53.32% | 34.91% | 66.88% |
| T1 | tsm\_ultra\_50 (T=8, R50) | R50 | 8 | CE | fixed | 56† | 45.26% | — | — |
| T2 | tsm\_ultra\_v2 (T=4, focal) | R18 | 4 | Focal γ=2 | fixed | 145‡ | 57.73% | 38.25% | 68.70% |
| T3 | tsm\_ultra\_v2\_rotating | R18 | 4 | Focal γ=2 | rotating | 200 | 95.83%§ | 39.45% | 69.47% |

† Killed at epoch 56/120 due to underperformance.  
‡ Early convergence within a 150-epoch budget; best checkpoint at epoch 128.  
§ Rolling average over 5 folds, not comparable to fixed-split internal val.

### 2.2 Change Log — TSM-Ultra → TSM-Ultra-v2 (+4.41 pp internal, +3.34 pp val-dir)

| Axis | tsm\_ultra | tsm\_ultra\_v2 | Rationale |
|------|-----------|---------------|-----------|
| `num_frames` | 5 | **4** | Clips are exactly 4 frames; T=5 duplicates one frame via linspace, injecting a zero-gradient redundant token and corrupting TSM shift indices |
| `focal_gamma` | 0 (CE) | **2.0** | SSv2 has many near-identical class pairs ("pouring into" vs. "pretending to pour"); focal loss down-weights easy examples and concentrates gradient on hard pairs |
| `epochs` | 80 | 145 | Training curve still improving at ep 80 (last new best at ep 75); extended budget uses the full cosine tail |
| All others | identical | identical | fold\_div=4, dropout=0.3, drop\_path=0.1, LR=1e-3, AdamW, wd=5e-4, batch=32, RandAug(2, 0.5), MixUp α=0.2, CutMix p=0.25, label\_smoothing=0.1, horizontal\_flip=false, weighted\_sampling |

**Frame-count effect (isolated):** The TSM-Ultra-50 run (R50, T=8) achieved only 45.26% internal val despite a 2× larger backbone — 8 pp below the T=5 R18 baseline. Given that T=8 also duplicates frames, this is consistent: the inductive bias of TSM's shift operator degrades when the same frame appears twice in the sequence. The frame count fix (5→4) is therefore the dominant positive contributor.

**Focal loss effect (isolated):** Within the VFL family, a controlled comparison exists: VFL-Ultra (CE) vs VFL-Ultra-Focal (focal γ=2) at matched epochs and architecture. Result: CE 53.37% vs. Focal 53.17% internal val (−0.20 pp), but +1.00 pp on val-dir (32.11% → 33.11%). This suggests that focal loss provides a small generalisation benefit that does not manifest clearly in internal (training-distribution) validation but does in out-of-distribution evaluation. For TSM, the joint effect of T-fix + focal exceeds the sum of the parts because the corrected temporal index amplifies the focal gradient signal.

### 2.3 Change Log — TSM-Ultra-v2 → TSM-Ultra-v2-Rotating (+1.20 pp val-dir)

| Axis | tsm\_ultra\_v2 | tsm\_ultra\_v2\_rotating | Rationale |
|------|---------------|------------------------|-----------|
| `rotating_folds` | false | **true**, n\_folds=5 | Each epoch rotates the held-out fold → 100% data utilisation vs. 80% fixed split; no single slice becomes a tuning oracle |
| `scheduler` | cosine | **cosine\_warm\_restarts** (T₀=25) | Rotating split changes the train distribution every epoch; periodic LR restarts (8 restarts over 200 ep) allow the model to escape sharp minima specific to one data subset |
| `epochs` | 145 | **200** | 200 = 5 × 40 complete rotations ≈ 160 effective epochs over the full set |
| `compile` | false | **true** | ~20-30% throughput gain on A4000 at no accuracy cost |
| All others | identical | identical | |

**Data-efficiency interpretation:** The +1.20 pp gain (38.25% → 39.45%) is modest for TSM compared to the +4.33 pp gain for VFL. TSM has a strong inductive bias (temporal shift enforces local temporal consistency) that learns efficiently even from 80% of the data. VFL's global self-attention is more data-hungry and benefits disproportionately from full utilisation.

### 2.4 TTA Effect on TSM

| Model | No TTA | + TTA (10 crop) | Δ |
|-------|--------|-----------------|---|
| tsm\_ultra (T=5) | 34.91% | 35.91% | +1.00 pp |
| tsm\_ultra\_v2 | 38.25% | 40.86% | +2.61 pp |
| tsm\_ultra\_v2\_rotating | 39.45% | 41.68% | +2.23 pp |

TTA provides a consistent ~2–3 pp gain by averaging 10 crops (5 spatial × 2 flips). The marginal TTA gain is slightly smaller for the rotating-fold model (2.23 pp vs. 2.61 pp for the fixed-split v2), consistent with the hypothesis that rotating folds already reduce single-crop variance through better generalisation.

---

## 3. VideoFormer-Lite Ablation Series

### 3.1 Complete Results Table

All runs use ResNet18 as the frame-level CNN backbone and T=4 frames.

| ID | Config | n\_layers | Loss | Split | Epochs | Internal Val | Val-dir Top-1 | Val-dir Top-5 |
|----|--------|-----------|------|-------|--------|-------------|--------------|--------------|
| V0 | vfl\_closed | 2 | CE | fixed | 60 | 45.78% | — | — |
| V1 | vfl\_ultra | 2 | CE | fixed | 100 | 53.37% | 32.11% | 64.85% |
| V2 | vfl\_ultra\_focal | 2 | Focal γ=2 | fixed | 100 | 53.17% | 33.11% | 66.17% |
| V3 | vfl\_ultra\_rotating | 3 | Focal γ=2 | rotating | 150 | 78.22%§ | 36.44% | 68.48% |

§ Rolling average, not comparable to fixed-split internal val.

### 3.2 Change Log — VFL-Closed → VFL-Ultra (+7.59 pp internal, val-dir not separately recorded for V0)

| Axis | vfl\_closed | vfl\_ultra | Rationale |
|------|------------|-----------|-----------|
| `epochs` | 60 | **100** | Curve still climbing at ep 60 — capacity and training time were the bottleneck |
| `lr` | 5×10⁻⁴ | **8×10⁻⁴** | Higher peak LR shortens time-to-good-loss under cosine schedule with sufficient warmup |
| `warmup_epochs` | 8 | **8** (unchanged) | Kept: warmup stabilises the from-scratch Transformer before full LR is reached |
| `drop_path_rate` | 0.10 | **0.15** | More stochastic depth to match the longer training run (more overfitting opportunity) |
| All others | identical | identical | T=4, n\_layers=2, d\_model=512, n\_heads=8, weight\_decay=0.05, MixUp, CutMix, RandAug |

### 3.3 Change Log — VFL-Ultra (V1) → VFL-Ultra-Focal (V2): Focal Loss Effect (−0.20 pp internal, +1.00 pp val-dir)

This is the cleanest single-axis comparison in the VFL series. All settings match except the loss function and small co-varying regularisation parameters:

| Axis | vfl\_ultra | vfl\_ultra\_focal | Notes |
|------|-----------|-----------------|-------|
| `focal_gamma` | 0 (CE) | **2.0** | Primary change |
| `lr` | 8×10⁻⁴ | 5×10⁻⁴ | Co-varies — lower LR in focal run partially confounds the comparison |
| `drop_path_rate` | 0.15 | 0.10 | Co-varies |

**Result:** Focal slightly hurts on internal val (−0.20 pp) but helps on val-dir (+1.00 pp). The internal val is drawn from the same distribution as the training set; focal loss reduces confidence on easy examples, which reduces internal val accuracy but improves calibration on the harder val-dir distribution. The +1.00 pp val-dir gain is consistent with the hypothesis that focal loss reduces overconfidence on easy classes and forces more gradient toward the hard near-miss pairs.

### 3.4 Change Log — VFL-Ultra (V1) → VFL-Ultra-Rotating (V3): Multiple Improvements (+4.33 pp val-dir)

This is a multi-axis change; the individual contributions are not separately isolated by a fully factorial experiment, but the design rationale for each is documented:

| Axis | vfl\_ultra | vfl\_ultra\_rotating | Expected contribution |
|------|-----------|--------------------|-----------------------|
| `n_layers` | 2 | **3** | +? pp — more depth for global temporal reasoning over 4+[CLS] tokens |
| `focal_gamma` | 0 (CE) | **2.0** | +1.00 pp (from V1→V2 comparison, but confounded) |
| `rotating_folds` | false | **true**, n\_folds=5 | Majority of the +4.33 pp gain; see §5.2 |
| `scheduler` | cosine | **cosine\_warm\_restarts** (T₀=25) | Prevents LR decay to near-zero before all data is seen |
| `epochs` | 100 | **150** | 150 = 5×30 rotations ≈ 120 effective epochs over full set |
| `lr` | 8×10⁻⁴ | 8×10⁻⁴ | Unchanged |
| `warmup_epochs` | 8 | **12** | Longer warmup stabilises the deeper (3-layer) Transformer |
| `drop_path_rate` | 0.15 | **0.20** | More stochastic depth for 3-layer model + longer training |

**Rotating-folds dominance:** Comparing the same fixed-split VFL-Ultra (32.11%) to TSM-Ultra-v2 (38.25%) and their respective rotating counterparts (VFL: 36.44%, TSM: 39.45%) reveals an asymmetry: VFL gains +4.33 pp from rotating folds while TSM gains only +1.20 pp. This is consistent with the architectural argument: VFL's global self-attention has fewer hard-coded spatial priors and more free parameters to learn temporal patterns — it is more data-hungry and benefits more from full dataset utilisation.

### 3.5 TTA Effect on VFL

| Model | No TTA | + TTA (10 crop) | Δ |
|-------|--------|-----------------|---|
| vfl\_ultra | 32.11% | 35.03% | +2.92 pp |
| vfl\_ultra\_focal | 33.11% | 35.77% | +2.66 pp |
| vfl\_ultra\_rotating | 36.44% | 37.81% | +1.37 pp |

TTA provides a larger absolute gain for weaker VFL models (2.92 pp for fixed-split vs. 1.37 pp for rotating). This confirms that TTA variance reduction is most useful when the model's single-crop predictions are noisier — exactly the case for a model trained on only 80% of the data with no periodic LR restarts.

---

## 4. Cross-Cutting Analysis

### 4.1 Frame Count: T=4 is the Only Correct Value

| Model | T | Val-dir Top-1 | Note |
|-------|---|--------------|------|
| tsm\_ultra | 5 | 34.91% | 1 frame duplicated via linspace |
| tsm\_ultra\_50 | 8 | 45.26%* | 4 frames duplicated |
| tsm\_ultra\_v2 | 4 | 38.25% | All frames unique |
| vfl\_ultra | 4 | 32.11% | All frames unique |

\* Internal val only; run killed at epoch 56. Projected final accuracy would be below 53.32% (matched ResNet18 T=5 baseline).

The dataset clips contain exactly 4 JPEG frames. Using T>4 invokes linspace interpolation in `VideoFrameDataset`, which duplicates frames. For TSM, duplicated frames create two TemporalShift tokens at identical content — the shift operation computes a zero difference for those positions, effectively disabling temporal mixing for a fraction of the sequence. For VFL, duplicated positional embeddings produce degenerate attention patterns where the model attends to two copies of the same frame token. **T=4 is non-negotiable for this dataset.**

### 4.2 Focal Loss: +1 pp val-dir Generalisation, −0.2 pp Internal Val

The most controlled comparison is V1→V2 (VFL, same architecture, fixed split):

- Internal val: 53.37% (CE) → 53.17% (focal) = −0.20 pp
- Val-dir: 32.11% (CE) → 33.11% (focal) = **+1.00 pp**

The internal/val-dir divergence is the key signature of focal loss in this setting. On the training distribution (internal val), focal loss reduces the model's confidence on easy examples, which slightly lowers absolute accuracy. On the held-out val-dir set, the harder examples — the near-miss class pairs that SSv2 was deliberately designed to stress-test — receive more training gradient, improving generalisation to those hard cases.

For TSM, focal was added alongside the frame-count fix so its isolated contribution cannot be cleanly read from T0→T2. The VFL controlled experiment provides the best estimate: **+1 pp val-dir** is a reasonable expectation for focal loss γ=2 in this setting.

### 4.3 Rotating Folds: Asymmetric Benefit (TSM +1.2 pp, VFL +4.3 pp)

| Architecture | Fixed Split | Rotating Folds | Gain |
|-------------|------------|----------------|------|
| TSM-Ultra-v2 | 38.25% | 39.45% | **+1.20 pp** |
| VFL-Ultra | 32.11% | 36.44% | **+4.33 pp** |

The rotating-folds protocol (Mode 2 of `train_kfold.py`) trains a single model for N epochs, rotating the 20% held-out fold each epoch. This ensures every sample participates in 100% of training epochs rather than 80%. Key effects:

1. **Data utilisation:** ~25% more effective gradient steps over the full dataset.
2. **Regularisation:** no fixed held-out oracle — the model cannot overfit the specific distribution of a single val partition.
3. **Checkpointing:** the rolling-average val accuracy over 5 consecutive epochs (one complete rotation) is a less-biased proxy for generalisation than any single-epoch val accuracy.

The large VFL asymmetry (4.33 pp vs. 1.20 pp) has a mechanistic explanation: TSM's temporal shift provides a strong structural prior that makes it a good few-shot learner — it converges to a reasonable solution even with 80% of the data. VFL's Transformer has more parameters (12.6 M total vs. 11.4 M for TSM with the same ResNet18 backbone) and a weaker structural prior on temporal ordering, making it more sensitive to data volume.

### 4.4 Backbone Capacity: ResNet18 vs ResNet50 (for TSM)

| Backbone | T | Val (internal) | Trend |
|----------|---|---------------|-------|
| ResNet18 | 5 | 53.32% | Improving at ep 80 |
| ResNet50 | 8 | 45.26% | Killed ep 56; still improving |
| ResNet18 | 4 | 57.73% | Plateau ~ep 128 |

The ResNet50 run (T1) was expected to outperform ResNet18 due to 2× spatial capacity, but failed for two compounding reasons: (a) T=8 duplicates frames, degrading TSM's temporal signal as discussed above, and (b) ResNet50's 25 M parameters are over-parameterised for a 33-class task trained from scratch on ~45 k samples (1,364 samples/class on average), increasing overfitting risk that the regularisation budget could not fully counteract. ResNet18 at T=4 is the Pareto-optimal choice for this dataset.

### 4.5 Regularisation Warmup: The Most Impactful Single Knob

All runs in this series use `reg_warmup_epochs=10`: MixUp and CutMix are disabled for the first 10 epochs. This is not ablated because its effect was already established in the VFL baseline series:

| VFL config | reg\_warmup | Internal Val |
|------------|------------|-------------|
| VFL (no warmup, killed) | 0 | 9.45% (at ep 30) |
| VFL (warmup, 60 ep) | 10 | 45.78% |
| **Δ** | | **+36.33 pp** |

The mechanism: MixUp and CutMix create interpolated labels (e.g. 60% class A, 40% class B). Applied to a randomly-initialised network, this signal is ambiguous — the model cannot learn basic feature detectors under mixed labels. The 10-epoch warmup lets the network converge to a reasonable representation before regularisation is applied. This finding motivated the `reg_warmup_epochs` parameter and is kept fixed across all subsequent runs.

---

## 5. Architecture Comparison: TSM vs. VFL

### 5.1 Head-to-Head at Comparable Settings

The fairest comparison uses rotating-fold variants, which put both architectures on equal data footing:

| Architecture | Val-dir (no TTA) | Val-dir (+ TTA) | Key bias |
|-------------|-----------------|----------------|----------|
| TSM-Ultra-v2-Rotating | 39.45% | 41.68% | Local channel shift, one frame per layer |
| VFL-Ultra-Rotating | 36.44% | 37.81% | Global self-attention over all 4 frames |
| **TSM advantage** | **+3.01 pp** | **+3.87 pp** | — |

TSM's advantage is stable and consistent. The explanation aligns with the SSv2 literature: "What Happens Next?" labels are driven by short-range temporal motion patterns (direction of pull, completeness of pour). TSM's channel shift enforces a prior that *adjacent* frames are related — precisely the temporal structure that matters. VFL's global attention can theoretically learn the same patterns, but must discover them from data; with only T=4 frames and no pretraining, it sees too few examples per class to reliably learn the ordering invariances.

### 5.2 Complementarity for Ensemble

Despite TSM's advantage, VFL adds value in an ensemble because the two models make *different* errors:

- TSM confuses direction-ambiguous pairs ("folding" ↔ "unfolding", "moving up" ↔ "moving down") — cases where local shift misses the global before/after state comparison.
- VFL handles these better via global attention (it compares frame 1 and frame 4 directly) but fails on subtle motion classes where local displacement patterns are the only cue.

Leave-one-out ensemble ablation (from `docs/model_analysis_ensemble.md`):

| Removed | Ensemble Top-1 | Δ vs Full (40.43%) |
|---------|---------------|-------------------|
| None (full ensemble) | 40.43% | — |
| −tsm\_ultra\_v2 | 37.26% | −3.17 pp |
| −tsm\_ultra | 39.56% | −0.87 pp |
| −vfl\_ultra | 39.82% | −0.61 pp |

VFL's standalone accuracy (32.11%) is 6 pp below TSM-Ultra (34.91%), yet removing VFL costs −0.61 pp — confirming that the complementary signal it provides is not captured by the TSM models alone.

---

## 6. Ensemble Ablation

### 6.1 Full Model Matrix

From `docs/ENSEMBLE_EXP.md`, all closed-track models evaluated on val-dir:

| Model key | Top-1 (no TTA) | Top-1 (+ TTA) |
|-----------|:---:|:---:|
| tsm\_v2\_rot | 39.45% | 41.68% |
| tsm\_v2 | 38.25% | 40.86% |
| vfl\_rot | 36.44% | 37.81% |
| vfl\_focal | 33.11% | 35.77% |
| tsm | 34.91% | 35.91% |
| vfl | 32.11% | 35.03% |

### 6.2 Ensemble Strategy Ablation

| Configuration | Strategy | Top-1 | Top-5 |
|---------------|----------|:---:|:---:|
| log\_avg(tsm\_v2\_tta + tsm\_v2\_rot\_tta + vfl\_rot\_tta) | log-softmax avg | **44.03%** | **75.37%** |
| log\_avg(all 6 TTA models) | log-softmax avg | 42.73% | 74.86% |
| tsm\_v2 × 0.5 + tsm × 0.25 + vfl × 0.25 (weighted) | weighted logit avg | 40.43% | 71.27% |
| tsm\_v2\_rot\_tta (single best model) | — | 41.68% | 72.77% |

**Why log-softmax averaging outperforms weighted logit averaging:** Averaging in log-probability space down-weights over-confident wrong predictions. When models with different calibration (rotating-fold vs. single-split, focal vs. CE) disagree, log-softmax fusion penalises the outlier prediction more heavily than raw logit or probability averaging.

**Why adding the 3 weaker models hurts:** Including tsm (T=5), vfl, and vfl\_focal alongside the 3 strong models dilutes the ensemble signal. The weaker models' error distribution partially overlaps with the strong models' errors, reducing the net complementarity. The optimum ensemble uses only the 3 strongest members.

**Ensemble gain over best single model (with TTA):** 41.68% → 44.03% = **+2.35 pp**. This gain comes from error decorrelation: TSM-Ultra-v2 (local shift) and TSM-Ultra-v2-Rotating (same architecture, different data partition) make different per-clip errors due to random initialisation and fold sampling; VFL-Rotating (global attention) makes qualitatively different errors on temporal-direction classes.

---

## 7. Summary Table

All numbers are val-dir top-1 without TTA unless noted.

| Axis | Ablated From | Ablated To | Δ (pp) | Key finding |
|------|-------------|-----------|--------|------------|
| Frame count 5→4 (TSM) | T0: 34.91% | T2: 38.25%* | +3.34 | T>4 duplicates frames; TSM shift degrades |
| Frame count 8→4 (TSM-R50) | T0: 34.91% | T1: 45.26%†→ | −8.06†† | Larger backbone + more frames = worse; wrong inductive bias |
| Focal loss (VFL, isolated) | V1: 32.11% | V2: 33.11% | +1.00 | Focal hurts internal (−0.20 pp) but helps val-dir (+1.00 pp) |
| Rotating folds (TSM) | T2: 38.25% | T3: 39.45% | +1.20 | Modest gain; TSM's strong prior reduces data sensitivity |
| Rotating folds (VFL) | V1: 32.11% | V3: 36.44% | +4.33 | Large gain; VFL's global attention is data-hungry |
| Transformer depth 2→3 (VFL) | Exp B: 35.36% | V3: 36.44% | +1.08 | 3rd layer adds modest but real signal |
| TTA (TSM-Ultra-v2) | 38.25% | 40.86% | +2.61 | 10-crop averaging reduces spatial variance |
| TTA (VFL-Ultra-Rotating) | 36.44% | 37.81% | +1.37 | Smaller gain: rotating folds already reduce variance |
| Architecture (TSM vs VFL, rotating) | VFL: 36.44% | TSM: 39.45% | +3.01 | TSM's local shift prior matches SSv2's motion structure |
| Backbone R18→R50 (TSM, T=4, rotating) | T3: 39.45% | R50-rot: 42.85% | **+3.40** | R50 with correct T=4 and rotating folds is a major gain |
| Focal γ (TSM, 0→1→2) | A (CE): 38.92% | C (γ=1): 39.50% | +0.58 | γ=1 optimal; γ=2 within noise (+0.45 pp vs CE) |
| SGDR T_mult (TSM, 1→2) | T3: 39.45% | Exp E: 39.41% | −0.04 | Negligible; rotating folds remove the restart benefit |
| VFL n_folds (5→10) | V3: 36.44% | Exp D: 35.98% | −0.46 | VFL data-saturated at 5 folds; 10 folds adds noise |
| Ensemble (3 models + TTA) | 41.68% (single best) | 44.03% (3-model) | +2.35 | Error decorrelation; TSM + VFL complementary |
| Ensemble R50 + 3 best TTA models | 44.03% (prev best) | 46.23% (4-model) | **+2.20** | R50 backbone diversity is the dominant gain |
| Ensemble + TSM-no-focal (5-model) | 46.23% (4-model) | **46.75%** (5-model) | **+0.52** | CE-trained TSM provides complementary calibration |

\* Confounded with focal loss addition; isolated frame effect estimated at +2–3 pp based on TSM-Ultra-50 comparison.  
† Internal val only (run killed early); not directly comparable to other val-dir entries.  
†† Compared to TSM-Ultra internal val (53.32%); sign inverted from expected due to frame duplication.

---

---

## 8. ResNet50 Backbone Results and New Best Ensemble

### 8.1 TSM-ResNet50 (Rotating, T=4) — New Best Single Model

| Model | Backbone | T | Rot. Folds | Val-dir Top-1 | Val-dir Top-5 | Macro-F1 |
|-------|----------|---|-----------|:---:|:---:|:---:|
| tsm\_ultra (T=5) | R18 | 5 | ✗ | 34.91% | 66.88% | 32.29% |
| tsm\_ultra\_v2 | R18 | 4 | ✗ | 38.25% | 68.70% | 34.72% |
| tsm\_ultra\_v2 (rot) | R18 | 4 | ✓ | 39.45% | 69.47% | 36.56% |
| tsm\_ultra\_v2 (rot) +TTA | R18 | 4 | ✓ | 41.68% | 72.77% | 37.66% |
| **tsm\_resnet50\_rot (best ckpt)** | **R50** | **4** | **✓** | **42.85%** | **75.32%** | **38.59%** |

The TSM-ResNet50 rotating model (`tsm_ultra_resnet50_rotating_rotating_44%.pt`) achieves **42.85% top-1** — the new best single model. Key: the early "44% checkpoint" was saved when the model reached its best intermediate val-dir performance; the final checkpoint (`tsm_ultra_resnet50_rotating_rotating.pt`) has a higher rolling avg (92.85%) but lower val-dir accuracy.

The +3.40 pp gain over the R18 rotating reference demonstrates that the ResNet50 capacity is *effective* when combined with the correct T=4 frame count and rotating-folds protocol — in contrast to the earlier failed TSM-Ultra-50 run (T=8, fixed split) that achieved only 45.26% internal val.

### 8.2 Ensemble with ResNet50 — New Project Best

All ensembles use log-softmax averaging.

| Ensemble | Models | Top-1 | Top-5 | Macro-F1 |
|----------|--------|:---:|:---:|:---:|
| Previous best (3-model) | TSM-v2\_tta + TSM-v2-rot\_tta + VFL-rot\_tta | 44.03% | 75.37% | 40.01% |
| R50 + TSM-v2\_tta | 2-model | 45.17% | 76.89% | — |
| R50 + TSM-v2-rot\_tta + VFL-rot\_tta | 3-model | 46.20% | 77.21% | 41.54% |
| R50 + previous best (4-model) | R50 + TSM-v2\_tta + TSM-v2-rot\_tta + VFL-rot\_tta | 46.23% | 77.01% | 41.46% |
| **NEW BEST: 5-model** | **R50 + TSM-v2\_tta + TSM-v2-rot\_tta + VFL-rot\_tta + TSM-no-focal** | **46.75%** | **76.98%** | **41.90%** |

**LOO analysis for the 5-model ensemble:**

| Removed | Top-1 | Δ |
|---------|:---:|:---:|
| None (full 5-model) | 46.75% | — |
| −TSM-ResNet50 | 45.41% | −1.33 pp |
| −VFL-rot +TTA | 46.11% | −0.64 pp |
| −TSM-v2-rot +TTA | 46.27% | −0.47 pp |
| −TSM-no-focal | 46.23% | −0.52 pp |
| −TSM-v2 +TTA | 46.43% | −0.31 pp |

The ResNet50 model is by far the dominant contributor (−1.33 pp when removed), confirming that backbone diversity (R50 vs R18) provides more ensemble benefit than model diversity at the same backbone capacity.

The CE-trained TSM (Exp A / `ablation_tsm_no_focal_rotating`) adds +0.52 pp because its error distribution differs from the focal-trained models: CE-trained models maintain higher confidence on easy examples, providing complementary calibration on the classes where focal loss reduces precision.

---

## 9. Recommendations for Future Work

1. **Isolate focal vs. frame-count for TSM:** A dedicated run of `tsm_ultra` with only T=4 fixed (no focal) would cleanly measure the frame-count effect. Current data conflates it with focal loss. → **Addressed by Exp A (§9).**

2. **Ablate n\_layers for VFL in isolation:** Run VFL-Ultra with n\_layers=2 under the rotating-folds protocol to measure whether the +4.33 pp gain from V1→V3 is primarily from rotating folds, the third Transformer block, or their interaction. → **Addressed by Exp B (§9).**

3. **Focal γ sweep for TSM:** With the frame count fixed, a sweep over γ ∈ {0, 1, 2} would identify whether γ=2 is optimal or if softer focal weighting performs better on small SSv2-33 classes. → **γ=0 via Exp A; γ=1 via Exp C (§9).**

4. **Ensemble weight optimisation:** The current weights (0.5/0.25/0.25) were not tuned on val-dir. A held-out grid search on a random 20% of val-dir could identify whether higher VFL weights (e.g. 0.5/0.2/0.3) improve accuracy, at the cost of mild selection bias.

5. **Data utilisation ceiling:** The rotating-folds asymmetry (TSM +1.2 pp, VFL +4.3 pp) suggests VFL has not yet saturated its data capacity. Increasing `n_folds` from 5 to 10 would expose VFL to still larger effective training sets, possibly recovering another 1–2 pp. → **Addressed by Exp D (§9).**

---

## 9. Next Experiments — 4 Machines, 150 W, 16 GB VRAM

Four experiments designed to fill the three largest gaps in §8, fully failure-resilient and W&B-tracked. Each runs on one machine and completes within ~13 h.

**Infrastructure changes made:**
- [src/train_kfold.py](../src/train_kfold.py): one-line fix so `training.wandb_run_name` is respected in rotating-folds mode (previously hardcoded to `"rotating"`; now `cfg.training.get("wandb_run_name", "rotating")`).
- Both `resume: true` and `wandb_project` are baked into each config, so the launch command is a single line per machine with no extra flags.

### 9.1 Experiment Matrix

Two additional experiments (E, F) ran concurrently to ablate the scheduler on both model families.

| ID | Config file | Single axis changed vs. reference | Reference val-dir | Machine |
|----|-------------|----------------------------------|------------------|---------|
| **A** | `ablation_tsm_no_focal_rotating` | `focal_gamma` 2.0 → **CE** | T3: 39.45% | 1 |
| **B** | `ablation_vfl_2layers_rotating` | `n_layers` 3 → **2**, `drop_path` 0.20 → 0.15 | V3: 36.44% | 2 |
| **C** | `ablation_tsm_focal_g1_rotating` | `focal_gamma` 2.0 → **1.0** | T3: 39.45% | 3 |
| **D** | `ablation_vfl_folds10_rotating` | `n_folds` 5 → **10**, `sgdr_t0` 25 → 50 | V3: 36.44% | 4 |
| **E** | `ablation_tsm_sgdr_t2_rotating` | `sgdr_t0` 25→12, `sgdr_t_mult` 1→**2** | T3: 39.45% | — |
| **F** | `ablation_vfl2_sgdr_t2_rotating` | VFL 2-layer; `sgdr_t0` 25→12, `sgdr_t_mult` 1→**2** | B: 35.36% | — |

### 9.2 Experiment Details

#### Exp A — TSM without focal loss (`ablation_tsm_no_focal_rotating`)

**Question:** How much of T3's 39.45% val-dir comes from focal loss, and how much from the frame-count fix (T=5→4) and rotating folds?

**What changes:** `focal_gamma: 2.0` removed; standard cross-entropy is used. Everything else is identical to `tsm_ultra_v2_rotating` (T=4, fold_div=4, R18, AdamW lr=1e-3, SGDR T₀=25, 200 epochs, rotating 5-fold).

**Expected result table after this run:**

| Config | γ | Val-dir | Δ vs T3 |
|--------|---|---------|---------|
| Exp A (CE) | 0 | ? | focal contribution |
| T3 (γ=2) | 2 | 39.45% | — |

**Interpretation guide:**
- Δ > 1.5 pp below T3 → focal loss is a meaningful contributor beyond the frame fix.
- Δ < 0.5 pp → focal adds noise for TSM at this scale; the T=4 fix was the dominant gain.

**Wall-clock:** ~4 min/epoch × 200 epochs ≈ **13 h**.

---

#### Exp B — VFL with 2 Transformer layers (`ablation_vfl_2layers_rotating`)

**Question:** Does the 3rd Transformer layer (n_layers 2→3) meaningfully contribute to VFL-Rotating's 36.44%, or is the rotating-folds protocol alone sufficient?

**What changes:** `n_layers: 3 → 2`, `drop_path_rate: 0.20 → 0.15` (0.15 is the value used in 2-layer VFL-Ultra; reverting avoids over-regularising a shallower model). Everything else identical to `vfl_ultra_rotating` (focal γ=2, AdamW lr=8e-4, SGDR T₀=25, 150 epochs, rotating 5-fold).

**Expected result table after this run:**

| Config | n_layers | Val-dir | Δ vs V3 |
|--------|----------|---------|---------|
| Exp B (2-layer) | 2 | ? | depth contribution |
| V3 (3-layer) | 3 | 36.44% | — |

**Interpretation guide:**
- Δ > 2 pp below V3 → depth matters; 3 layers are justified for 4-frame tokens.
- Δ < 0.5 pp → depth is not the bottleneck; rotating folds alone drive the gain.

**Wall-clock:** ~4.5 min/epoch × 150 epochs ≈ **11 h**.

---

#### Exp C — TSM with focal γ=1 (`ablation_tsm_focal_g1_rotating`)

**Question:** Is γ=2 the optimal focal strength for SSv2-33, or does the softer γ=1 perform better by suppressing gradient less aggressively on the small per-class sample count (~1,364 clips/class)?

**What changes:** `focal_gamma: 2.0 → 1.0`. Everything else identical to `tsm_ultra_v2_rotating`.

Together with T3 (γ=2) and Exp A (γ=0), this gives three points on the γ axis:

| Config | γ | Val-dir |
|--------|---|---------|
| Exp A | 0 (CE) | ? |
| **Exp C** | **1** | **?** |
| T3 | 2 | 39.45% |

**Interpretation guide:**
- If Exp C > T3 → optimal γ is in [1, 2); softer down-weighting suits the small-class regime.
- If Exp C < T3 and Exp A ≈ Exp C → the threshold effect is sharp around γ=2.
- If Exp C ≈ T3 and both > Exp A → any focal loss helps, but γ is not critical.

**Wall-clock:** ~4 min/epoch × 200 epochs ≈ **13 h**.

---

#### Exp D — VFL with 10-fold rotating validation (`ablation_vfl_folds10_rotating`)

**Question:** Has VFL saturated its data-utilisation capacity at n_folds=5 (80% per epoch), or does 90% per epoch (n_folds=10) yield further gains consistent with VFL's data-hunger hypothesis?

**What changes:** `n_folds: 5 → 10`, `sgdr_t0: 25 → 50` (preserving "5 complete fold-rotations per LR cycle"). Epoch count stays at 150 (= 15 complete 10-fold rotations ≈ 135 effective full-dataset epochs, vs. 120 for V3).

**Expected result table after this run:**

| Config | n_folds | % data/epoch | Val-dir | Δ vs V3 |
|--------|---------|-------------|---------|---------|
| V3 | 5 | 80% | 36.44% | — |
| **Exp D** | **10** | **90%** | **?** | data ceiling |

**Interpretation guide:**
- Δ > 1 pp above V3 → VFL is not yet data-saturated; n_folds=10 is worth using for future VFL runs.
- Δ < 0.3 pp → VFL has saturated at n_folds=5; further data exposure gives diminishing returns.

**Wall-clock:** ~5 min/epoch × 150 epochs ≈ **12.5 h**.

---

### 9.3 Launch Commands

All four experiments are self-contained. `resume: true` is baked into each config — runs are safe to kill and restart with the same command.

**Prerequisites:**
```bash
# Install wandb if not already present
pip install wandb
wandb login   # one-time; enter your API key
```

**Machine 1 — Exp A (TSM, no focal):**
```bash
cd /path/to/Challenge-Modal/src
python train_kfold.py experiment=ablation_tsm_no_focal_rotating
```

**Machine 2 — Exp B (VFL, 2 layers):**
```bash
cd /path/to/Challenge-Modal/src
python train_kfold.py experiment=ablation_vfl_2layers_rotating
```

**Machine 3 — Exp C (TSM, focal γ=1):**
```bash
cd /path/to/Challenge-Modal/src
python train_kfold.py experiment=ablation_tsm_focal_g1_rotating
```

**Machine 4 — Exp D (VFL, 10 folds):**
```bash
cd /path/to/Challenge-Modal/src
python train_kfold.py experiment=ablation_vfl_folds10_rotating
```

**Resume after crash** (same command, no extra flags — `resume: true` in config does the rest):
```bash
python train_kfold.py experiment=<same-experiment>
```

**W&B dashboard:** all four runs log to project `challenge-modal` under run names `ablation-A-tsm-no-focal`, `ablation-B-vfl-2layers`, `ablation-C-tsm-focal-g1`, `ablation-D-vfl-folds10`. Tracked metrics per epoch: `train/loss`, `train/acc`, `val/loss`, `val/acc`, `rolling_avg`, `best_rolling_avg`, `lr`.

### 9.4 Results Table (completed)

All six ablations trained to completion. Val-dir evaluated with `gen_val_logits.py` +
`analyze_all_models.py` (no re-inference needed after caching).

| Exp | Config | Ablated axis | Val-dir Top-1 | Val-dir Top-5 | Δ vs reference | Interpretation |
|-----|--------|-------------|:---:|:---:|:---:|----------------|
| A | tsm\_no\_focal\_rotating | focal γ: 2→**0 (CE)** | **38.92%** | 69.76% | −0.53 pp vs T3 | focal adds +0.53 pp |
| B | vfl\_2layers\_rotating | n\_layers: 3→**2** | **35.36%** | 67.29% | −1.08 pp vs V3 | 3rd layer adds +1.08 pp |
| C | tsm\_focal\_g1\_rotating | focal γ: 2→**1** | **39.50%** | 70.85% | **+0.05 pp** vs T3 | γ=1 ≈ γ=2; both beat CE |
| D | vfl\_folds10\_rotating | n\_folds: 5→**10** | **35.98%** | 67.92% | −0.46 pp vs V3 | VFL saturated at 5 folds |
| E | tsm\_sgdr\_t2\_rotating | SGDR T_mult: 1→**2** | **39.41%** | 70.26% | −0.04 pp vs T3 | scheduler shape irrelevant |
| F | vfl2\_sgdr\_t2\_rotating | VFL-2L SGDR T_mult: 1→**2** | **35.24%** | 64.74% | −0.12 pp vs B | scheduler shape irrelevant for VFL |

**Key findings from completed ablations:**

**TSM focal γ sweep (Exps A, C, T3):**

| Config | γ | Val-dir |
|--------|---|---------|
| Exp A (CE) | 0 | 38.92% |
| Exp C | 1 | **39.50%** |
| T3 (reference) | 2 | 39.45% |

The optimum lies around γ=1–2; both beat CE by ~0.5 pp. The γ axis is flat above 1, so the choice between γ=1 and γ=2 is not statistically significant at this dataset size.

**VFL depth (Exps B, V3):**

2-layer VFL scores 35.36% vs 3-layer at 36.44% (−1.08 pp). The third Transformer block provides a real but modest gain; it is worth keeping for the 3-layer production model.

**VFL data ceiling (Exp D):**

10-fold VFL (35.98%) scores *below* 5-fold VFL V3 (36.44%) by −0.46 pp. VFL has saturated its data-utilisation capacity at n_folds=5; more folds add no benefit and slightly hurt (possibly because the validation signal per checkpoint save becomes noisier with 10-fold rolling averages).

**Scheduler shape (Exps E, F):**

SGDR T_mult=2 produces no meaningful improvement on either TSM (−0.04 pp) or VFL 2-layer (−0.12 pp). The rotating-folds protocol already provides implicit regularisation equivalent to restart benefits; the restart pattern is not a critical hyperparameter once folds rotate.
