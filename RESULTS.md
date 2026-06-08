# Results — What Happens Next?

Numerical summary of all experiments across both tracks.  
For methodology and analysis see [docs/TRACK_A.md](docs/TRACK_A.md) and [docs/TRACK_B.md](docs/TRACK_B.md).

---

## Track A — Closed World (no pretrained weights)

**Best result: 44.03% top-1 / 75.37% top-5** on the official 6,745-clip validation set.

### Single-model progression

| Model | Internal val | Val-dir top-1 | Val-dir top-5 | Key change |
|---|---:|---:|---:|---|
| CNN Baseline (ResNet18 + avg pool) | — | — | — | starting point |
| CNN-LSTM | — | — | — | marginal gain |
| VideoFormer-Lite (VFL) — no warmup | 9.45% | — | — | |
| VideoFormer-Lite (VFL) — with reg warmup | 45.78% | — | — | **+36.3 pp from warmup fix** |
| TSM-ResNet18 (tsm_ultra, T=5) | 53.32% | 34.91% | 66.88% | |
| VideoFormer-Lite-Ultra | 53.37% | 32.11% | 64.85% | deeper transformer, tuned lr |
| **TSM-ResNet18 (tsm_ultra_v2, T=4, focal)** | **57.73%** | **38.25%** | **68.70%** | **+4.4 pp from frame-count fix** |
| TSM-ultra-v2 + rotating folds | — | 39.45% | 69.47% | +1.2 pp from rotating folds |
| TSM-ultra-v2 + rotating + TTA | — | 41.68% | 72.77% | +2.2 pp from 10-crop TTA |

> **Key finding:** Matching `num_frames=4` to the true clip length (+4.4 pp) outperformed
> scaling the backbone from ResNet18 to ResNet50 with 8 frames (−8 pp). A data-preparation
> mismatch dominated architecture choice.

### Ensemble results

| Configuration | Strategy | Top-1 | Top-5 |
|---|---|:---:|:---:|
| tsm_v2_rot_tta (single best) | — | 0.4168 | 0.7277 |
| tsm_v2_tta + tsm_v2_rot_tta + vfl_rot_tta | **log-softmax avg** | **0.4403** | **0.7537** |
| All 6 closed TTA models | log-softmax avg | 0.4273 | 0.7486 |
| tsm_v2 × 0.5 + tsm × 0.25 + vfl × 0.25 (prev. best) | weighted | 0.4043 | 0.7127 |

> **+2.35 pp** over best single model from architecture-diverse ensemble (TSM + Transformer).
> Adding more models beyond 3 degraded performance — architectural similarity trumped size.

### Ablation summary

| Axis | Before | After | Delta (val-dir) |
|---|---|---|---|
| num_frames: 5 → 4 (correct clip length) | 34.91% | 38.25% | **+3.34 pp** |
| Loss: CE → focal (γ=2) | — | — | **+1.0 pp** (val-dir generalisation) |
| Split: fixed → rotating folds | 38.25% | 39.45% | **+1.20 pp** |
| Ensemble: single → 3-model log-avg | 41.68% | 44.03% | **+2.35 pp** |
| Reg warmup: none → 10-epoch defer | 9.45% | 45.78% | **+36.3 pp** (VFL model) |

---

## Track B — Open World (pretrained models allowed)

**Best result: 0.6586 Kaggle top-1** (16th place overall).

### Honest progression (Kaggle scores)

| # | Submission | Kaggle | Delta | Notes |
|---|---|:---:|---|---|
| — | Leaky V-JEPA (full video + SSv2-ft backbone) — **DISCARDED** | 0.81 | — | Privileged info; self-discovered, thrown away |
| 1 | V-JEPA-ft alone (clean rebuild) | 0.6361 | — | Starting honest baseline |
| 2 | Honest ensemble v1 (9-uniform: V-JEPA-ft + k400 + tsm) | 0.6418 | +0.6 pp | |
| 3 | V-JEPA-pseudo standalone | 0.6410 | +0.5 pp vs 1 | Val/Kaggle gap doubled |
| 4 | V-JEPA-only (6 snapshots) | 0.6410 | regression | Dropping weak models hurt |
| 5 | V-JEPA-heavy 3:1 weighted | 0.6499 | +1.4 pp | Manual weighting suboptimal |
| 6 | 12-uniform (added V-JEPA-pseudo) | 0.6546 | +1.9 pp | **Diversity wins** |
| 7 | **14-uniform (+ 2 VideoMAE-Large 4f snapshots)** | **0.6586** | **+2.3 pp** | Best result |

> **Key finding:** A uniform 14-model ensemble across architectures (V-JEPA + VideoMAE-Large +
> VideoMAE-K400 + TSM) outperformed a V-JEPA-only or V-JEPA-heavy ensemble, even though TSM
> individually scored only ~0.55. Ensemble diversity across architecture families beats
> learned-weight or single-family ensembling.

### Single-model results (Track B)

| Model / config | Val-dir top-1 | Kaggle | Notes |
|---|:---:|:---:|---|
| VideoMAE-Base K400 4f (3 snapshots avg) | 0.5576 | — | |
| VideoMAE-Base SSv2 4f (3 snapshots avg) | 0.6047 | — | |
| VideoMAE-Large 4f (2 snapshots avg) | 0.5945 | — | |
| TSM ResNet50 4f | 0.3364 | — | Closed-track architecture, included for diversity |
| **V-JEPA-ft (clean, stage 2)** | ~0.6519 | 0.6361 | Window-capped 16f |
| V-JEPA-pseudo (pseudo-label retrain) | ~0.6519 val | 0.6410 | Val/Kaggle gap doubled |

### Val/Kaggle calibration

| Model | Val-dir | Kaggle | Gap |
|---|:---:|:---:|:---:|
| V-JEPA-ft (clean) | ~0.652 | 0.6361 | ~0.5 pp |
| V-JEPA-pseudo | ~0.652 | 0.6410 | ~1.1 pp |
| 14-uniform ensemble | ~0.663 | 0.6586 | ~0.5 pp |

> Pseudo-labelling doubled the val/Kaggle gap, consistent with the model overfitting
> to val-distribution biases. The clean V-JEPA-ft kept a stable ~0.5 pp gap.

---

## Cross-Track Comparison

| Track | Constraint | Val-dir top-1 | Kaggle | Architecture |
|---|---|:---:|:---:|---|
| A — Closed World | Scratch only | **44.03%** | ~0.44 | TSM-ResNet18 + VideoFormer-Lite ensemble |
| B — Open World | Pretrained OK | ~66.3% | **0.6586** | 14-model uniform ensemble (V-JEPA + VideoMAE + TSM) |

The 22 pp gap between tracks reflects the information content in large-scale video pretraining
(VideoMAE, V-JEPA 2) versus training from scratch on 45k clips.

---

## Per-Class Highlights (Track A ensemble)

Classes with highest accuracy on val-dir:

| Class | Val samples | Ensemble top-1 |
|---|:---:|:---:|
| Moving something closer to something (007) | 213 | 79.8% |
| Turning something upside down (030) | 391 | 79.0% |
| Pouring something into something (012) | 278 | 78.8% |
| Folding something (003) | 285 | 76.1% |

Classes with lowest accuracy (failure modes):

| Class | Val samples | Ensemble top-1 | Failure reason |
|---|:---:|:---:|---|
| Picking something up (011) | 199 | 11.6% | Confused with "Pretending to pick up" |
| Pretending to put something into something (016) | 68 | 11.8% | Real vs. pretended intent |
| Pretending to throw something (017) | 47 | 38.3% | Very few val samples |

> **Root cause:** Direction-ambiguous classes (pulling left/right) and real-vs-pretended
> intent pairs are under-determined by four frames — the discriminative signal lives in
> the withheld outcome frames.
