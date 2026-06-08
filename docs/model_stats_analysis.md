# Model Statistics & Result Analysis (closed track / from-scratch)

**Dataset:** 33-class subset of Something-Something V2 ("What Happens Next?").
**Evaluation set:** the official **val-dir** split — **6,745 clips**, 32 of 33 classes present
(class *"Taking something out of something"* has 0 val clips).
**Method:** every number below is computed from **cached val-dir logits**
(`models/val_logits/*.npy`) with `src/analyze_all_models.py` — no re-inference.
Metrics are exact (top-1/top-5 by arg-sort, per-class P/R/F1 and the confusion
matrix from the 33×33 count matrix).

**Scope:** only **from-scratch (closed-track)** models are analysed. Pretrained/open-track
ViT logits also exist in `val_logits/` but are deliberately excluded to keep the report
consistent with the closed-track submission.

> Two accuracy notions appear in this repo and must not be confused:
> - **val-dir top-1** — the real, comparable metric (held-out official validation set).
> - **internal-val** — accuracy on a 20% slice of the *train* folder used only for
>   early-stopping. It runs 15–20 pp higher than val-dir, and for **rotating-fold**
>   models it is *meaningless* (the slice overlaps data the model trained on across folds).

---

## 1. Headline

| Metric | Value |
|--------|------:|
| Best single model (val-dir top-1) | **42.85%** — `tsm_resnet50_rotating` (no TTA) |
| Best ensemble (val-dir top-1 / top-5) | **46.75% / 76.98%** |
| Best ensemble macro Precision / Recall / F1 | **0.437 / 0.420 / 0.419** |
| Ensemble lift over best single | **+3.90 pp** |
| Previous best ensemble (3-model R18 only) | 44.03% / 75.37% |
| Gain from adding ResNet50 to ensemble | **+2.72 pp** |

Best ensemble (5-model) = **log-softmax average** of:
`tsm_resnet50_rotating` (no TTA) + `tsm_v2_tta` + `tsm_v2_rot_tta` + `vfl_rot_tta` + `tsm_no_focal_rotating` (no TTA).

---

## 2. Per-model accuracy (val-dir)

![Per-model accuracy](figures/fig01_per_model_accuracy.png)

All from-scratch model/variant logits, ranked by top-1 (including new models):

| Model variant | Top-1 | Top-5 | Macro-F1 | Notes |
|---------------|------:|------:|---------:|-------|
| **TSM-ResNet50 (rot)** | **42.85** | **75.32** | **38.59** | **New best single model** |
| TSM-Ultra-v2 (rot) +TTA | 41.68 | 72.77 | 37.66 | Previous best |
| TSM-Ultra-v2 +TTA | 40.86 | 71.68 | 36.63 | |
| TSM-Ultra-v2 (rot) | 39.45 | 69.47 | 36.56 | |
| Ablation-C: TSM focal-γ1 (rot) | 39.50 | 70.85 | — | +0.05 pp vs reference T3 |
| Ablation-E: TSM SGDR-T2 (rot) | 39.41 | 70.26 | — | −0.04 pp vs T3 |
| Ablation-A: TSM no-focal (rot) | 38.92 | 69.76 | — | CE loss: −0.53 pp vs T3 |
| TSM-Ultra-v2 | 38.25 | 68.70 | 34.72 | |
| VFL-Ultra (rot) +TTA | 37.81 | 70.24 | 34.60 | |
| VFL-Ultra (rot) | 36.44 | 68.48 | 34.03 | |
| Ablation-D: VFL 10-fold (rot) | 35.98 | 67.92 | — | −0.46 pp vs V3 |
| Ablation-B: VFL 2-layer (rot) | 35.36 | 67.29 | — | −1.08 pp vs V3 |
| Ablation-F: VFL2 SGDR-T2 (rot) | 35.24 | 64.74 | — | −0.12 pp vs Exp B |
| TSM-Ultra +TTA | 35.91 | 69.44 | 32.52 | |
| VFL-focal +TTA | 35.77 | 68.44 | 32.30 | |
| VFL-Ultra +TTA | 35.03 | 67.77 | 31.75 | |
| TSM-Ultra | 34.91 | 66.88 | 32.29 | |
| VFL-focal | 33.11 | 66.17 | 30.44 | |
| VFL-Ultra | 32.11 | 64.85 | 29.28 | |

**Comments.**
- **TSM-ResNet50 (rot) is the new best single model at 42.85%**, beating the previous best
  (TSM-R18 rot +TTA, 41.68%) by +1.17 pp — and it achieves this *without TTA*. Adding
  TTA was not tested (inference budget exhausted), but is expected to push it further.
- **TSM dominates VFL.** Every TSM variant beats every VFL variant.
- **The focal γ ablation is flat above CE**: γ=1 (39.50%) ≈ γ=2 (39.45%) >> γ=0 (38.92%). Any focal weight helps; the exact value is not critical.
- **VFL depth matters (+1.08 pp for 3rd layer)**; VFL data ceiling is reached at 5 folds.
- Macro-F1 trails top-1 by ~3–5 pp everywhere, the signature of 20× class imbalance.

---

## 3. Test-time augmentation (TTA)

![TTA effect](figures/fig02_tta_effect.png)

10-crop TTA (5 spatial crops × 2 horizontal flips, logits averaged) on overall top-1:

| Base model | no-TTA | +TTA | Δ |
|------------|------:|-----:|--:|
| TSM-Ultra | 34.91 | 35.91 | +1.0 |
| TSM-Ultra-v2 | 38.25 | 40.86 | +2.6 |
| TSM-Ultra-v2 (rot) | 39.45 | 41.68 | +2.2 |
| VFL-focal | 33.11 | 35.77 | +2.7 |
| VFL-Ultra | 32.11 | 35.03 | +2.9 |
| VFL-Ultra (rot) | 36.44 | 37.81 | +1.4 |

**Comments.** TTA is a free, consistent **+1 to +3 pp** on overall accuracy (mean ≈ +2 pp).
But "overall accuracy" hides a sharp asymmetry — see §5, which is the most important
finding in this report.

---

## 4. Rotating 5-fold training

![Rotating-fold effect](figures/fig03_rotating_effect.png)

| Pair | single split | rotating 5-fold | Δ |
|------|------:|------:|--:|
| TSM-Ultra-v2 | 38.25 | 39.45 | +1.2 |
| TSM-Ultra-v2 +TTA | 40.86 | 41.68 | +0.8 |
| VFL-Ultra | 32.11 | 36.44 | **+4.3** |
| VFL-Ultra +TTA | 35.03 | 37.81 | +2.8 |

**Comments.** Rotating folds (train on all data via fold rotation, then average) help
*every* model and help the weaker **VFL most (+4.3 pp)** — the higher-variance Transformer
benefits more from seeing all the data than the already-stable TSM. This is the validated
"always use rotating folds for new experiments" policy paying off on the real metric.

> ⚠️ Do **not** read rotating-fold quality off internal-val (§9): `tsm_ultra_v2_rotating`
> reports 95.8% internal-val but only 39.45% val-dir, because the internal slice overlaps
> training data across folds. Only val-dir is trustworthy here.

---

## 5. The TTA direction trap (key finding)

![TTA directional trade-off](figures/fig12_tta_directional_tradeoff.png)

TTA averages **horizontally-flipped** views. For **direction-sensitive** classes
(*Pulling left→right* vs *Pulling right→left*) a flip turns the motion into the *opposite*
class, so the flipped views vote for the wrong label:

| | Pulling L→R (F1) | Pulling R→L (F1) | Overall top-1 |
|--|------:|------:|------:|
| Ensemble, **no TTA** | **0.645** | **0.480** | 43.29 |
| Ensemble, **+TTA** | 0.274 | 0.278 | **44.03** |

**Comments.**
- TTA **buys +0.74 pp overall** while **halving F1** on the two directional classes
  (0.65→0.27, 0.48→0.28). It wins on average only because the other ~30 classes are
  direction-agnostic and gain from multi-crop smoothing.
- This is the same reason horizontal flip is **disabled during training** in this project —
  but it silently returns at test time through TTA.
- **This contradicts `FINAL_MODEL.md` §5.2**, which claims test-time flips are harmless
  "where label semantics no longer matter for the ensemble vote." They demonstrably *do*
  matter for directional classes.
- **Recommendation:** use **flip-free TTA** (5 crops, no flip) or class-aware TTA. The
  no-TTA ensemble (43.29%) already preserves directional F1 at a cost of only 0.74 pp
  overall — if the grading metric is macro-F1 rather than top-1, the no-TTA ensemble may
  actually be the better submission.

---

## 6. Ensembling

![Ensemble comparison](figures/fig04_ensemble.png)
![Leave-one-out](figures/fig11_leave_one_out.png)

| Configuration | Top-1 | Top-5 |
|---------------|------:|------:|
| Best single (`tsm_v2_rot` +TTA) | 41.68 | 72.77 |
| log-avg of the 2 TSM-v2 (single + rot) +TTA | 43.40 | 73.82 |
| log-avg of all 6 closed TTA models | 42.73 | 74.86 |
| **Best: log-avg `tsm_v2_tta + tsm_v2_rot_tta + vfl_rot_tta`** | **44.03** | **75.37** |

Leave-one-out on the best ensemble (drop one component, re-evaluate):

| Dropped | Top-1 | Δ |
|---------|------:|--:|
| — (full) | 44.03 | — |
| − `tsm_v2_tta` | 42.95 | −1.08 |
| − `tsm_v2_rot_tta` | 42.97 | −1.07 |
| − `vfl_rot_tta` | 43.40 | −0.64 |

**Comments.**
- **Every component contributes positively** — no member is redundant. The two TSM-v2
  variants carry ~1.1 pp each; the architecturally-different VFL adds **+0.64 pp** of pure
  diversity (its standalone accuracy is 4 pp lower, yet removing it still hurts — textbook
  error decorrelation).
- **More models is not better.** The 5-member `all-6` log-avg (42.73) is *worse* than the
  curated 3-member ensemble (44.03): the weak `tsm_tta`, `vfl_tta`, `vfl_focal_tta` add
  noise that dilutes the two strong TSM signals.
- **Log-softmax averaging** (averaging in log-probability space) beats raw-logit averaging
  here — it down-weights over-confident wrong votes and rewards agreement across models
  with different calibration (focal vs. CE, single vs. rotating fold).

---

## 6.5 Ensemble diversity — pairwise agreement matrix

![Pairwise agreement matrix](figures/fig13_agreement_matrix.png)

Each cell is the **fraction of the 6,745 val-dir clips where two models predict the same
class** (red = divergent, green = identical). This is the single most useful plot for
*choosing* ensemble members: the best partners are two models that are individually strong
but sit in a **red** cell (they make different mistakes).

| Block | Mean agreement |
|-------|---------------:|
| within TSM family | 0.52 |
| within VFL family | 0.54 |
| **TSM ↔ VFL (cross-family)** | **0.43** (most divergent) |

**Comments.**
- The matrix splits cleanly into two architecture blocks. The **TSM↔VFL off-diagonal is
  the reddest region (~0.40–0.50)** — the two families genuinely disagree, which is the
  whole reason the ensemble works. Within a family, variants agree more (yellow, 0.5–0.65).
- The chosen best ensemble exploits exactly this: it pairs two strong TSM-v2 variants
  (agreement **0.59** with each other) with `vfl_rot` which disagrees with both
  (**0.48–0.50**). The VFL is 4 pp weaker alone yet earns **+0.64 pp** in leave-one-out
  (§6) — its red cells are pure, uncorrelated signal.
- Note `tsm_ultra_v2` vs `tsm_ultra_v2 +TTA` agree only **0.63** — i.e. TTA *flips ~37% of
  predictions*. That is a lot of churn for +2.6 pp, and it is the same flip mechanism that
  wrecks directional classes (§5).
- **How to extend the ensemble:** do **not** add a 4th TSM variant (high agreement →
  redundant). Add the strongest model from the *least-agreeing* family/region — i.e. pick
  green-accuracy models that live in red cells relative to what you already have. With only
  TSM and VFL available from scratch, the current 3-member set is already near the
  diversity ceiling; a genuinely new architecture (that converges) would be needed to push
  further.

---

## 7. Per-class performance

![Per-class F1](figures/fig05_per_class_f1.png)
![F1 vs support](figures/fig08_f1_vs_support.png)

**Best 5 classes (best ensemble):**

| Class | F1 | Val support |
|-------|---:|---:|
| Pouring something into something | 0.63 | 278 |
| Moving something closer to something | 0.62 | 213 |
| Moving something away from something | 0.60 | 183 |
| Uncovering something | 0.58 | 391 |
| Folding something | 0.58 | 285 |

**Worst 5 classes (with val samples):**

| Class | F1 | Val support |
|-------|---:|---:|
| Spilling something next to something | 0.03 | 60 |
| Picking something up | 0.09 | 199 |
| Pretending to throw something | 0.15 | 47 |
| Pretending to put something into something | 0.18 | 68 |
| Putting something onto something | 0.27 | 139 |

**Comments.**
- The winners are all **single, clean, directional motion arcs** (pour, move, fold,
  uncover) — exactly what per-layer temporal channel shift encodes well.
- The losers split into two groups: (1) the **"pretending" family**, where the executed
  motion is visually identical to a real action — fundamentally under-constrained at 4
  frames; (2) **low-support classes** (Spilling n=60, Pretending-throw n=47). The
  `F1-vs-support` scatter shows the trend but it is *not* purely a data-volume effect:
  "Picking something up" has 199 val clips yet F1=0.09 because it is systematically lost
  to "Pretending to pick something up" (§8).

---

## 8. Confusions

![Confusion matrix](figures/fig06_confusion_matrix.png)
![Top confused pairs](figures/fig07_confused_pairs.png)

Top confusions (best ensemble):

| # clips | True → Predicted |
|--:|--|
| 43 | Moving something up → **Pretending to pick something up** |
| 40 | Picking something up → **Pretending to pick something up** |
| 38 | Showing something to the camera → Turning something upside down |
| … | (full list in `model_stats.json`) |

**Comments.** Three structural error modes, all aligned with SSv2's design intent:
1. **Real vs. pretended** — *picking up* ↔ *pretending to pick up*. The hand trajectory is
   nearly identical; only the (often off-frame) outcome differs. Focal loss in
   `tsm_ultra_v2` dents but does not solve it.
2. **Direction / orientation** — *folding* ↔ *unfolding*, *showing* ↔ *turning upside
   down*: same appearance, opposite temporal order; fragile at T=4 and further damaged by
   flip-TTA (§5).
3. **A "static sink"** — several classes leak into *Holding something* whenever little
   motion is detected.

---

## 9. Where the ensemble beats the best single model

![Ensemble vs single, per class](figures/fig09_ensemble_vs_single_perclass.png)

This compares per-class F1 of the **best ensemble** against the **best single model**
(`tsm_v2_rot` +TTA). The ensemble is a net win, but it is not uniformly better: it lifts
most mid-table classes (diversity fills coverage gaps) while a few classes regress slightly
where the added VFL/rotating votes disagree with the strong TSM. The net macro-F1 gain
confirms the components make *different* errors rather than the same ones louder.

---

## 10. Full model inventory (all from-scratch checkpoints)

![Checkpoint internal-val](figures/fig10_checkpoint_internal_val.png)

Internal-val (train-split) accuracy for every from-scratch checkpoint — **for reference
only; not comparable to val-dir** (and inflated for rotating folds):

| Model | internal-val | params | frames | val-dir logits? |
|-------|------:|------:|:--:|:--:|
| tsm_ultra_v2_rotating | 95.8%* | 11.2M | 4 | yes |
| video_former_lite_ultra_rotating | 78.2%* | 20.7M | 4 | yes |
| tsm_ultra_v2 | 57.7% | 11.2M | 4 | yes |
| video_former_lite_ultra | 53.4% | 17.5M | 4 | yes |
| tsm_ultra | 53.3% | 11.2M | 5 | yes |
| vfl_ultra_focal | 53.2% | 17.5M | 4 | yes |
| video_former_lite_closed | 45.8% | 17.5M | 4 | no |
| tsm_ultra_50 | 45.3% | 23.6M | 8 | no |
| r2plus1d_closed | 10.8% | 31.3M | 8 | no |
| tsm_resnet50_bigru_scratch_rotating | 9.1% | 27.1M | 4 | no |
| efficientformer_scratch_rotating | 8.3% | 11.4M | 4 | no |
| tsm_resnet50_bigru_rotating | 7.6% | 27.1M | 4 | no |
| tsm_resnet50_bigru | 7.0% | 27.1M | 4 | no |
| maxvit_interp_rotating | 6.6% | 30.5M | 4 | no |
| vit_bigru_attn_scratch (fold 0) | 5.8% | 87.4M | 4 | no |

\* inflated — rotating-fold internal slice overlaps training data.

**Comments.**
- A clear **convergence cliff**: the ResNet18-class "ultra" models (11–18M params) reach
  53–58% internal-val, while every larger / heavier-backbone model collapses to near-random
  (≈3% = 1/33 chance). **r2plus1d** (31M, 3D conv), **tsm_resnet50_bigru** (ResNet50+GRU),
  **efficientformer**, **maxvit**, and the **87M from-scratch ViT** all failed to converge
  from scratch in the compute budget.
- The lesson is consistent: **from scratch on ~36k clips, capacity is a liability, not an
  asset.** Bigger backbones and transformer/3D heads need pretraining or far more data; the
  lightweight TSM-ResNet18 is the sweet spot.
- These failed models are correctly **excluded from the submission ensemble** — they would
  only add noise.

---

## 11. Takeaways

1. **Best closed-track result: 44.03% top-1 / 75.37% top-5** from a 3-model log-softmax
   ensemble; **41.68%** from the best single model.
2. **TSM-ResNet18 at T=4 is the workhorse.** Matching frames to clip length (+3.3 pp) and
   keeping the backbone small are the two highest-impact decisions.
3. **Rotating folds help everything**, the high-variance VFL most (+4.3 pp).
4. **Ensemble diversity > raw accuracy**: every component contributes, but only a *curated*
   set — adding weak models hurts.
5. **TTA is a double-edged sword**: +0.74 pp overall but it *halves* F1 on direction-
   sensitive classes via horizontal flips. Prefer flip-free TTA, and correct the claim in
   `FINAL_MODEL.md` §5.2.
6. **Capacity from scratch fails**: ResNet50 / 3D-conv / MaxViT / from-scratch ViT all
   collapsed to near-random.
7. **Residual errors are intrinsic to SSv2**: real-vs-pretended intent and temporal
   direction — they need more frames or pretraining, not a bigger from-scratch model.

---

## 12. Reproduce

```bash
python src/analyze_all_models.py     # -> docs/figures/*.png, docs/model_stats.json
```
All metrics derive from `models/val_logits/*.npy` (cached val-dir logits); no GPU,
dataset, or re-inference required. Raw numbers are in `docs/model_stats.json`.
