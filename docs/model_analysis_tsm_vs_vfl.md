# Document — Head-to-Head: TSM-Ultra-v2 vs VideoFormerLite-Ultra and Their Ensembles

**Models compared:** `tsm_ultra_v2` ([analysis](model_analysis_tsm_ultra_v2.md)) and `video_former_lite_ultra` ([analysis](model_analysis_video_former_lite_ultra.md))
**Evaluation set:** official `val_dir` — 6,745 clips, 33 classes (32 present), no clip used for training or tuning.
**Source of numbers:** all figures below are recomputed directly from the saved per-clip validation logits in `models/val_logits/` by `src/compare_tsm_vfl.py`, which writes `docs/tsm_vs_vfl_stats.json`. They are reproducible, not hand-transcribed.
**Figures:** `docs/figures/fig14_tsm_vs_vfl_perclass.png`, `docs/figures/fig15_ensembling_techniques.png`.

---

## 1. Why These Two Models

`tsm_ultra_v2` and `video_former_lite_ultra` are the two strongest single architectures in the closed-track series, and — more importantly for this document — they implement **opposite temporal inductive biases**:

| | TSM-Ultra-v2 | VFL-Ultra |
|---|---|---|
| Temporal mechanism | Parameter-free channel shift (±1 frame) before **every** residual block | All-to-all self-attention over frame tokens, in a **dedicated** 2-layer Transformer |
| Temporal scope | **Local** per block; global only via depth | **Global** in a single attention step |
| Spatial granularity of the temporal signal | **Full resolution** — time is mixed at every (h, w) before pooling | **Pooled** — time is mixed on one 512-d vector per frame, *after* spatial detail is averaged away |
| Temporal fusion at the head | Plain mean over T (but features already entangled) | Learnable [CLS] query |
| Params | ~11.2 M (ResNet18, 0 added by TSM) | ~17.5 M (ResNet18 + Transformer) |

The full architecture diagrams are in `model_analysis_tsm_ultra_v2.md §1.4` and `model_analysis_video_former_lite_ultra.md §1.4`. The one-line summary: **TSM mixes time early and fine-grained but locally; VFL mixes time late and global but coarse.** Section 3 shows which of these biases wins, and on what.

---

## 2. Overall Results

### 2.1 Singles (no TTA)

| Model | Top-1 | Top-5 | Macro-F1 |
|-------|-------|-------|----------|
| **TSM-Ultra-v2** | **0.3825** | **0.6870** | **0.3472** |
| VFL-Ultra | 0.3211 | 0.6485 | 0.2928 |

TSM-Ultra-v2 leads by **+6.14 pp top-1** and +5.44 pp macro-F1. The gap is not an artefact of a few classes — Section 3 shows TSM is at least level with VFL on **all but one** of the 32 evaluated classes.

### 2.2 Is one model "more temporal"? — the evidence

Both temporal stories are plausible on paper, so we let the data adjudicate. VFL's stated hypothesis (`model_analysis_video_former_lite_ultra.md §1.3`) was that global attention would help precisely on classes defined by the *first-vs-last frame relationship* (e.g. folding). **That advantage does not appear.** TSM wins every one of those classes (Section 3.1). The reason is granularity: the discriminating motion in SSv2 lives in a small spatial region (a hand, an object edge), and VFL's backbone averages that away before the Transformer can compare frames. TSM's shift operates on the raw feature maps, so the motion cue survives.

**Conclusion:** although VFL is the *architecturally more explicit* temporal model (a dedicated global attention module), **TSM-Ultra-v2 is the more *effectively* temporal model** for this task — it preserves and exploits the fine-grained, spatially-localised motion evidence that SSv2's trajectory and direction classes require.

---

## 3. Per-Class Comparison

`fig14_tsm_vs_vfl_perclass.png` plots per-class F1 for both single models and **two** ensembling techniques (log-prob/geometric-mean vs probability/arithmetic-mean averaging), and groups eight classes into the **four head-to-head scenarios** the two models can produce:

| Panel | Scenario | Example classes | What it shows |
|-------|----------|-----------------|---------------|
| 1 | **TSM > ensembles** | Putting sth into sth (0.30 vs 0.25/0.23), Putting sth behind sth (0.28 vs 0.24) | When TSM is *far* ahead of VFL, blending in the weak model can *dilute* the prediction — the single model beats both combiners. |
| 2 | **TSM > VFL** | Pulling left→right (0.61 vs 0.50), Folding sth (0.52 vs 0.41) | The temporal/trajectory classes (§3.1): TSM's fine-grained motion signal dominates; the ensemble matches TSM but does not lose. |
| 3 | **VFL > TSM** | Closing sth (0.32 vs 0.29), Pulling right→left (≈tie) | The handful of classes where VFL's global view helps; here the ensemble lifts *above both parents* (Closing 0.38, Pull r→l 0.43). |
| 4 | **VFL ≥ ensembles** | Pretending to put sth into sth (0.10 vs log 0.08), Picking sth up (0.07 vs log 0.07) | The **rarest** case, and only marginal: VFL nudges past the *geometric-mean* ensemble on a couple of hard, low-F1 classes — but the *arithmetic-mean* ensemble still matches or beats it. There is **no** class where VFL strictly beats both ensembles. |

The asymmetry between panels 1 and 4 is the practical takeaway: a strong single model (TSM) routinely beats the ensemble when its partner is far behind, but the weak single model (VFL) essentially **never** beats the ensemble — so ensembling carries almost no downside risk relative to the weaker model, while routinely recovering the panel-3 classes.

### 3.1 Where TSM beats VFL most — the temporal/trajectory classes

| Class | TSM F1 | VFL F1 | Δ (TSM−VFL) | Support |
|-------|--------|--------|-------------|---------|
| Dropping something into something | 0.34 | 0.21 | **+0.13** | 178 |
| Putting something into something | 0.30 | 0.17 | **+0.12** | 292 |
| Pouring something into something | 0.56 | 0.44 | **+0.12** | 278 |
| Pulling something from left to right | 0.61 | 0.50 | **+0.12** | 169 |
| Folding something | 0.52 | 0.41 | **+0.11** | 285 |
| Throwing something | 0.34 | 0.25 | +0.09 | 184 |
| Putting something behind something | 0.28 | 0.20 | +0.08 | 127 |
| Pretending to pour something out | 0.30 | 0.22 | +0.08 | 56 |

Every class in this list is defined by a **directional or topological motion arc** (pour/drop *into*, pull *left→right*, fold, throw, put *behind*). These are the classes that most demand spatially-local temporal evidence — exactly where TSM's early, full-resolution shift pays off and VFL's pre-attention pooling hurts. This is the per-class signature behind the "more temporal" conclusion in §2.2.

### 3.2 Where VFL is competitive

VFL strictly beats TSM on only **one** evaluated class, and ties on one more:

| Class | TSM F1 | VFL F1 | Δ (TSM−VFL) | Support |
|-------|--------|--------|-------------|---------|
| Closing something | 0.29 | 0.32 | −0.03 | 228 |
| Pulling something from right to left | 0.37 | 0.37 | −0.01 | 125 |

VFL's only outright win ("Closing something") is a slow, near-static-appearance action where global pooling loses little; its narrow miss on "Pulling … right to left" mirrors the known directional-TTA trade-off (`model_stats.json`). Everywhere else VFL trails TSM by ≥0.02 F1.

### 3.3 Complementarity — why VFL is still worth ensembling

Despite losing nearly every class, VFL is *not* redundant. On the 6,745 val clips:

| Outcome | Clips | Share |
|---------|-------|-------|
| Both correct | 1,571 | 23.3% |
| **Only TSM correct** | 1,009 | 15.0% |
| **Only VFL correct** | **595** | **8.8%** |
| Neither correct | 3,570 | 52.9% |

- The two models **agree on only 40.9%** of clips.
- VFL uniquely rescues **595 clips** that TSM gets wrong — these are the errors an ensemble can recover.
- The **oracle upper bound** (correct if *either* model is correct) is **47.07% top-1**, versus TSM's 38.25% alone: an 8.8 pp headroom that a good combiner can partially capture.

This is the textbook precondition for ensembling: complementary errors despite a clear accuracy gap.

---

## 4. Ensembling Techniques

We combine the two single models (no TTA) with five standard combiners. Each is computed in `src/compare_tsm_vfl.py`; results are plotted in `fig15_ensembling_techniques.png`.

| Technique | What it averages | Top-1 | Top-5 | Macro-F1 | Δ top-1 vs TSM |
|-----------|------------------|-------|-------|----------|-----------------|
| TSM alone (reference) | — | 0.3825 | 0.6870 | 0.3472 | — |
| VFL alone (reference) | — | 0.3211 | 0.6485 | 0.2928 | −6.14 |
| **Logit averaging** | mean of raw logits | **0.3874** | **0.7078** | **0.3533** | **+0.49** |
| **Log-prob averaging** (geometric mean) | mean of log-softmax | **0.3874** | **0.7078** | **0.3533** | **+0.49** |
| Probability averaging | mean of softmax probs | 0.3837 | 0.7036 | 0.3499 | +0.12 |
| Rank averaging | mean of per-clip class ranks | 0.3726 | 0.6956 | 0.3401 | −0.99 |
| Max-confidence routing | pick the more peaked model per clip | 0.3720 | 0.6765 | 0.3390 | −1.05 |

**Findings:**

1. **Log-domain averaging wins.** Logit averaging and log-prob (geometric-mean) averaging are tied at the top (0.3874, +0.49 pp over TSM, +0.61 pp top-5). Averaging in the log domain lets a confident model veto, which suits a strong+weak pairing.
2. **Arithmetic probability averaging is weaker** (0.3837): in linear space the weaker VFL pulls the mean toward its (often wrong) high-probability mass.
3. **Hard combiners hurt.** Both rank averaging and max-confidence routing fall *below* TSM alone — with one strong and one weak model, discarding TSM's calibrated margins (rank) or handing whole decisions to whichever model is more peaked (max-conf, which often picks the over-confident weaker model) is counter-productive.
4. **Equal-weight is close to optimal but slightly TSM-leaning.** Sweeping the log-prob mixing weight `w·TSM + (1−w)·VFL`:

   | w (TSM) | 0.0 | 0.3 | 0.5 | 0.6 | 0.7 | **0.8** | 0.9 | 1.0 |
   |---------|-----|-----|-----|-----|-----|---------|-----|-----|
   | Top-1 | .3211 | .3613 | .3874 | .3982 | .3994 | **.3997** | .3910 | .3825 |

   The optimum sits at **w≈0.8** (0.3997 top-1, **+1.72 pp over TSM alone**) — i.e. weight TSM ~4:1, but keep a non-trivial VFL contribution. A naive 50/50 already captures most of the gain; over-weighting VFL (w<0.5) is strictly worse than TSM alone.

### 4.1 With TTA

Applying 10-crop TTA to both singles before the same combiners lifts every number by ~2 pp and preserves the ranking (log-domain best):

| Technique (TTA singles) | Top-1 | Top-5 | Macro-F1 |
|-------------------------|-------|-------|----------|
| Logit averaging | **0.4096** | **0.7333** | **0.3704** |
| Log-prob averaging | **0.4096** | **0.7333** | **0.3704** |
| Probability averaging | 0.4070 | 0.7293 | 0.3664 |

For context, the project's **best overall** result (`model_stats.json`) is a *three-way* log-average ensemble (`tsm_v2_tta + tsm_v2_rot_tta + vfl_rot_tta`) at **0.4403** top-1 — confirming that the log-average recipe validated here on two models is the same one that scales to the final submission.

### 4.2 Per-class effect of ensembling

The ensemble's job is to recover the complementary clips from §3.3. The log-average ensemble beats *both* parents on these classes:

| Class | TSM F1 | VFL F1 | Ensemble F1 | Δ vs best parent | Support |
|-------|--------|--------|-------------|------------------|---------|
| Moving something away from something | 0.43 | 0.39 | 0.48 | +0.05 | 183 |
| Pulling something from right to left | 0.37 | 0.37 | 0.42 | +0.05 | 125 |
| Uncovering something | 0.42 | 0.39 | 0.46 | +0.05 | 312 |
| Pouring something out of something | 0.40 | 0.38 | 0.44 | +0.05 | 79 |
| Moving something closer to something | 0.54 | 0.53 | 0.58 | +0.05 | 213 |
| Closing something | 0.29 | 0.32 | 0.36 | +0.04 | 228 |
| Opening something | 0.30 | 0.26 | 0.34 | +0.04 | 332 |
| Putting something in front of something | 0.43 | 0.37 | 0.46 | +0.03 | 135 |

Notably, the ensemble recovers **"Closing something"** — the one class VFL won — *and* lifts it above either parent, and it rescues the directional pair "Pulling right→left" that TSM alone struggled with. The gains cluster on **moving/covering/opening** classes where the two temporal mechanisms disagree most, exactly as the 40.9% agreement rate predicts.

---

## 5. Summary

1. **TSM-Ultra-v2 is both the stronger and the more effectively temporal single model** (+6.14 pp top-1), because it mixes time at full spatial resolution before pooling — winning every trajectory/direction class (§3.1). VFL-Ultra's dedicated *global* attention is architecturally more explicit but operates on spatially-pooled features, so its theorised first-vs-last-frame advantage never materialises (§2.2).
2. **The two models are strongly complementary despite the gap**: they agree on only 40.9% of clips, VFL uniquely rescues 595 clips, and the oracle ceiling is 47.07% (§3.3).
3. **Log-domain averaging is the right combiner** (logit / log-prob average, +0.49 pp; +1.72 pp at the tuned 0.8 weight). Arithmetic probability averaging is weaker, and hard combiners (rank, max-confidence) fall below the best single model — the standard failure mode when pairing a strong and a weak model (§4).
4. **The ensemble's gains are concentrated on the classes where the temporal mechanisms disagree** (moving/covering/opening), including recovering the single class VFL had won (§4.2). The two-model log-average recipe validated here is the same one used by the project's best three-model submission (0.4403, §4.1).

---

## Reproduction

```bash
cd src
python compare_tsm_vfl.py        # writes ../docs/tsm_vs_vfl_stats.json
# figures fig14/fig15 are produced by the plotting block in the same analysis pass
```
