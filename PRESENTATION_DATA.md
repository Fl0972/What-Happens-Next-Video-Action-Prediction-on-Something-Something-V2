# Presentation — chart-ready data

Each section is a candidate chart: the data + the one-line story it tells.
Numbers are all from this project's logs (Kaggle), `val_dir` evaluation, or
`train.py` per-epoch output. CSV blocks are ready to paste.

---

## Chart 1 — The honest journey (Kaggle scores over time)

**Type:** vertical bar chart, colour-coded by phase (leaky vs clean).
**Story:** "The integrity correction cost ~21 pp; everything since has been climbing
honestly."

```csv
label,kaggle,phase
"Leaky V-JEPA (full video + SSv2 backbone) — DISCARDED",0.81,leaky
"V-JEPA-ft (clean rebuild) standalone",0.6361,clean
"Honest ensemble v1 (9 models)",0.6418,clean
"V-JEPA-pseudo standalone",0.6410,clean
"Honest ensemble v2 (9 models, swap in pseudo)",0.6418,clean
"V-JEPA-only (6 snapshots)",0.6410,clean
"V-JEPA-heavy (3:1 manual weights)",0.6499,clean
"Honest ensemble v3 (12-uniform)",0.6546,clean
"Honest ensemble v4 (14-uniform) ★",0.6586,clean
```

Visual cue: colour the leaky bar red with a strikethrough label, the honest
bars green/blue. Add a horizontal dashed line at **0.74** = 2nd-place reference.

---

## Chart 2 — Ensemble size vs Kaggle ("diversity wins")

**Type:** scatter/line, x = number of models in the uniform ensemble, y = Kaggle.
**Story:** "Per-model returns aren't diminishing — the right move was *more*
diverse honest models, not smarter weights."

```csv
n_models,composition,kaggle
1,V-JEPA-ft alone,0.6361
1,V-JEPA-pseudo alone,0.6410
6,"VFT + VPS (V-JEPA-only, 2 trajectories)",0.6410
9,VFT + k400 + tsm,0.6418
9,VPS + k400 + tsm,0.6418
12,VFT + VPS + k400 + tsm,0.6546
14,"+ 2 attempt1 Large 4f",0.6586
```

Add an arrow / callout: at **n=6** dropping k400 & tsm *hurts* even though
they individually score only ~0.55 — the weak honest models add uncorrelated
mistakes that the ensemble averages out.

---

## Chart 3 — Uniform vs manual weighting (same models, different weights)

**Type:** horizontal bar chart.
**Story:** "Manual weight tuning isn't the lever. Uniform mean of more diverse
models wins."

```csv
strategy,n_models,kaggle
"V-JEPA-only (6, uniform)",6,0.6410
"V-JEPA-heavy 3:1 (12, weighted)",12,0.6499
"Uniform (9 models)",9,0.6418
"Uniform (12 models)",12,0.6546
"Uniform (14 models)",14,0.6586
```

Annotate that the heavy-weighting variant (0.6499) is *between* uniform-9 and
uniform-12 — manual prior knowledge beats the smaller uniform but loses to
just adding more members.

---

## Chart 4 — V-JEPA training curves (the progressive-unfreezing story)

**Type:** multi-line plot of EMA val accuracy (train-split) over epochs.
**Story:** "Progressive unfreezing + LLRD: each stage's warm-start launches
slightly above the previous stage's plateau and keeps climbing."

```csv
phase,epoch,ema_val_trainsplit
"Frozen probe",1,0.5755
"Frozen probe",2,0.6203
"Frozen probe",3,0.6386
"Frozen probe",4,0.6476
"Frozen probe",5,0.6635
"Frozen probe",6,0.6725
"FT stage 1 (top-6 unfrozen + LLRD)",1,0.6764
"FT stage 1 (top-6 unfrozen + LLRD)",2,0.6914
"FT stage 1 (top-6 unfrozen + LLRD)",3,0.7006
"FT stage 1 (top-6 unfrozen + LLRD)",4,0.7052
"FT stage 2 (full encoder + LLRD)",1,0.7015
"FT stage 2 (full encoder + LLRD)",2,0.7184
"FT stage 2 (full encoder + LLRD)",3,0.7267
"FT stage 2 (full encoder + LLRD)",4,0.7281
"+ Pseudo-label retrain (full + LLRD)",1,0.7180
"+ Pseudo-label retrain (full + LLRD)",2,0.7273
"+ Pseudo-label retrain (full + LLRD)",3,0.7370
"+ Pseudo-label retrain (full + LLRD)",4,0.7415
```

Visualisation tip: plot epoch on x with each stage in a different colour;
use a shared epoch axis so the "warm-start jump" between stages is obvious
where one curve picks up where the previous left off.

---

## Chart 5 — The integrity discovery (where do the shipped frames sit?)

**Type:** histogram of `frame_003`'s position in the source clip (fraction).
**Story:** "The competition gives you ~40% of the video. My initial extraction
sampled all the way to 100% — that included the action's outcome. Caught it,
threw the run away."

```csv
fraction_of_source_video,count
0.16,1
0.37,1
0.39,1
0.40,5
0.45,1
0.46,1
0.48,1
0.51,1
0.52,1
0.90,1
```

(14 train clips sampled — exact distribution from the measurement.)

Annotate:
- Dashed vertical line at **0.40** = "median end of provided window"
- Shaded region **0.50 → 1.00** = "withheld future — using these = data leakage"
- Optional: overlay a second histogram showing the corrected extraction's
  window-end fractions (`median=0.39, p90=0.48, max=0.50`).

---

## Chart 6 — Mapping verification (same-ID vs random-ID best-match MAE)

**Type:** two grouped bars (or two-density violins) — same-ID vs random-ID.
**Story:** "Before doing the source extraction I needed to *prove* the folder
names map to real SSv2 videos. Same-ID match error is 2.4× lower than
random-ID — the mapping is real."

```csv
group,median_MAE,mean_MAE,extremum
SAME id,28.3,29.1,max=70.6
RANDOM id,67.6,69.4,min=38.6
```

If you have raw numbers preferred (40-clip discriminative test):
- SAME id distribution clustered around 28–35 (image-pair MAE on 0–255 scale).
- RANDOM id distribution clustered around 50–80.

---

## Chart 7 — Train-split val vs honest `val_dir` (calibration)

**Type:** scatter of `(train_split_val, val_dir)` per model, with `y=x` line.
**Story:** "The internal train-split val was always optimistic. After the
pseudo-label retrain the gap *doubled* — early warning that pseudo labels
were over-fitting val."

```csv
model,trainsplit_emaval,val_dir_top1,val_kaggle_gap
"V-JEPA-ft (stage 2 full)",0.7281,0.6411,0.6361
"V-JEPA-pseudo (full + pseudo)",0.7415,0.6519,0.6410
```

Add Kaggle as a third bar per model — the comparison is what tells the story:

```csv
model,trainsplit_emaval,val_dir,kaggle
"V-JEPA-ft",0.7281,0.6411,0.6361
"V-JEPA-pseudo",0.7415,0.6519,0.6410
```

Note the gap widening: trainsplit→Kaggle is 0.092 for V-JEPA-ft but
0.100 for V-JEPA-pseudo; val_dir→Kaggle is 0.005 → 0.011 — the pseudo run's
val_dir/Kaggle distance *doubled*.

---

## Chart 8 — Single-model val_dir comparison (where the diversity comes from)

**Type:** horizontal bar chart sorted by val_dir.
**Story:** "Even the weak models add value because their errors are
uncorrelated with V-JEPA's. The 14-uniform beats the V-JEPA-only ensemble."

(Measured on the 4-frame validation pipeline used during the gradient-ensemble
analysis; all single-checkpoint, 10-view TTA.)

```csv
model,val_dir_top1
videomae_ovn1_ssv2_top1,0.6059
videomae_ovn1_ssv2_top2,0.5956
videomae_ovn1_ssv2_top3,0.5927
videomae_ovn2_large_attempt1_top2,0.5999
videomae_ovn2_large_attempt1_top3,0.5686
videomae_ovn1_k400_top1,0.5543
videomae_ovn1_k400_top2,0.5566
videomae_ovn1_k400_top3,0.5537
```

```csv
model,val_dir_top1
"V-JEPA-ft (top-3 ensemble + TTA, 16f win-capped)",0.6411
"V-JEPA-pseudo (top-3 ensemble + TTA, 16f win-capped)",0.6519
```

Colour-code by family (V-JEPA vs VideoMAE vs TSM) so the architectural
diversity is visible.

---

## Chart 9 — Learned ensemble weights collapse onto the strongest models

**Type:** stacked or grouped bar of learned softmax weights per model.
**Story:** "When I fitted ensemble weights by gradient descent, the optimiser
concentrated almost all the mass on the two strongest models — which is
exactly what makes uniform weighting still hard to beat once those strong
members dominate."

```csv
checkpoint,learned_global_weight
videomae_ovn2_large_attempt1_top2,0.4286
videomae_ovn2_large_attempt1_top3,0.0013
videomae_ovn1_ssv2_top1,0.3922
videomae_ovn1_ssv2_top2,0.1363
videomae_ovn1_ssv2_top3,0.0168
videomae_ovn1_k400_top1,0.0049
videomae_ovn1_k400_top2,0.0151
videomae_ovn1_k400_top3,0.0048
```

(Eight-model fit; large_attempt1_top2 and ssv2_top1 took ~82% of total
weight; k400 was effectively zeroed.)

---

## Chart 10 — Compute budget / disk layout (optional, "what infra mattered")

**Type:** simple labelled diagram or a 2-bar comparison.
**Story:** "NFS home had a 30 GB quota that *silently* truncated checkpoints
until the `fsync` fix turned the silent failure into an honest error.
Solved by moving model checkpoints + HF cache to local NVMe."

```csv
location,size_gb,used_for
"NFS home (quota)",30,"submissions, code, small cached softmax"
"Local NVMe (/Data)",561,"model checkpoints, HF model cache, 16-frame extracted dataset, source SSv2 videos"
```

---

## Bonus — summary numbers worth keeping handy

| Metric | Value |
|---|---|
| Frames per clip shipped | 4 |
| Last shipped frame in source video (median fraction) | 0.40 |
| Window-cap I applied (median, max) | 0.39, 0.50 |
| Source SSv2 size | ~220 847 webm, ~19 GB |
| Re-extracted dataset (16 frames @256px window-capped) | ~14 GB |
| Final ensemble members | 14 (best so far) |
| Best honest Kaggle | 0.6586 |
| Leaky Kaggle (discarded) | 0.81 |
| Pre-V-JEPA baseline (mixed honesty) Kaggle | 0.64 |
| 2nd place reference | 0.74 |
| GPU | RTX 4000 Ada, 21 GB |
| Largest single-model train time | ~7 h (V-JEPA stage 2, 4 epochs full FT) |

---

## Suggested narrative arc for the slide deck

1. **The task** (Chart 5 sets up "only first 40% is available — anticipation
   matters").
2. **The honest journey** (Chart 1 — including the leaky bar with a clear
   "discarded" label).
3. **What I built** (cite §3 of PRESENTATION.md; show Chart 4 = the
   progressive-unfreezing training curve as evidence of careful engineering).
4. **The empirical insight** (Charts 2 + 3 — diversity > weighting; the
   pivot from "smarter weights" to "more models").
5. **Calibration & integrity discipline** (Chart 7 — train/val/Kaggle
   gap doubled after pseudo; Chart 9 — learned weights agree with our
   intuition; Chart 5/6 = the audit you did before trusting the source
   extraction).
6. **Where we ended up + honest limits** (final numbers + the "21 GB
   GPU is the ViT-g ceiling" caveat).
