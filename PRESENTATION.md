# What Happens Next? — Project Presentation Notes

> Companion to **`CHANGES.md`** (full technical changelog, ~870 lines).
> This document is structured for a presentation: **what I did, what I reused
> and where it came from, how I reasoned**, and a small results section.

---

## 0. TL;DR

- **Task.** Anticipate which of 33 action classes is happening in a video, given
  only the **first ~40%** of its frames (curated subset of Something-Something v2).
  Track B (Open World): external data and pretrained models permitted; *use of
  privileged information / data leakage is strictly prohibited*.
- **Final approach.** Uniform soft-vote ensemble of 14 honest, rules-clean
  models trained on the provided frames + extra real frames re-extracted from
  the source SSv2 (capped per-clip to the *provided* temporal window).
- **Headline result.** **0.6586 Kaggle** with the honest 14-uniform; an
  earlier, *leaky* run hit 0.81 — I **caught and discarded it** when I realised
  the leakage; see §4 for the full integrity story.
- **What I want to highlight.** Cited sources, original engineering, and an
  honest narrative — including the moment I had to throw away a 0.81 run.

---

## 1. Problem & dataset

| | |
|---|---|
| **Challenge** | CSC_43M04_EP "What Happens Next?" (Track B, Open World) |
| **Data** | Curated subset of Something-Something-v2, **4 shipped frames per clip** at 224×224 (`frame_000.jpg`…`frame_003.jpg`) |
| **Splits** | train 44 993 (class-prefixed folders), val 6 745, test 6 913 |
| **Classes** | 33 (numeric folder prefixes 000–032; prefix 027 missing from train) |
| **Critical constraint** | "Only the first portion of each video (e.g. 60%) is available." Empirically the shipped `frame_003` sits at median **0.40** of the source clip — the action's *outcome* is withheld and is the task to *anticipate*. |
| **Metric** | Top-1 accuracy (Top-5 secondary) on a hidden test set |

---

## 2. Sources & attributions (clearly cited)

Listing everything I **reused** so my own contributions in §3 are
unambiguous. CHANGES.md cites each in its respective section.

### 2.1 Pretrained models / HuggingFace checkpoints
| Checkpoint | Used as | Reference |
|---|---|---|
| `MCG-NJU/videomae-base-finetuned-ssv2` | Earlier ovn1 (later dropped — see §4) | Tong et al. NeurIPS 2022 |
| `MCG-NJU/videomae-base-finetuned-kinetics` | K400 4f Base in ensemble | Tong et al. NeurIPS 2022 |
| `MCG-NJU/videomae-large-finetuned-kinetics` | Large 4f in ensemble; Large 16f in new training | Tong et al. NeurIPS 2022 |
| `facebook/vjepa2-vitl-fpc16-256-ssv2` | First V-JEPA run (later **discarded** — leakage) | Bardes et al. 2025 |
| `facebook/vjepa2-vitl-fpc64-256` | **Clean SSL backbone** used in the final run | Bardes et al. 2025 |

### 2.2 Papers / standard techniques (and where I used them)
- **VideoMAE** (Tong, Song, Wang, Wang, NeurIPS 2022, arXiv:2203.12602) —
  primary video backbone family.
- **V-JEPA 2** (Bardes et al., Meta, arXiv:2506.09985) — chosen because top
  leaderboard teams use it; predicts in feature space instead of pixels.
- **TSM** (Lin, Gan, Han, ICCV 2019, arXiv:1811.08383) — temporal-shift
  module on a ResNet-50 backbone (an ensemble member).
- **ResNet-50** (He et al., CVPR 2016) — backbone for TSM.
- **AdamW** (Loshchilov & Hutter, ICLR 2019).
- **OneCycleLR** (Smith, arXiv:1708.07120) — LR schedule.
- **LLRD** (Clark et al. ELECTRA, ICLR 2020 → Bao et al. BEiT, ICLR 2022 →
  He et al. MAE, CVPR 2022) — layer-wise LR decay for finetuning ViTs.
- **MixUp** (Zhang et al., ICLR 2018) + **CutMix** (Yun et al., ICCV 2019).
- **EMA of weights** (Polyak averaging; standard).
- **Spatial 10-crop / temporal multi-clip TTA** (standard).
- **Snapshot ensembling** (Huang et al., ICLR 2017, arXiv:1704.00109).
- **Stacked generalisation / linear stacking** (Wolpert, *Neural Networks*
  1992; Breiman, *Machine Learning* 1996) — for learned ensemble weights.
- **Pseudo-labelling / noisy-student** (Lee 2013; Xie et al. 2020).
- **Something-Something v2** (Goyal et al., ICCV 2017, arXiv:1706.04261).

### 2.3 Libraries
- **PyTorch** 2.8 (CUDA 12.8).
- **HuggingFace `transformers`** 4.57 (VideoMAE + V-JEPA 2 official impl).
- **Hydra / OmegaConf** (config management).
- **`ffmpeg`** (source-frame re-extraction).

---

## 3. Original contributions (my work)

What I implemented or designed specifically for this challenge, beyond
calling library APIs. CHANGES.md section numbers are in parentheses for the
in-code companion.

### 3.1 Engineering / infrastructure

- **Reboot-resilient training stack** (§3.7): `train.py` writes a
  durable `<ckpt>.resume.pt` every epoch with `fsync`, `loop_until_done.sh`
  re-runs the orchestration script on crash, `run_step` markers skip
  finished stages, and a systemd-user unit restarts the whole thing on
  reboot. Survived multiple kills and a host reboot during this work.
- **Local-disk checkpoint strategy** (`/Data` instead of NFS) — diagnosed
  that the NFS home's silent **30 GB quota** was producing 0-byte
  checkpoints. The `fsync` save turned the silent failure into an honest
  `OSError: Disk quota exceeded`. Subsequent Large-model runs go to local
  NVMe with `HF_HOME=/Data/.../hf_cache` so the 1–3 GB HF downloads also
  bypass the quota.
- **fsync-durable atomic saves** — close + `os.fsync` + atomic rename for
  every checkpoint write (top-k snapshots, resume state, final mirror).
- **Cache-dir consolidation** for cross-architecture ensembling — pinned
  `training.softmax_cache_dir` so V-JEPA on `/Data` and VideoMAE on NFS share
  one canonical cache directory.

### 3.2 Methodological / scientific

- **`extract_ssv2_frames.py`** — re-extracts denser real frames from the
  source SSv2 videos, **per-clip window-capped** to that clip's own
  shipped-frame coverage. Each clip's `frame_003` is matched against
  low-res source frames to find the end of the *provided* window;
  sampling is then linspace-uniform within `[0, that index]`, and the
  search is hard-bounded to the first 50 % of the source so the withheld
  action outcome (at 0.6+) is *unreachable*. This is the key piece that
  lets me use more real frames *without* using privileged information.
- **Honest cross-preprocessing ensemble** (`honest_ensemble.py`) — combines
  test softmax tensors that were computed on **different preprocessing**
  per model (V-JEPA 16f/256 win-capped vs k400/TSM 4f/224 shipped vs Large
  4f/224). The existing ensemble scripts assume a single dataset/transform;
  mine just loads the per-model caches and combines.
- **Test-only softmax caching** (`cache_test_softmax.py`) — saves the
  full val pass when we already trust the train-split val and only need
  test inference for the ensemble.
- **Learned-weight ensemble** (`ensemble_gradient.py`, `learned_honest_ensemble.py`)
  — gradient-descent fit of softmax-normalised per-model weights on val
  NLL, with an L2-toward-uniform regulariser and an honest held-out
  estimate. Two modes: global per-model and per-class. Linear-stacking
  recipe (Breiman 1996), wired to the per-model cached softmax.
- **Progressive unfreezing + LLRD for V-JEPA** (`models/vjepa.py`,
  `build_vjepa_param_groups`) — V-JEPA wasn't in the codebase originally;
  I wrote the wrapper, plumbed `image_size` end-to-end, and built the
  layer-wise LR decay helper analogous to the VideoMAE one. Staged
  schedule: frozen attentive probe → top-6 unfrozen → full encoder
  unfrozen, with `init_weights_from` warm-starting each stage from the
  previous one.
- **Cross-backbone ensembling fix** (§4.6) — the original ensemble scripts
  switched architecture only on `model_name`, but VideoMAE Base and Large
  both save `model_name="videomae"`. The pipeline silently misloaded
  Large weights into a Base graph until the cache happened to miss. I
  rewrote the inference loop to **rebuild from each checkpoint's own
  stored config**, which is correct for any Base/Large mix.
- **Pseudo-labelling pipeline** (§3.8) — already present in the codebase
  for an earlier model; I retargeted it at V-JEPA-ft top-3 + TTA to
  generate `vjepa_pseudo.csv` (conf ≥ 0.85), then continue-finetuned
  V-JEPA full + LLRD on `train ∪ pseudo-test`.
- **Label-aware horizontal flip** (§1.5) — when an hflip-augmented clip
  is sampled from a *direction-encoded* SSv2 class (e.g. "Pulling
  something from left to right"), the visual content now matches the
  mirror class, so the label is swapped to its mirror pair. Also applied
  at TTA so the flipped views' logits are re-permuted before averaging.
- **`grad_*` config surface for the learned ensemble** — added
  `grad_weight_mode`, `grad_steps`, `grad_lr`, `grad_l2_uniform`,
  `grad_holdout_frac` to the Hydra struct so the ensemble is fully
  configurable from the command line.
- **`image_size` plumbed end-to-end** — added a `dataset.image_size`
  config; `train.py` builds transforms with it and saves it in the
  checkpoint payload; inference scripts read it back from the checkpoint
  — so a 256-pixel V-JEPA and a 224-pixel VideoMAE can coexist in one
  ensemble run.

### 3.3 Empirical / experimental design contributions

- The decision tree behind every model choice is in §4 below. Notable
  empirical insights produced during the work, **not from any paper**:
  - **Diversity beats weighting on this task.** A uniform 12-model
    ensemble (V-JEPA-ft + V-JEPA-pseudo + k400 + tsm) outperforms a 6-model
    "V-JEPA-only" ensemble *and* a manually V-JEPA-heavy weighted ensemble
    on Kaggle, even though k400 and tsm score only ~0.55 individually.
  - **The val/Kaggle gap doubled after pseudo-labelling** (0.5 → 1.1 pp),
    consistent with the model learning V-JEPA-ft's biases on val that
    don't fully generalise.

### 3.4 What I deliberately did *not* do (integrity)
- Did not look up SSv2's published train/val labels for the test set even
  though folder names are SSv2 IDs (it would be one lookup away and
  would defeat the task).
- After realising the source-frame extraction had been using the
  *withheld* portion, I rebuilt the whole pipeline rather than ship the
  inflated number. See §4.

---

## 4. Reasoning narrative & the integrity story

This is the part I most want to communicate: the project's decisions
weren't a straight line, and the most important one was throwing away a
0.81 Kaggle submission.

### 4.1 Phase 1 — the baseline (~0.64)
- Started from a TSM + VideoMAE pipeline already in the codebase.
- Added Large VideoMAE finetuning + snapshot ensembling + per-class and
  uniform soft-vote (§4.1–4.5 of CHANGES). Landed at ~0.64 Kaggle.

### 4.2 Phase 2 — V-JEPA, source frames, **and the leak**
- Top leaderboard teams were reportedly using V-JEPA 2 → I added a V-JEPA
  wrapper with frozen-encoder attentive probe.
- V-JEPA needs ≥16 frames to shine; the shipped data has 4. The folder
  names (`video_<id>`) match SSv2's `1..220 847` range, so I re-extracted
  16 frames from the *source* SSv2 webm files and pointed training at
  the new tree (`val2_16f`). Course explicitly permits external data.
- Trained V-JEPA-ft → **Kaggle 0.81**. Way above 2nd place (0.74).
- *Pause and audit.* The competition rules say
  *"only the first portion (e.g. 60 %) of each video is available… use of
  privileged information or data leakage is strictly prohibited."*
  I measured where the shipped `frame_003` sits in the source clip:
  **median fraction 0.40** (max ~0.52). My extraction sampled
  `0 → 100 %` of the source, so the model was seeing the
  withheld outcome — and a SSv2-label-finetuned backbone had also been
  trained on those same SSv2 videos *with their labels*. Two leakage
  vectors, both prohibited.
- **Discarded the 0.81 submission and the whole leaky checkpoint family.**

### 4.3 Phase 3 — the clean rebuild
- Rewrote the extraction (`extract_ssv2_frames.py`) to **per-clip
  window-cap**: find each clip's last shipped frame in the source, sample
  16 frames only within `[0, that index]`, and hard-bound the search to
  the first 50 % of the clip so it can *never* reach the withheld
  outcome. Verified empirically: median window 0.39, max 0.50.
- Switched the V-JEPA backbone from the SSv2-finetuned variant (saw test
  labels) to **self-supervised `vjepa2-vitl-fpc64-256`** (no labels in
  pretraining). Random-init head, no warm-start from SSv2.
- Result: V-JEPA-ft alone Kaggle **0.6361**. Honest, defensible, and ~21 pp
  below the leaky number — that delta is exactly what the leak was buying.

### 4.4 Phase 4 — progressive unfreezing + LLRD
- Two-stage finetune: top-6 layers unfrozen + LLRD warm-started from the
  probe, then full encoder unfrozen + LLRD warm-started from stage 1.
  Stage 1 + stage 2 climbed the train-split EMA val from 0.672 → 0.705 →
  0.728.

### 4.5 Phase 5 — honest ensembling and the diversity insight
- Initial honest ensemble (V-JEPA-ft + k400 + tsm, uniform): **0.6418**.
- Pseudo-label retrain of V-JEPA → **0.6410 standalone** (val/Kaggle gap
  doubled; pseudo overfits to val-distribution).
- Surprising experiment: added V-JEPA-ft *and* V-JEPA-pseudo to the
  ensemble (12-uniform) → **0.6546** (+1.3 pp). Manual V-JEPA-heavy
  weighting was *worse*. **Uniform wins because the weak models add
  uncorrelated mistakes.**
- Added 2 cached attempt1 Large 4f checkpoints (14-uniform) → **0.6586**.
- In progress at time of writing: a new VideoMAE-Large K400 finetune at
  16f/224 on `val2_win16` for further diversity → 17-uniform.

---

## 5. Results

### 5.1 Honest progression (Kaggle)
| # | Submission | Kaggle | Delta | Notes |
|---|---|---|---|---|
| — | (leaky V-JEPA, **discarded**) | 0.81 | — | privileged info; not submitted as final |
| 1 | V-JEPA-ft alone (clean) | 0.6361 | — | starting honest baseline |
| 2 | Honest ensemble v1 (9-uniform: V-JEPA-ft + k400 + tsm) | 0.6418 | +0.6 pp | |
| 3 | V-JEPA-pseudo standalone | 0.6410 | +0.5 pp vs 1 | val gap doubled |
| 4 | 12-uniform (added V-JEPA-pseudo) | 0.6546 | +1.3 pp | **diversity wins** |
| 5 | V-JEPA-only (6 snapshots) | 0.6410 | regression | dropping weak models *hurt* |
| 6 | V-JEPA-heavy (3:1) | 0.6499 | between 4&5 | over-weighting V-JEPA suboptimal |
| 7 | **14-uniform (added attempt1 Large 4f)** | **0.6586** | best so far | |
| 8 | 17-uniform (+ new Large 16f/224, *in progress*) | TBD | projected 0.66–0.67 | |

### 5.2 Calibration
- Val/Kaggle gap stayed at ~0.5 pp for the cleanly-trained V-JEPA-ft, so
  `val_dir` was a faithful proxy for picking submissions.
- Gap **doubled** for the pseudo-label retrain (0.6519 val → 0.6410 Kaggle)
  — early warning that pseudo-labels can over-fit to val-distribution.

---

## 6. Honest limits / what didn't work

- **V-JEPA-Giant** would be the natural next step (Meta reports 77.3 % on
  SSv2 with ViT-g) but doesn't fit on one 21 GB GPU even for inference at
  its native 384/64f config. ViT-L is the practical ceiling here.
- **Learned ensemble weights** (`learned_honest_ensemble.py`): would
  almost certainly produce a result *similar* to uniform on this set of
  models. Skipped at the end to save GPU hours close to the deadline.
- **More aggressive pseudo-label iterations**: stopped at one round
  after the val/Kaggle gap revealed the over-fitting risk.
- **6 → 12 model uniform** was the biggest single jump (+1.3 pp); adding
  the 13th and 14th models bought another +0.4 pp; per-model marginal
  return is diminishing.

---

## 7. Pointers (where to look in the code)

| Topic | File |
|---|---|
| Window-capped source-frame extraction | `src/extract_ssv2_frames.py` |
| V-JEPA wrapper, freeze/LLRD helpers | `src/models/vjepa.py` |
| VideoMAE wrapper, LLRD helpers | `src/models/videomae.py` |
| Training loop, resume, LLRD dispatch | `src/train.py` |
| Test-only softmax caching | `src/cache_test_softmax.py` |
| Cross-preprocessing uniform ensemble | `src/honest_ensemble.py` |
| Learned-weight ensemble (val NLL fit) | `src/ensemble_gradient.py`, `src/learned_honest_ensemble.py` |
| Per-class weighted ensemble | `src/ensemble_per_class.py` |
| Snapshot ensemble + single-ckpt inference | `src/create_submission.py`, `src/evaluate.py` |
| Pseudo-label generator | `src/pseudo_label.py` |
| End-to-end run scripts (systemd-durable) | `src/run_vjepa_ft.sh`, `src/run_honest_pseudo.sh`, `src/run_vmae_l_win.sh` |
| Resume wrapper | `~/loop_until_done.sh` |
| systemd unit | `~/.config/systemd/user/ovn2-train.service` |
| Full changelog | `CHANGES.md` |

---

## 8. Things to emphasise verbally in the presentation

1. **The leakage story.** I had a 0.81 submission, audited it against the
   rules myself, found two leakage vectors, and threw it away. Closing the
   audit took rebuilding the whole pipeline (new extraction + new
   backbone + new training) and a 21 pp drop on Kaggle. *That* is the
   project's most important slide.
2. **Engineering over flashy modelling.** The biggest single Kaggle gain
   (+1.3 pp) didn't come from a new architecture — it came from realising
   that uniform soft-voting over genuinely diverse honest models beats
   weighting tricks on this task.
3. **Reproducibility & durability.** Everything runs under systemd with
   resume + skip-if-done + retry, on local disk, with `fsync`-durable
   saves — designed because the cluster had quota-induced silent
   corruption and external process kills.
4. **What I built vs reused.** Sections 2 and 3 above are the explicit
   line.
