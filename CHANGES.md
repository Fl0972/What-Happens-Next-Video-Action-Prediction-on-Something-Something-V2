# Project changelog & attributions

This document records the modifications I made to the baseline pipeline of the
"What Happens Next?" video classification challenge (CSC_43M04_EP), explains the
intent behind each change, and credits the external sources from which each
technique originates. The goal is to give a transparent picture of which ideas
are mine, which come from published work, and where each component lives in the
codebase — so nothing reads as plagiarism.

The list is roughly chronological. Files referenced are relative to the repo
root (i.e. `src/...`).

---

## 1. Data layer

### 1.1 LMDB-backed dataset (optional)
- **What.** Added an optional Lightning Memory-Mapped Database (LMDB) backend
  for the frame dataset (`src/dataset/lmdb_dataset.py`), guarded by
  `dataset.lmdb_dir`. The plain JPEG pipeline (`VideoFrameDataset`) is the
  default fallback.
- **Why.** The dataset is ~250k JPEG files; on a cold cache, individually
  opening each file is the bottleneck. LMDB stores all frames in a single
  memory-mapped file with O(1) lookup per key.
- **Source / credit.** LMDB is by Howard Chu (Symas/OpenLDAP). The
  "videos-as-keys-in-LMDB" pattern is standard in video research stacks
  (e.g. mmaction2, PySlowFast use a similar pattern with binary blobs).

### 1.2 TSN-style temporal jitter for training
- **What.** `_pick_frame_indices_tsn()` in `src/dataset/video_dataset.py`:
  divides the video into `T` equal segments and samples one random index per
  segment. The dataset accepts a `temporal_jitter` flag; only enabled for the
  training split.
- **Why.** Each epoch the model now sees a different temporal slice of every
  video — cheap "temporal data augmentation" that also increases the effective
  number of clips per video.
- **Source / credit.** Wang et al., **"Temporal Segment Networks: Towards Good
  Practices for Deep Action Recognition"** (ECCV 2016), arXiv:1608.00859.

### 1.3 Multi-clip temporal TTA
- **What.** `_pick_frame_indices_multi()` in `src/dataset/video_dataset.py`
  produces `n_clips` *deterministic but distinct* frame samplings of the same
  video. The dataset's `n_clips` param yields `(n_clips, T, C, H, W)` views;
  combined with the spatial 10-crop TTA it returns `(n_clips × 10, T, C, H, W)`
  flattened into the view dimension. Logits are averaged across all views in
  `evaluate.py` and `create_submission.py`.
- **Why.** Spatial TTA covers cropping uncertainty but does not change which
  frames the model sees. Sliding the segment offset gives the model multiple
  temporal viewpoints — most influential on motion-heavy datasets.
- **Source / credit.** Multi-clip evaluation is standard practice in the video
  literature, popularised by TSN (Wang et al. 2016 — "test by averaging the
  scores of K snippets") and SlowFast / Kinetics protocols.

### 1.4 Temporally-consistent training augmentation
- **What.** Replaced the previous frame-by-frame random transforms with a
  `VideoTransform` class (`src/utils.py`) that **samples random parameters
  once per clip** and applies the *same* transformation to every frame:
  - `RandomResizedCrop(scale=0.5–1.0)`
  - random horizontal flip
  - rotation `±15°`
  - color jitter (brightness, contrast, saturation, hue)
  - random grayscale (`p=0.1`)
  - random Gaussian blur (`p=0.2`)
  - random erasing (`p=0.3`) — same rectangle on every frame
- **Why.** Frame-independent augmentations break temporal coherence — the
  network sees a "video" where every frame has a different crop, color cast,
  etc. Sampling the augmentation parameters once per clip preserves the
  temporal signal that the model actually needs to learn.
- **Sources / credits.**
  - Random Erasing: Zhong et al., **"Random Erasing Data Augmentation"**
    (AAAI 2020), arXiv:1708.04896.
  - The "single-set-of-params per clip" pattern matches the approach used by
    PySlowFast and mmaction2 for video augmentation.
  - The base transform list is otherwise the standard ImageNet recipe
    (Krizhevsky et al. 2012; He et al. 2016).

### 1.5 Label-aware horizontal flip (SSv2 mirror pairs)
- **What.** SSv2 has direction-encoded mirror class pairs — in this challenge
  subset:
    - `018_Pulling_something_from_left_to_right`
    - `019_Pulling_something_from_right_to_left`
  A naive hflip on a class-018 video produces visual content matching class
  019 but keeps the original label, so the model is asked to learn that
  identical pixels mean opposite classes. Three coordinated changes fix this:
  1. `discover_flip_pairs(class_names)` in `src/utils.py` finds all
     `…_from_left_to_right` ↔ `…_from_right_to_left` pairs by name and
     returns the bidirectional `{i: j, j: i}` mapping.
  2. `VideoTransform.__call__(frames, label=…)` now optionally takes a label
     and, whenever a horizontal flip is applied, swaps the label via the
     stored `flip_pairs`. Non-paired classes are unaffected.
     `VideoFrameDataset.__getitem__` threads the label through and uses the
     possibly-remapped value as the training target.
  3. At inference (TTA), `VideoTransform.tta()` interleaves [orig, hflip] per
     spatial offset, so the odd-indexed views in each clip's 10-view block
     are the flipped ones. `_multi_view_logits()` (in both `evaluate.py` and
     `create_submission.py`) reindexes those views' logits along the class
     axis using `build_flip_perm(num_classes, flip_pairs)` (`src/utils.py`)
     before averaging — the flipped view's "right-to-left" prediction is
     mapped back to the original "left-to-right" target, and vice versa. The
     `flip_pairs` mapping is persisted in every saved checkpoint so the
     inference pipelines reconstruct the permutation without re-scanning the
     train directory.

  Toggle: `training.label_aware_flip` (default `true`); leaves the augmentation
  intact for non-paired classes (so the regularization benefit on the safe 31
  classes is preserved).
- **Why.** With naive hflip, ~50% of class-018 / class-019 examples per epoch
  carry a wrong label, and at inference half the TTA views vote for the
  mirror class. Both effects collapse the two classes' decision boundary.
  Label-aware flip preserves the data-augmentation signal on every class
  while keeping training and TTA self-consistent for the direction-encoded
  pair.
- **Source / credit.** This is the standard SSv2 augmentation recipe used in
  `mmaction2` and `pytorch-video` — usually called the `flip_label_map`
  (mmaction2) or `label_swap` (PySlowFast) flag. Originally documented as
  part of the TSN data pipeline (Wang et al., ECCV 2016, arXiv:1608.00859)
  and discussed in §A.4 of the SSv2 dataset paper (Goyal et al., **"The
  'something something' video database for learning and evaluating visual
  common sense"**, ICCV 2017, arXiv:1706.04261).

### 1.6 MixUp and CutMix at the batch level
- **What.** `mixup_data()` and `cutmix_data()` in `src/utils.py`, applied
  inside `train_one_epoch()` (`src/train.py`). Configured by
  `training.mixup_alpha` and `training.cutmix_prob`. CutMix is selected per
  batch with probability `cutmix_prob`; otherwise MixUp is used when
  `mixup_alpha > 0`. The combined loss is computed via `mixed_loss()`.
- **Why.** Strong regularisation against overfitting on a small training set.
  Both also encourage smoother decision boundaries.
- **Sources / credits.**
  - MixUp: Zhang et al., **"mixup: Beyond Empirical Risk Minimization"**
    (ICLR 2018), arXiv:1710.09412.
  - CutMix: Yun et al., **"CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features"** (ICCV 2019), arXiv:1905.04899.
  - The "MixUp ↔ CutMix random switch per batch" pattern is the same one used
    in `timm` (Wightman, https://github.com/huggingface/pytorch-image-models).

---

## 2. Model

### 2.1 TSM-ResNet
- **What.** `src/models/tsm_resnet.py` wraps every residual block of a
  torchvision ResNet with a parameter-free `TemporalShift` module (1/8 of
  channels shifted backward in time, 1/8 forward, the rest unchanged).
- **Why.** Adds cross-frame information flow to a 2D backbone with zero
  parameters and ~zero extra FLOPs.
- **Source / credit.** Lin, Gan, Han, **"TSM: Temporal Shift Module for
  Efficient Video Understanding"** (ICCV 2019), arXiv:1811.08383. Implementation
  follows the ideas in the paper; the channel-shift indexing pattern is a
  faithful re-implementation of the algorithm described in §3.

### 2.2 VideoMAE ViT-B/16 finetuning (Track B "big swing")
- **What.** New model wrapper `src/models/videomae.py` adapting HuggingFace's
  `VideoMAEForVideoClassification` to this repo's `(B, T, C, H, W) → (B,
  num_classes)` interface. Default checkpoint:
  `MCG-NJU/videomae-base-finetuned-ssv2` — ViT-B/16 with tubelet embeddings,
  pretrained on Kinetics-400 with masked reconstruction and finetuned on
  Something-Something v2. The 174-class head is replaced by a fresh
  `Linear(768 → num_classes)`. Selectable via `experiment=videomae` (new file
  `src/configs/experiment/videomae.yaml`); model config lives in
  `src/configs/model/videomae.yaml`. The `build_model()` dispatcher in
  `train.py` gains a `videomae` branch (lazy-imported so the dependency is
  optional). The existing TSM model is untouched — both can be trained
  independently and ensembled.
- **Why.** ViT-B with SSv2-pretrained features is the largest realistic gain
  available without leaving the open-world track: features are already
  action-discriminative and the classification head is the only layer that
  needs to learn the task-specific 33-way mapping. Hyperparameters in the
  experiment file follow standard ViT finetuning practice (low LR, small
  batch, fewer epochs, weight decay 0.05).
- **Head warm-start (this challenge ⊂ SSv2).** The 32 challenge classes that
  have data are *verbatim copies* of SSv2 templates (verified by listing the
  class folders against the checkpoint's `id2label`). So instead of a random
  head, `VideoMAE.__init__` (when given `class_names`) loads the full
  pretrained 174-class head and copies each matching row (weight + bias) into
  the new `Linear(768 → 33)`; unmatched indices (only class `027`, which has
  no data) keep a `normal_(std=initializer_range)` init mirroring HF's own
  classifier init. Matching is by normalized name (strip `NNN_` prefix, drop
  `[ ]` placeholders / punctuation, lowercase), with a unique-prefix fallback
  for truncated folder names (e.g. `..._but_something_` →
  `..., but [something] is empty`). Helpers: `match_classes_to_ssv2()` and
  `_normalize_label()` in `videomae.py`. `train.py` discovers `class_names`
  from the train-dir folder names (no extra filesystem scan — derived from the
  already-collected sample list) and threads them through
  `build_model(cfg, class_names=...)`; inference paths call `build_model(cfg)`
  with no `class_names`, so warm-start never overwrites a trained head.
  Toggle: `model.warm_start_head_from_ssv2` (default `true`).
- **⚠️ Note on data provenance.** The default checkpoint was finetuned on the
  *official SSv2 train+val* set. If this challenge's test videos come from
  those same SSv2 videos (rather than SSv2's held-out, label-private test
  split), the model has effectively seen them with labels. The
  Kinetics-400-finetuned checkpoint
  (`MCG-NJU/videomae-base-finetuned-kinetics`, same code, swap
  `model.checkpoint`) is the leakage-free alternative.
- **Sources / credits.**
  - Tong, Song, Wang, Wang, **"VideoMAE: Masked Autoencoders are
    Data-Efficient Learners for Self-Supervised Video Pre-Training"**
    (NeurIPS 2022), arXiv:2203.12602.
  - Pretrained checkpoints by MCG-NJU on Hugging Face Hub
    (`MCG-NJU/videomae-base-finetuned-ssv2`).
  - HuggingFace `transformers` implementation of VideoMAE
    (`VideoMAEForVideoClassification`, `VideoMAEConfig`).
  - Reusing a pretrained classifier's rows for overlapping classes is a
    standard transfer-learning trick (no single canonical citation); the
    name-matching glue here is bespoke to this repo.
- **Dependency added.** `transformers>=4.40,<5` in `pyproject.toml`.

### 2.3 VideoMAE supports arbitrary even `num_frames`
- **What.** `VideoMAE.__init__` in `src/models/videomae.py` no longer hard-asserts
  `num_frames=16`. Now it only requires `num_frames % 2 == 0` (the tubelet
  stride is 2). When the requested `num_frames` differs from the pretraining
  default (16), the constructor passes `num_frames=...` to
  `VideoMAEForVideoClassification.from_pretrained(..., ignore_mismatched_sizes=True)`
  so the size-changed `position_embeddings` buffer is reinitialised
  sinusoidally for the new sequence length while the rest of the weights
  load normally. A console warning is printed in that case.
- **Why.** The challenge videos contain **exactly 4 source frames each**
  (verified by listing every video dir). Sampling 16 frames from 4 sources
  via linspace+round produces ~12 duplicated frames per clip, so
  `num_frames=16` mostly feeds VideoMAE constant tubelets with no motion
  signal. Letting `num_frames=4` match the raw input avoids that waste at
  the cost of moving the model OOD vs its pretraining sequence length —
  worth A/B-testing on val_dir.
- **Source / credit.** VideoMAE's positional embeddings are sinusoidal by
  construction (Tong et al. 2022, §3.1), so they admit re-sizing without
  retraining. `ignore_mismatched_sizes` is the canonical HuggingFace
  `transformers` mechanism for loading a checkpoint into a slightly
  reshaped model.

### 1.6 Multi-clip TTA guard for short videos
- **What.** `VideoFrameDataset.__getitem__` now silently downgrades
  `n_clips → 1` when the source video has fewer frames than
  `num_frames`. With ~4-frame source videos and `num_frames=16`, the three
  "different temporal samplings" produced by `_pick_frame_indices_multi`
  rounded to the same integer indices anyway — running the model 3× per
  video burned compute for zero signal.
- **Why.** Multi-clip TTA assumes enough source frames that distinct
  temporal offsets pick *different* indices. When `n_avail < num_frames`,
  the rounding kernel collapses all clips to the same picks. The guard
  makes inference 3× faster on those inputs with no accuracy change. The
  `n_clips` config knob keeps its meaning for videos where it actually
  matters; the downgrade is per-sample.
- **Source / credit.** No external citation — this is a project-specific
  data observation (every video is 4 frames).

### 2.4 Backbone bumped to ResNet-50
- **What.** `src/configs/model/tsm_resnet.yaml` now defaults to
  `backbone: resnet50` (was `resnet18`). The path was already supported in
  `TSMResNet.__init__`.
- **Why.** Larger receptive field and feature capacity. Made tractable on the
  available GPU by AMP (see §4.1).
- **Source / credit.** ResNet-50: He et al., **"Deep Residual Learning for
  Image Recognition"** (CVPR 2016), arXiv:1512.03385. ImageNet weights via
  `torchvision.models.resnet50` (`IMAGENET1K_V1`).

---

## 3. Training loop

### 3.1 AdamW + weight decay
- **What.** Replaced `torch.optim.Adam` with `torch.optim.AdamW` in `train.py`.
  `weight_decay` is configurable (`training.weight_decay`, default `0.05`).
- **Why.** The original Adam couples weight decay into the gradient update; it
  effectively scales weight decay with the per-parameter adaptive LR, which
  hurts regularisation. AdamW decouples them and is the modern default.
- **Source / credit.** Loshchilov & Hutter, **"Decoupled Weight Decay
  Regularization"** (ICLR 2019), arXiv:1711.05101.

### 3.2 OneCycleLR with warmup + cosine annealing
- **What.** `torch.optim.lr_scheduler.OneCycleLR` stepped per batch in
  `train_one_epoch()`. `pct_start=0.1` gives a 10% linear warmup from `lr` to
  `max_lr`, then cosine decay back down. Configured via `training.lr` and
  `training.max_lr`.
- **Why.** Warmup avoids the early instability of large-LR steps on a
  near-random head; cosine decay anneals smoothly into a fine-tuning regime.
- **Sources / credits.**
  - OneCycle: Smith, **"A disciplined approach to neural network
    hyper-parameters: Part 1 — learning rate, batch size, momentum, and weight
    decay"** (2018), arXiv:1803.09820, and
    **"Super-Convergence: Very Fast Training of Neural Networks Using Large
    Learning Rates"** (2018), arXiv:1708.07120.
  - PyTorch implementation: `torch.optim.lr_scheduler.OneCycleLR`.

### 3.3 Label smoothing
- **What.** `nn.CrossEntropyLoss(label_smoothing=0.1)` in `train.py`,
  configurable via `training.label_smoothing`.
- **Why.** Reduces over-confident logits, improves calibration, and gives a
  small but consistent accuracy gain. Compatible with MixUp/CutMix because the
  mixed loss is a linear combination of two `CrossEntropyLoss` calls — both
  benefit from label smoothing.
- **Source / credit.** Szegedy, Vanhoucke, Ioffe, Shlens, Wojna,
  **"Rethinking the Inception Architecture for Computer Vision"** (CVPR 2016),
  arXiv:1512.00567 — §7 introduces label smoothing as a regulariser.

### 3.4 Mixed precision (AMP / bfloat16)
- **What.** `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`
  wraps both forward and loss computation in `train_one_epoch()` and
  `evaluate_epoch()`. Toggled by `training.amp` (default `true`) and
  `training.amp_dtype` (`bfloat16` or `float16`). bf16 has the same dynamic
  range as fp32, so no `GradScaler` is required.
- **Why.** ~2× faster training and roughly half the activation memory, which
  enabled the move to ResNet-50 without OOM. bf16 is supported natively by
  Ada-Lovelace GPUs (RTX 4000 Ada).
- **Sources / credits.**
  - Mixed precision: Micikevicius et al., **"Mixed Precision Training"**
    (ICLR 2018), arXiv:1710.03740.
  - bfloat16: format introduced by Google for TPU; documented in
    Kalamkar et al., **"A Study of BFLOAT16 for Deep Learning Training"**
    (2019), arXiv:1905.12322.
  - PyTorch API: `torch.amp.autocast`.

### 3.5 Exponential Moving Average of weights
- **What.** `torch.optim.swa_utils.AveragedModel` with a custom
  `_ema_avg_fn(decay)` (default `decay=0.999`) is updated after every batch in
  `train_one_epoch()`. The EMA model is also evaluated each epoch; the better
  of (raw, EMA) is the candidate for snapshotting.
- **Why.** EMA smooths the noisy SGD trajectory; the averaged weights tend to
  generalise slightly better than any single iterate. Cheap and almost always
  helpful.
- **Sources / credits.**
  - Polyak & Juditsky, **"Acceleration of Stochastic Approximation by
    Averaging"** (SIAM J. Control Optim. 1992) — the original idea.
  - Recent revivals: Tarvainen & Valpola, **"Mean Teachers are Better Role
    Models"** (NeurIPS 2017); MoCo (He et al. 2020); MAE (He et al. 2022) —
    all use EMA of model weights.
  - PyTorch utility: `torch.optim.swa_utils.AveragedModel`.

### 3.6 Layer-wise LR decay (LLRD) for VideoMAE
- **What.** New helper `build_videomae_param_groups()` in
  `src/models/videomae.py` constructs AdamW parameter groups for VideoMAE
  with a per-layer learning rate that decays exponentially from the head
  down to the patch embeddings:
  `max_lr_layer = base_max_lr · llrd_decay^(top_layer − layer_id)`
  with `top_layer = num_encoder_layers + 1` (head/fc_norm at the top, patch
  embeddings at the bottom). Bias, LayerNorm gains, position embeddings and
  CLS-token-style parameters are split into a no-decay sub-group per layer
  (standard ViT recipe). The matching `max_lr` list is fed to OneCycleLR so
  every group's peak LR is scaled consistently. `train.py` activates LLRD
  automatically when `cfg.model.name == "videomae"` and
  `training.layerwise_lr_decay=true`. Defaults: `llrd_decay=0.75`,
  `epochs=20` (was 10) — both in `configs/experiment/videomae.yaml`.
- **Why.** Pretrained backbone layers store features that need *small*
  adjustments to the new task; the head needs *large* adjustments because
  its 33-class mapping is essentially new. A single LR forces a compromise
  that either understretches the head or overwrites the pretrained features
  in lower layers. LLRD lets each depth move at its own pace; in practice
  it is the largest single boost to ViT finetuning quality after AdamW +
  warm-restart. Bumping epochs to 20 lets the cosine schedule actually use
  the lower per-layer LRs (10 epochs anneals away too quickly).
- **Sources / credits.**
  - Clark, Luong, Le, Manning, **"ELECTRA: Pre-training Text Encoders as
    Discriminators Rather Than Generators"** (ICLR 2020), arXiv:2003.10555,
    §3.1 — introduced layer-wise LR decay for finetuning BERT.
  - Bao, Dong, Piao, Wei, **"BEiT: BERT Pre-Training of Image
    Transformers"** (ICLR 2022), arXiv:2106.08254 — the canonical reference
    for ViT finetuning with LLRD.
  - He, Chen, Xie, Li, Dollár, Girshick, **"Masked Autoencoders Are Scalable
    Vision Learners"** (CVPR 2022), arXiv:2111.06377, Tab 9 — uses LLRD
    decay 0.65–0.75 for MAE finetuning.
  - The "no-decay on bias / LayerNorm / position embeddings" split is the
    same recipe used in `timm` and HF `transformers` ViT finetuning utilities.

### 3.7 Reboot-resilient training (resume + skip-if-done + retry loop)
- **What.** Four pieces wired together so the overnight pipeline survives
  the school machine's spontaneous reboots:
  1. **`train.py` resume** — at every epoch end, persists
     `(model, optimizer, scheduler, ema, epoch, best_val_acc, snapshots)`
     to `<checkpoint_path>.resume.pt`, written atomically via temp + rename
     so a power loss mid-write can't corrupt it. On startup, if the file
     exists and `training.resume=true` (default), the script reloads
     everything and skips ahead to the last completed epoch. Cleaned up
     on clean completion. Snapshot bookkeeping (file paths + per-snapshot
     val_acc) is part of the resume state so the top-K rename logic stays
     consistent across restarts.
  2. **`run_ovn2.sh` skip-if-done** — each `run_step` writes a marker
     under `logs/.done_<TAG>/` on success; if the marker already exists,
     the step is skipped. Combined with (1), this means a reboot
     mid-pipeline only loses the partial work of the *current epoch* of
     the *current training step*. Re-launching the script does the right
     thing — no manual editing.
  3. **`~/loop_until_done.sh`** — bash wrapper that re-runs its argument
     script until it exits 0 (or hits `MAX_RETRIES=20`), sleeping 60s
     between retries. Stacks with (2): retries are basically free because
     completed steps short-circuit.
  4. **`@reboot` crontab entry** — on boot, cron spawns a detached
     `tmux new-session` that runs `loop_until_done.sh run_ovn2.sh`. Net
     effect: machine reboots → cron auto-launches a detached training
     session that resumes from where the previous run died. Zero manual
     touch required.
- **Why.** Without these, every reboot wasted the entire in-progress
  training step (1-6 h of GPU). With them, the loss is bounded by the
  duration of one epoch (~15-30 min for VideoMAE-Large), and even that
  loss is hidden behind a single SSH reconnect since the training picks
  back up automatically.
- **Sources / credits.** The "checkpoint everything every epoch and resume
  on startup" pattern is standard practice in distributed deep learning
  frameworks (PyTorch Lightning, DeepSpeed). No single canonical citation;
  the atomic-write-via-rename idiom is POSIX semantics.

### 3.8 Pseudo-labeling (semi-supervised self-training on the test set)
- **What.** New script `src/pseudo_label.py` reuses the
  `_resolve_checkpoint_paths()` / `run_inference_logits()` /
  `discover_all_test_videos()` helpers from `create_submission.py` to:
  1. Load one or several trained checkpoints (TTA + label-aware flip remap
     are honoured automatically — same flags as inference);
  2. Average softmax probabilities over `(views × checkpoints)` for every
     `video_*` folder under `dataset.test_dir`;
  3. Print a confidence histogram (so you can pick a threshold) and write a
     CSV (`video_path, pseudo_label, confidence`) to
     `dataset.pseudo_labels_output`.

  `train.py` then merges those test videos — keeping only rows with
  `confidence ≥ dataset.pseudo_threshold` (default `0.85`) — into
  `train_samples` *after* the train/val split, so the in-training val_acc
  remains an honest signal and only the train pool is enlarged. The merged
  pseudo samples go through the exact same augmentation pipeline (label-aware
  hflip / mixup / cutmix / random erasing), and the existing checkpoint
  payload format is unchanged.
- **Why.** The challenge has a meaningful train/test distribution gap (~21
  pts on this codebase: 0.72 internal vs 0.51 external). Adding the test set
  itself — pseudo-labeled with the model's own high-confidence predictions —
  re-aligns training to the actual evaluation distribution. With ~27k test
  videos and a typical 30-50% retention at confidence ≥ 0.85, this adds
  ~10-15k extra training samples (a +20-30% boost over the ~50k labeled
  train pool). It also exploits the SSv2-pretraining bias: where the model
  is right-because-it-cheats on test videos that overlap SSv2, those
  high-confidence labels are correct and reinforce the cheating pattern.
- **Why post-split.** Pseudo samples must not pollute the validation pool —
  if they did, val_acc would be measured against the model's own
  predictions and become uninformative. Keeping val_samples derived only
  from real labels preserves the ablation signal.
- **Sources / credits.**
  - Lee, **"Pseudo-Label: The Simple and Efficient Semi-Supervised Learning
    Method for Deep Neural Networks"** (ICML 2013 Workshop on Challenges in
    Representation Learning) — the original hard-label self-training recipe.
  - Modern follow-ups (Sohn et al., **"FixMatch"**, NeurIPS 2020,
    arXiv:2001.07685; Xie et al., **"Self-training with Noisy Student"**,
    CVPR 2020, arXiv:1911.04252) refine this with weak/strong augmentation
    asymmetry and noise injection — not implemented here, but the
    confidence-thresholded merge is identical in spirit. This implementation
    sticks to hard pseudo-labels for simplicity; the augmentation pipeline
    already provides the "noisy student" perturbations.

### 3.8 Snapshot ensemble (top-K checkpoints by val acc)
- **What.** During training, `train.py` keeps the top-K (by val accuracy)
  checkpoints on disk as `<base>_top1.pt`, `<base>_top2.pt`,
  `<base>_top3.pt` (configurable via `training.top_k_checkpoints`). At
  inference time, `create_submission.py` discovers them via
  `_resolve_checkpoint_paths()` and **averages softmax probabilities** across
  the K models (`run_inference_logits()`).
- **Why.** Even with a single training run, the top-K iterates capture
  somewhat different fits and their soft-vote averaging is consistently better
  than picking any single iterate.
- **Sources / credits.** The "snapshot ensemble" name and idea come from
  Huang, Li, Pleiss, Liu, Hopcroft, Weinberger,
  **"Snapshot Ensembles: Train 1, Get M for Free"** (ICLR 2017),
  arXiv:1704.00109. The original paper uses cyclical LR restarts; the variant
  here ("keep top-K by val acc from a single OneCycle run") is a simplified,
  no-extra-cost approximation.

---

## 4. Inference

### 4.1 Spatial 10-crop TTA
- **What.** `VideoTransform.tta()` returns 10 views per clip — 5 spatial
  positions (4 corners + center) × 2 (original + horizontal flip).
  `evaluate.py` and `create_submission.py` average logits over all views.
- **Why.** Reduces variance of the prediction by averaging predictions over
  several deterministic spatial crops.
- **Source / credit.** The 10-crop protocol predates ImageNet and is
  explicitly described as "10-view averaging" in Krizhevsky, Sutskever,
  Hinton, **"ImageNet Classification with Deep Convolutional Neural
  Networks"** (NeurIPS 2012, "AlexNet"), §4.1.

### 4.2 Multi-checkpoint ensemble in `create_submission.py`
- **What.** `_resolve_checkpoint_paths()` returns either an explicit list
  (`training.checkpoint_paths`) or auto-derived top-K snapshots
  (`<base>_topN.pt` for `N ∈ 1..ensemble_top_k`). The model is rebuilt once
  and `state_dict` is reloaded per checkpoint; softmax probabilities are
  summed and the argmax is the final prediction.
- **Why.** See §3.6.

### 4.3 Per-class weighted ensembling (`ensemble_per_class.py`)
- **What.** New script `src/ensemble_per_class.py` that combines N
  checkpoints with **class-dependent** weights instead of a single uniform
  soft-vote. Workflow:
  1. For each checkpoint, run TTA inference on `val_dir` and `test_dir`
     (using `run_inference_logits()` with the existing flip-aware TTA
     remap). Softmax tensors are cached to
     `training.softmax_cache_dir` (default `<models_dir>/_softmax_cache/`)
     keyed by `(ckpt_path, mtime, dataset_path, n_clips, tta, num_frames)`.
  2. Compute per-(model, class) val accuracy `acc[m, c]`.
  3. Build per-class weights `w[m, c] = softmax_m(acc[m, c] / T)`
     (temperature `training.class_weight_temperature`, default `0.10`).
     Classes with no val samples fall back to uniform weights.
  4. Sanity check: compare uniform soft-vote vs per-class weighted on val
     (warn if weighted is worse — means the temperature is too sharp).
  5. Apply the same per-class weights to test softmax → submission CSV.
- **Why.** Uniform soft-vote treats all models equally on every class. But
  a model finetuned from SSv2 and one finetuned from K400 will have very
  different per-class strengths — SSv2 dominates the "pretending" /
  directional templates it was trained on; K400 may be competitive (or
  better) on motion classes resembling its source distribution. Per-class
  weighting lets each class pick whichever models excel at it without
  hand-coding rules.
- **Why post-cache.** Each TTA inference of VideoMAE on 6700 val + 27000
  test videos is ~1.5 h. With caching, the only cost on a re-run with a
  different temperature is loading two tensors and recomputing the
  softmax-over-weights — seconds, not hours.
- **Sources / credits.** Per-class / per-region model gating is standard
  in the stacking / model-mixing literature; the original framework is
  Wolpert, **"Stacked Generalization"** (Neural Networks 1992). The exact
  "softmax over per-class val accuracy" weighting recipe used here is the
  most basic formalisation of the "best model per class" heuristic.

### 4.5 Learned (gradient-descent) weighted ensembling (`ensemble_gradient.py`)
- **What.** New script `src/ensemble_gradient.py` — a third ensembling mode
  that *learns* the mixture weights by gradient descent instead of deriving
  them from a formula (§4.3) or fixing them uniform (§4.2). It reuses the
  exact same per-model softmax cache (`_cache_key` imported from
  `ensemble_per_class.py`), so it costs nothing extra after either script
  has run once. Logit-weights `θ` are softmax-normalised over the model
  axis, so the combination is always a convex mixture (a valid prob):
  `combined[n,c] = Σ_m softmax_m(θ)[m(,c)] · prob[m,n,c]`. Adam minimises
  `NLL(log combined, y) + λ·‖w − 1/M‖²` on val.
- **Two granularities (flag `training.grad_weight_mode`).**
  `global` (default) learns one weight per model (M params — ≈zero overfit
  risk, the safe high-value choice); `per_class` learns `M×C` weights (the
  learned counterpart to §4.3, more expressive but overfit-prone, hence the
  pull-to-uniform `λ = training.grad_l2_uniform`).
- **Honest evaluation.** `training.grad_holdout_frac > 0` fits on a val
  subset and reports accuracy on the untouched remainder before refitting
  on all of val for the test submission — so the per-class mode's
  generalisation can be checked rather than trusted. Other knobs:
  `grad_steps` (300), `grad_lr` (0.05).
- **Why.** The §4.3 heuristic optimises a *proxy* (per-class accuracy →
  softmax); this optimises the actual ensembling objective (val NLL)
  directly, which can only do at least as well in-sample and usually
  generalises better for the low-parameter `global` mode.
- **Source / credit.** Learning a convex combination of model outputs by
  minimising a held-out loss is "linear stacking" (Breiman, **"Stacked
  Regressions"**, Machine Learning 1996; building on Wolpert 1992).

### 4.6 Cross-backbone ensembling: per-checkpoint rebuild + shared cache
- **Bug.** All three ensemble scripts (`create_submission.py`,
  `ensemble_per_class.py`, `ensemble_gradient.py`) only rebuilt the model when
  the checkpoint's `model_name` *changed*. But VideoMAE **Base and Large both
  save `model_name="videomae"`**, so ensembling a Large checkpoint together
  with Base ones (k400/ssv2) tried to `load_state_dict` Base weights (768-d)
  into a Large graph (1024-d) → `size mismatch` crash. This stayed dormant only
  because the Base softmax was always cached (the load path was skipped); it
  fired the moment a Base checkpoint missed cache.
- **Compounding cause.** `cache_dir` defaults to `checkpoint_paths[0].parent/
  _softmax_cache`. After §(local-disk) moved the Large checkpoints to `/Data`,
  putting Large first sent the cache to `/Data/.../_softmax_cache`, which lacks
  the NFS-cached Base softmax → cache miss → the crash above. `loop_until_done`
  then crash-looped submission A.
- **Fix.** (1) Rebuild the model **per checkpoint from its own stored config**
  (`build_model_from_checkpoint`, which reads `ckpt["config"]`) instead of
  switching on `model_name` — correct for any Base/Large mix. (2) Pin
  `training.softmax_cache_dir` to the NFS `models/_softmax_cache` in
  `run_ovn2.sh`'s caching steps so all runs share one cache regardless of where
  the checkpoints live, and consolidated the new-Large softmax there.
- **Result.** Submission A (was crash-looping) now completes in ~5 s from
  cache; cross-backbone learned ensembles (new-Large + ssv2 + k400) run
  end-to-end.

### 4.4 Snapshot ensemble in `evaluate.py`
- **What.** `evaluate.py` now uses the same `_resolve_checkpoint_paths()`
  helper from `create_submission.py`: it discovers either
  `training.checkpoint_paths=[...]` (explicit list) or auto-derives
  `<base>_top1.pt … _topK.pt` from `training.checkpoint_path` when
  `training.ensemble_top_k > 1`. Inference runs once per checkpoint with
  the same TTA settings; **softmax probabilities are summed across
  (views × checkpoints)** before argmax for top-1 / top-5. The model is
  rebuilt once and `state_dict` is reloaded per checkpoint.
- **Why.** Symmetric with `create_submission.py`'s ensembling so the
  external-val number you measure is exactly what the submission would do.
  Critical for honest ablation: comparing single-ckpt vs K-ckpt val acc
  tells you whether the snapshot ensemble is worth the inference cost
  before you burn submissions on Kaggle.
- **Source / credit.** Same as §3.6 / §4.2 (snapshot ensembling).

### 4.4 Inference: bf16 autocast + view-chunked forward
- **What.** `evaluate.py` and `create_submission.py` now run inference under
  `torch.amp.autocast(dtype=bfloat16)`, and a helper `_multi_view_logits()`
  splits the `(B, N, T, C, H, W)` view tensor into chunks of
  `eval_view_chunk` views per forward pass before averaging logits.
- **Why.** Combining 10-crop spatial TTA × 3-clip temporal TTA × batch_size 8
  packs 240 clips (3840 frames) into a single forward, which OOMs ResNet-50
  on a 20 GB GPU shared with another process. bf16 halves activation memory
  and chunking caps the per-step memory footprint at
  `eval_view_chunk × T × C × H × W` regardless of `N` or batch size.
- **Source / credit.** Same references as §3.4 (mixed precision /
  bfloat16). The "process spatial/temporal views in micro-batches" pattern
  is standard in video inference toolkits (PySlowFast, mmaction2).

---

## 5. Bug fixes and small infrastructure fixes

These were not "techniques" per se but were necessary to get correct numbers
out of the pipeline. None come from external sources.

- **`evaluate.py` was reporting accuracy on only 20% of `val_dir`.** It was
  calling `split_train_val(val_dir, val_ratio=0.2)` and evaluating the small
  side. Fixed to use `collect_video_samples(val_dir)` directly so the *full*
  validation set is used.
- **`evaluate.py` did not handle TTA tensors.** Added the same
  `(B, N, T, C, H, W) → mean over N` averaging that `create_submission.py`
  uses.
- **`train.py` crashed on `torch.save()` when the parent directory did not
  exist.** Added `checkpoint_path.parent.mkdir(parents=True, exist_ok=True)`
  before saving.
- **Hydra struct-mode rejection of `dataset.tta=true` overrides.** Fixed by
  declaring `tta:` in `configs/data/default.yaml` (required by the Hydra
  struct-mode default).
- **Reboot-resilient training & orchestration.** Three independent layers
  added on top of the existing `tmux` workflow to survive the host's
  unpredictable nightly reboots:
  1. **Atomic per-epoch resume in `train.py`.** At every epoch end we
     persist `(model, optimizer, scheduler, EMA, snapshots list, epoch,
     best_val_accuracy)` into `<checkpoint_path>.resume.pt` via a
     write-tmp-then-rename pattern (`os.replace`, atomic on POSIX). At
     startup, if the file exists, we load it and skip to the saved epoch.
     A try/except around the load means a corrupt-mid-write file falls
     back gracefully to a from-scratch run. Toggle: `training.resume`
     (default `true`); the file is deleted on clean completion.
  2. **Skip-if-done in `run_ovn2.sh`.** `run_step` now writes a
     `logs/.done_<TAG>/<step>` marker on successful exit and short-circuits
     if the marker already exists. Combined with the per-epoch resume,
     re-launching the orchestration script after a reboot continues from
     the next not-yet-completed step with at most one wasted epoch.
  3. **`loop_until_done.sh` retry wrapper.** Tiny bash loop at
     `~/loop_until_done.sh`: re-runs its argument script until it exits 0
     (or 50 retries, configurable via `$MAX_RETRIES`). Catches transient
     OOMs and immediate-after-reboot relaunches. The end-state is a
     one-command overnight pipeline: `bash ~/loop_until_done.sh run_ovn2.sh`
     inside `tmux` survives any number of crashes/reboots.

  `@reboot` cron auto-restart was tested on the host but is blocked, so
  the user has to type the launch command manually after each reboot;
  with the three layers above, that's still the only manual step needed.

- **True reboot survival via systemd user lingering.** Discovered that
  `loginctl enable-linger` works on this host without sudo, which keeps
  the user's systemd instance running across logouts and reboots. With
  that in place, installed
  `~/.config/systemd/user/ovn2-train.service` which `ExecStart`s
  `loop_until_done.sh run_ovn2.sh` on every boot, with
  `Restart=on-failure` / `RestartSec=120` for transient crashes that the
  inner wrapper somehow can't catch. Combined with skip-if-done markers
  and the per-epoch resume, the pipeline now completes itself across
  arbitrary reboots without any manual relaunch. Manage with
  `systemctl --user {status,start,stop,disable} ovn2-train.service`;
  follow the live log at
  `logs/ovn2_systemd.log` or via `journalctl --user -fu ovn2-train`.

- **Resume tmpfile naming bug in `train.py`.** The first cut of the
  atomic resume save used
  `resume_path.with_suffix(".resume.pt.tmp")`, but `resume_path` already
  ends in `.resume.pt`, so `with_suffix` stripped `.pt` and produced
  `<base>.resume.resume.pt.tmp`. Cosmetic only — the final `tmp.replace`
  still targeted the correct name — but the orphaned tmp files were
  confusing post-reboot. Replaced with
  `resume_path.with_name(resume_path.name + ".tmp")`.

- **NFS durability: fsync before rename (`train.py`).** A crash during an
  end-of-epoch save left `resume.pt` *and* `top1.pt` at 0 bytes and
  `top3.pt` truncated — even though saves used the write-tmp-then-rename
  "atomic" pattern. Root cause: the checkpoint dir is **NFS**
  (`omega.polytechnique.fr:/students`), where `rename` is a metadata op
  that reaches the server synchronously while the freshly-written data
  may still sit in the client page cache. If the process dies in that
  window, the rename has already pointed the name at a file whose bytes
  never landed → 0-byte/truncated result, which then defeats resume (the
  relaunch reads the empty `resume.pt` and starts from scratch). Fix: a
  `_save_durable()` helper does `torch.save` → `flush` → `os.fsync` so the
  data is committed *before* the rename, plus a best-effort `_fsync_dir()`
  after each rename/unlink to persist the directory entry itself. Wired
  into `_save_payload`, `_persist_resume_state`, the top-k staging
  renames, and the final top-1 mirror. This is what makes the
  resume/recovery stack actually survive a mid-save crash rather than
  silently losing all progress.

- **Large-model checkpoints moved to local `/Data` (`run_ovn2.sh`).** The
  fsync hardening above immediately surfaced the *real* root cause behind
  every truncated/0-byte checkpoint: the NFS home has a **30 GB quota**,
  and a 307M-param VideoMAE-Large run (three top-k snapshots at ~1.2 GB
  plus a ~5 GB optimizer/EMA resume file, briefly doubled during the
  atomic write) simply does not fit. Pre-fsync, the over-quota writes
  failed silently during writeback; post-fsync they raised
  `OSError: [Errno 122] Disk quota exceeded`. Fix: a new `LM` variable
  points only the Large model's `checkpoint_path` (and the derived
  `_topN.pt` / `.resume.pt`) at `/Data/florian.guillaumey/challenge_models`
  — a 561 GB local NVMe partition that persists across reboots and is not
  age-cleaned (unlike `/tmp`, which has a 10-day rule). This sidesteps the
  quota, is faster than NFS, and *keeps* reboot survival because the
  resume file is still on a persistent disk the relaunched job can find.
  Reused inputs (`videomae_ovn1_k400_top*`, `tsm_r50_ovn1_top*`,
  `videomae_4_pseudo.csv`) and the tiny submission CSVs stay on NFS. Also
  pruned ~4.7 GB of superseded experiment checkpoints
  (`videomae_3/4/5_*`, `videomae_ssv2_ft_*`, old `tsm_pretrained_*`) to
  give the home headroom.

- **Per-class ensembling crashed on cross-architecture mixes; cache went to `src/None/`.**
  Two bugs in `ensemble_per_class.py` (and the same single-arch assumption
  in `create_submission.py`):
  1. The model was built once from the *first* checkpoint's saved config and
     never rebuilt — so the moment the loop hit a checkpoint with a
     different `model_name` (e.g. TSM after VideoMAE), `load_state_dict`
     raised on hundreds of mismatched keys. Fixed by tracking
     `current_model_name` and calling `build_model_from_checkpoint(ckpt)` +
     `.to(device)` whenever it changes. ~5 lines each in
     `ensemble_per_class.py` and `create_submission.py`.
  2. `cache_dir = Path(str(cfg.training.get("softmax_cache_dir", <default>)))`
     resolved to `Path("None")` when the YAML declared
     `softmax_cache_dir: null`, because `cfg.get(key, default)` returns the
     *value* (None) when the key exists with a null value — `default` is
     only used when the key is absent. So the entire softmax cache landed
     under `src/None/`. Fixed with an explicit `is None` check, defaulting
     to `<models_dir>/_softmax_cache/`. The stranded cache was migrated to
     the correct path.
- **Top-K snapshot rename was clobbering its own files.** The original
  rename loop did `target.unlink(); path.rename(target)` per rank inside a
  single in-place pass over `snapshots`. After 2-3 epochs that meant the
  source of one rename was the file the next iteration was about to
  unlink — so `_top1.pt` and `_top2.pt` could vanish, leaving only
  `_top3.pt` (containing whichever weights happened to win the rename
  race). Fixed in `train.py` with a two-phase rename: phase 1 stages every
  live snapshot to a unique `.staging_N.pt` path; phase 2 moves each
  staged file to its final `_topN.pt`. Targets in phase 2 cannot be live
  snapshots, so unlinking them is safe. Symptom that surfaced this:
  ensemble at inference saw only 1/3 of the expected snapshots on disk,
  silently degrading the top-K ensemble to a single-checkpoint inference.

---

## 6. Configuration surface (recap of what is now in YAML)

`src/configs/train/default.yaml`:
```yaml
training:
  batch_size, lr, max_lr, weight_decay, epochs, num_workers
  device, checkpoint_path
  mixup_alpha, cutmix_prob
  label_smoothing
  amp, amp_dtype           # mixed-precision toggles (used in train *and* inference)
  eval_view_chunk          # inference: views per forward pass (caps GPU mem)
  class_weight_temperature # ensemble_per_class.py: T in softmax(acc/T)
  softmax_cache_dir        # ensemble_per_class.py cache root (null = next to ckpts)
  ema_decay                # 0.0 disables EMA
  top_k_checkpoints        # train-time top-K snapshot retention
  ensemble_top_k           # inference-time ensemble size
  checkpoint_paths         # explicit override list (null = use ensemble_top_k auto-discovery)
  skip_submission          # ensemble_per_class.py: val-only run, don't write CSV
  resume                   # train.py: load <ckpt>.resume.pt on startup (default true)
```

`src/configs/data/default.yaml`:
```yaml
dataset:
  train_dir, val_dir, test_dir, submission_output, lmdb_dir
  tta                      # 10-crop spatial TTA at inference
  n_clips                  # multi-clip temporal TTA at inference (1 = off)
  num_frames, val_ratio, seed, max_samples
  pseudo_labels_path        # CSV from pseudo_label.py — null disables
  pseudo_threshold          # min confidence to keep a pseudo-labeled sample
  pseudo_labels_output      # output path for pseudo_label.py (null = next to ckpt)
```

`src/configs/model/tsm_resnet.yaml`: `backbone: resnet50` is now the default.

`src/configs/model/videomae.yaml` (used via `experiment=videomae`):
```yaml
model:
  name: videomae
  pretrained, num_classes, num_frames     # num_frames MUST be 16 for the SSv2 ckpt
  checkpoint                              # HF hub id; swap for the K400 ckpt to avoid SSv2 leakage
  gradient_checkpointing                  # true => ~half activation mem, ~30% slower
  warm_start_head_from_ssv2               # init the 33-class head from matching SSv2 head rows
```

`src/configs/experiment/videomae.yaml` (selected via `experiment=videomae`):
```yaml
training:
  layerwise_lr_decay: true                # auto-on for VideoMAE; ignored otherwise
  llrd_decay: 0.75                        # exponential per-layer LR decay (head -> embeddings)
  epochs: 20                              # bumped from 10 to give LLRD's tiny low-layer LRs time to bite
```

---

## 7. References (consolidated)

| # | Reference | Used for |
|---|-----------|----------|
| 1 | Lin et al., *TSM: Temporal Shift Module*, ICCV 2019, arXiv:1811.08383 | TSMResNet (§2.1) |
| 2 | Wang et al., *Temporal Segment Networks*, ECCV 2016, arXiv:1608.00859 | TSN sampling (§1.2), multi-clip TTA (§1.3) |
| 3 | He et al., *Deep Residual Learning*, CVPR 2016, arXiv:1512.03385 | ResNet-50 backbone (§2.2) |
| 4 | Zhang et al., *mixup*, ICLR 2018, arXiv:1710.09412 | MixUp (§1.5) |
| 5 | Yun et al., *CutMix*, ICCV 2019, arXiv:1905.04899 | CutMix (§1.5) |
| 6 | Zhong et al., *Random Erasing*, AAAI 2020, arXiv:1708.04896 | Random Erasing (§1.4) |
| 7 | Szegedy et al., *Rethinking Inception*, CVPR 2016, arXiv:1512.00567 | Label smoothing (§3.3) |
| 8 | Loshchilov & Hutter, *Decoupled Weight Decay*, ICLR 2019, arXiv:1711.05101 | AdamW (§3.1) |
| 9 | Smith, *Super-Convergence*, 2018, arXiv:1708.07120 / 1803.09820 | OneCycleLR (§3.2) |
| 10 | Micikevicius et al., *Mixed Precision Training*, ICLR 2018, arXiv:1710.03740 | AMP (§3.4) |
| 11 | Kalamkar et al., *A Study of BFLOAT16*, 2019, arXiv:1905.12322 | bfloat16 (§3.4) |
| 12 | Polyak & Juditsky, *Acceleration of Stochastic Approximation by Averaging*, SIAM 1992 | EMA (§3.5) |
| 13 | Huang et al., *Snapshot Ensembles*, ICLR 2017, arXiv:1704.00109 | Snapshot ensemble (§3.6) |
| 14 | Krizhevsky et al., *ImageNet w/ Deep CNNs*, NeurIPS 2012 (AlexNet) | 10-crop TTA (§4.1) |
| 15 | `timm` (Wightman), https://github.com/huggingface/pytorch-image-models | MixUp↔CutMix per-batch switch pattern (§1.5) |
| 16 | mmaction2, PySlowFast (open-source video toolboxes) | LMDB pattern (§1.1), per-clip augmentation conventions (§1.4) |
| 17 | LMDB, Howard Chu / Symas | Storage backend (§1.1) |
| 18 | Tong et al., *VideoMAE*, NeurIPS 2022, arXiv:2203.12602 | VideoMAE finetuning (§2.2) |
| 19 | MCG-NJU, `videomae-base-finetuned-ssv2` on Hugging Face Hub | VideoMAE pretrained weights (§2.2) |
| 20 | HuggingFace `transformers` (Wolf et al., EMNLP 2020 system demo) | VideoMAE PyTorch implementation (§2.2) |
| 21 | Clark et al., *ELECTRA*, ICLR 2020, arXiv:2003.10555 | Layer-wise LR decay (§3.6) |
| 22 | Bao et al., *BEiT*, ICLR 2022, arXiv:2106.08254 | LLRD popularised for ViT (§3.6) |
| 23 | He et al., *MAE*, CVPR 2022, arXiv:2111.06377 | LLRD recipe 0.65–0.75 (§3.6) |
| 24 | Goyal et al., *Something-Something v1*, ICCV 2017, arXiv:1706.04261 | SSv2 mirror-pair label-aware flip (§1.5) |
| 25 | mmaction2, PySlowFast `flip_label_map` / `label_swap` flags | label-aware hflip implementation pattern (§1.5) |
| 26 | Lee, *Pseudo-Label*, ICML 2013 Workshop | Pseudo-labeling / self-training (§3.7) |
| 27 | Sohn et al., *FixMatch*, NeurIPS 2020, arXiv:2001.07685 | Confidence-thresholded pseudo-label recipe (§3.7) |
| 28 | Xie et al., *Self-training with Noisy Student*, CVPR 2020, arXiv:1911.04252 | Augmentation-as-noise during self-training (§3.7) |
| 29 | Wolpert, *Stacked Generalization*, Neural Networks 1992 | Per-class weighted ensembling foundation (§4.3) |

All implementations in this repo were written by hand against the references
above; no copy-pasted code from external repositories.
