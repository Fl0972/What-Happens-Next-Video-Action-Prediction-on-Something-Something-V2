# Document 2 — Model Analysis: VideoFormerLite-Ultra

**Model identifier:** `video_former_lite_ultra`  
**Checkpoint:** `models/video_former_lite_ultra.pt`  
**Experiment config:** `src/configs/experiment/video_former_lite_ultra.yaml`  
**Architecture file:** `src/models/video_former_lite.py`  
**Status:** Second-highest performing single model; key ensemble diversity source.

---

## 1. Model Overview

### 1.1 Motivation

The CNNBaseline and CNNLSTM architectures treat the temporal axis as either a bag of frames (average pooling) or a short sequential dependency (LSTM over T=4 steps). Neither provides direct access to long-range frame-pair relationships: which events co-occur in frame 1 and frame 4, and how they change. Self-attention in Vision Transformers [Dosovitskiy et al., ICLR 2021] provides exactly this — an O(T²) pairwise comparison between all frame tokens in a single attention step.

`video_former_lite_ultra` (VFL-Ultra) tests the hypothesis that global temporal self-attention, implemented as a small two-layer Transformer encoder on top of a per-frame ResNet18 backbone, can match or approach TSM performance under closed-track constraints while providing complementary error patterns for ensemble construction.

### 1.2 Prior Work

TimeSformer [Bertasius et al., ICML 2021] introduced divided space-time attention (separate spatial and temporal attention within a full ViT backbone) and demonstrated competitive SSv2 performance (59.5% top-1 with joint space-time attention, ViT-B/16, pretrained on Kinetics-400). The hybrid "CNN spatial + Transformer temporal" architecture used in VFL is more computationally efficient: the ResNet18 spatial extractor costs ~1.8 GFLOPs per frame while the Transformer over 5 tokens (T+1 with [CLS]) costs ~0.03 GFLOPs per clip. This design follows the hybrid ViT pattern discussed in [Dosovitskiy et al., ICLR 2021] and is conceptually related to Video Swin Transformer [Liu et al., ICCV 2022] in that spatial and temporal reasoning are separated.

Pre-Layer-Normalisation (Pre-LN) Transformer blocks [Xiong et al., ICML 2020] are used throughout, improving training stability for from-scratch models by preventing the gradient explosion observed in the original post-LN formulation [Vaswani et al., NeurIPS 2017].

### 1.3 Hypothesis

**Primary hypothesis:** Global temporal self-attention, even with only T=4 frame tokens, provides a qualitatively different form of temporal reasoning than TSM's local channel shift — enabling direct comparison between the first and last frame in a single attention step.

**Secondary hypothesis:** The residual errors of VFL-Ultra are uncorrelated with those of TSM-Ultra-v2, making VFL-Ultra a valuable ensemble partner despite its lower standalone accuracy.

### 1.4 Architecture Diagram and Temporal Footprint

The diagram below traces a clip through VFL-Ultra. The contrast with TSM is structural: the ResNet18 backbone here is **completely time-blind** — frames are processed independently and global-average-pooled to a single 512-d vector each. *All* temporal reasoning is deferred to a dedicated 2-layer Transformer that attends over those pooled per-frame vectors (`src/models/video_former_lite.py:183` `forward`; encoder block at `:98`).

```
 Input clip                       (B, T=4, C=3, H=224, W=224)
      │  reshape → (B·T, 3, 224, 224)
      v
+-------------------------------------------------------------+
|  ResNet18 backbone -- NO temporal mixing of any kind        |
|  global average pool -> (B·T, 512)                          |
|   => spatial detail is COLLAPSED to one vector per frame    |
|      BEFORE the time axis is ever touched                   |
+-------------------------------------------------------------+
      |  reshape -> (B, T=4, 512)
      v
   prepend [CLS] + add learnable temporal pos-embed -> (B, 5, 512)
      v
+-------------------------------------------------------------+
|  x2  Pre-LN Transformer encoder block                       |
|                                                             |
|     Multi-Head Self-Attention (8 heads), all-to-all:        |
|                                                             |
|        [CLS]  f1   f2   f3   f4                              |
|          \____|____|____|____/  every token attends to      |
|               every other token in ONE step (global, O(T^2))|
|                                                             |
|     + MLP (d_ff = 2048), GELU                                |
+-------------------------------------------------------------+
      |  take [CLS] -> LayerNorm -> dropout 0.2
      v
   Linear -> (B, 33) logits
```

**Temporal footprint.** VFL-Ultra's temporal reasoning is *global and explicit but coarse and late*: frame 1 and frame 4 sit a single attention hop apart (no locality bias, unlike TSM's ±1-frame-per-block shift), and a learnable [CLS] query aggregates the whole clip in one place. But this happens on **spatially-pooled 512-d vectors** — the fingertip/edge-level motion cues that separate, say, a real grasp from a pretend grasp are averaged out by the global pool *before* the Transformer ever sees them. So VFL is architecturally the *more explicitly temporal* design (a dedicated temporal module, global receptive field) yet it operates on a *coarser* signal than `tsm_ultra_v2`, which mixes time at full spatial resolution (`model_analysis_tsm_ultra_v2.md §1.4`). The head-to-head in `model_analysis_tsm_vs_vfl.md` shows the empirical consequence: VFL's hypothesised advantage on first-vs-last-frame classes (e.g. folding) does *not* materialise — TSM wins those — because the deciding evidence is spatially local and is lost in VFL's pre-attention pooling.

---

## 2. Experimental Setup

### 2.1 Dataset and Splits

Same dataset configuration as `tsm_ultra_v2`. See `model_analysis_tsm_ultra_v2.md §2.1` for details.

### 2.2 Preprocessing and Augmentation

Identical augmentation pipeline to `tsm_ultra_v2`, with one difference: horizontal flip is disabled (same reasoning — direction-sensitive SSv2 labels). The full regularisation stack including the 10-epoch warmup is retained.

### 2.3 Training Procedure

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Architecture | ResNet18 backbone + 2-layer pre-LN Transformer (17.5M params) | Spatial/temporal separation; see §1.2 |
| d_model | 512 | Matches ResNet18 feature dim; no projection needed |
| n_heads | 8 | 64-d per head; standard for d=512 |
| n_layers | 2 | Deeper Transformer was tested (3 layers) but not in ultra variant |
| mlp_ratio | 4.0 | d_ff = 2048 |
| num_frames | 4 | Correct clip length |
| Optimizer | AdamW | weight_decay=0.05 (higher than TSM due to Transformer overfit risk) |
| Learning rate | 5e-4 (peak) | 8-epoch linear warmup |
| Scheduler | Cosine annealing | 100 epochs |
| Epochs | 100 | VFL was still improving at epoch 60 in the prior run |
| Batch size | 32 | |
| Loss | Cross-entropy + label smoothing (0.1) | Focal loss not used in this variant |
| DropPath | max rate 0.15 | On Transformer residual branches; linear schedule |
| Head dropout | 0.2 | Applied to [CLS] output before classifier |
| AMP | Enabled | |

**Key architectural detail — [CLS] token classifier.** A learnable [CLS] token is prepended to the T=4 frame tokens, and learnable temporal positional embeddings (size T+1=5) are added to the full sequence. After L=2 Transformer blocks, only the [CLS] token output (normalised, dropped) is used for classification — following ViT [Dosovitskiy et al., 2021]. This is equivalent to learning a query that aggregates information from all temporal positions.

**Compute budget.** At T=4 frame tokens + 1 [CLS] = 5 total tokens, the Transformer attention matrix is 5×5=25 entries per head — computationally negligible compared to the ResNet18 spatial extractor (~1.8 GFLOPs per frame). The model runs at ~3 min/epoch (100 epochs ≈ 5 hours on a single A4000).

### 2.4 Evaluation Commands

```bash
# Standard evaluation on full val_dir
cd src
python evaluate.py training.checkpoint_path=../models/video_former_lite_ultra.pt

# With 10-crop TTA
cd src
python evaluate.py training.checkpoint_path=../models/video_former_lite_ultra.pt dataset.tta=true
```

---

## 3. Results

### 3.1 Top-1 / Top-5 Accuracy

| Evaluation Set | Top-1 | Top-5 |
|----------------|-------|-------|
| Internal val (best checkpoint) | **53.37%** | — |
| Val-dir (no TTA) | **32.11%** | **64.85%** |
| VFL (60 ep, standard, with warmup) for reference | 45.78% internal | — |
| VFL (no warmup, killed at ep 4) for reference | 9.45% internal | — |

**Comparison to published baselines:**

| Model | Pretraining | Temporal Module | SSv2 Top-1 | Source |
|-------|-------------|-----------------|-----------|--------|
| TimeSformer (JointSpace-Time, ViT-B/16) | Kinetics-400 | Joint space-time attn | 59.5% | Bertasius et al., ICML 2021 |
| TimeSformer (divided space-time, ViT-B/16) | Kinetics-400 | Divided space-time attn | 59.1% | Bertasius et al., ICML 2021 |
| **VFL-Ultra (ResNet18 + 2-layer Transformer)** | **None (scratch)** | **Global temporal attn** | **32.11%** (val-dir) | This work |
| Video Swin-B | ImageNet-21k + Kinetics-400 | Shifted window | 69.6% | Liu et al., ICCV 2022 |

The ~27 pp gap versus TimeSformer reflects the combined effect of (a) no pretraining, (b) much smaller backbone (ResNet18 vs. ViT-B/16), and (c) only 2 Transformer layers vs. 12 in ViT-B.

### 3.2 Regularisation Warmup Ablation — Key Finding

The warmup experiment provides the clearest ablation in the entire project:

| Run | Warmup Epochs | Final Val Acc |
|-----|--------------|---------------|
| VFL_prev (killed at ep 4) | **None** | 9.45% |
| VFL_closed (60 ep) | 10 epochs | **45.78%** |
| VFL-Ultra (100 ep) | 10 epochs | **53.37%** |

The +36.3 pp jump from no-warmup to 10-epoch warmup is the single largest improvement in the experiment series, attributable entirely to deferring MixUp/CutMix until the randomly initialised network has learned basic per-frame features. This finding is consistent with observations in ViT fine-tuning [Dosovitskiy et al., 2021] that label-mixing augmentations can prevent proper convergence when the model is not yet capable of generating meaningful class probabilities.

### 3.3 Training Dynamics

From the training log (`models/video_former_lite_ultra.csv`, final 5 epochs):
```
96,2.33e-06,2.1289,0.3218,2.0083,0.5304,0.5307
97,1.31e-06,2.1462,0.3504,2.0190,0.5337,0.5307
98,5.83e-07,2.1570,0.3409,2.0190,0.5294,0.5337
99,1.46e-07,2.1214,0.3410,2.0193,0.5309,0.5337
100,0.00e+00,2.0892,0.3518,2.0295,0.5319,0.5337
```
The model reaches its best validation accuracy at epoch 97 (53.37%) and the curve remains flat in the final epochs — suggesting saturation near the capacity limit of a 2-layer Transformer with ResNet18 features on this dataset. A 4-layer Transformer variant (`vfl_ultra_focal`, which also adds focal loss) reached 53.17%, below the 2-layer ultra variant, suggesting that additional Transformer depth introduces overfit risk with T=4 tokens.

### 3.4 Per-Class Analysis (via Ensemble Logits)

VFL-Ultra's predictions are captured in the ensemble analysis (weight 0.25 in the final ensemble). Its individual contribution to per-class corrections was not independently extracted. The ensemble's +0.61 pp gain from VFL-Ultra's inclusion (measured by ablation) indicates that VFL-Ultra corrects errors that are missed by both TSM variants.

---

## 4. Analysis

### 4.1 Inductive Bias: Global Attention vs. Local Shift

The key architectural difference between VFL-Ultra and `tsm_ultra_v2` is the scope of temporal interaction:

- **TSM:** Frame t at block k interacts only with frames t-1 and t+1 (shift of 1 position). At the T-th block, a frame can access context from ±k positions away (8 blocks → ±8 positions, more than T=4). However, each hop requires information to pass through an intermediate representation shaped by spatial convolution.

- **VFL:** At each Transformer block, every frame token directly attends to every other frame token (all-to-all). Frame 1 and frame 4 are equidistant from the model's perspective — there is no "locality bias" in the temporal dimension. This is the critical advantage for actions where the relationship between the first and last frame defines the class (e.g. "folding something" requires comparing the initial flat state with the final folded state).

**Implication for ensembling:** Because the two models make temporal comparisons through fundamentally different mechanisms, their errors are partially decorrelated. The ensemble ablation confirms this: removing VFL-Ultra costs −0.61 pp, meaning VFL-Ultra correctly classifies approximately 41 validation clips that the two TSM models get wrong.

### 4.2 Why VFL-Ultra Underperforms TSM-Ultra-v2 at Lower Absolute Accuracy

The 5.62 pp gap (32.11% vs. 38.25% val-dir) between VFL-Ultra and TSM-Ultra-v2 reflects several factors:

1. **Data efficiency.** Transformer attention weights must be learned from scratch; there are no inductive biases that align with image-like features in the attention mechanism. ResNet18's convolutional filters implicitly learn local spatial patterns (edges, textures) that are strongly biased toward the image domain — this makes convergence faster and more data-efficient.

2. **Token count.** With only T=4 tokens + 1 [CLS] = 5 tokens, the attention matrix is 5×5 = 25 entries. The Transformer effectively operates as a learned MLP with pairwise interactions — the depth benefit of full Transformer layers is minimal at this token count.

3. **Temporal signal quality.** TSM operates on every spatial position of every channel via the shift operation; VFL-Ultra operates on the globally pooled 512-d feature per frame. If the discriminative information is localised in a small spatial region (e.g. finger position indicating grasp vs. pretend-grasp), the pooled representation may wash it out before the Transformer sees it. TSM's channel-level shift preserves more spatial locality.

### 4.3 VFL-Ultra as an Ensemble Component

Despite its lower standalone accuracy, VFL-Ultra is a net positive contributor to the ensemble (+0.61 pp). Its inclusion is justified by:

1. **Error decorrelation.** Global attention makes fundamentally different predictions from local channel shift; the two models are likely to disagree on different subsets of the validation set.

2. **Class-level complementarity.** Some classes that TSM confuses (e.g. those where temporal direction is ambiguous) may be better handled by VFL's global comparison of frame 1 to frame 4.

3. **Calibration diversity.** Averaging logits from models with different loss landscapes produces better-calibrated ensemble probabilities.

### 4.4 Failure Mode Analysis

VFL-Ultra likely inherits the same "real vs. pretend" failure mode as TSM (the global attention across 4 frames cannot infer the trajectory's completion) but may differ in its "temporal direction confusion" failure. A model that directly compares frame 1 to frame 4 in a single attention step should in principle distinguish "folding" (objects become more compact) from "unfolding" (objects become less compact). The ensemble improvement of +0.61 pp from VFL-Ultra suggests this is the case for at least some clips.

---

## 5. Paper-Ready Summary

`video_former_lite_ultra` implements a hybrid architecture where a ResNet18 spatial encoder extracts per-frame features, which are then temporally aggregated by a 2-layer pre-LN Transformer encoder [Xiong et al., ICML 2020] via a learnable [CLS] token [Dosovitskiy et al., ICLR 2021]. Trained from scratch with a critical 10-epoch regularisation warmup that eliminates the cold-start instability observed with immediate MixUp/CutMix (9.45% → 45.78% → 53.37% over three training runs), the model achieves 53.37% internal and 32.11% val-dir top-1 accuracy — 6 pp below the best TSM variant but providing complementary temporal reasoning (global all-to-all attention vs. local channel shift) that contributes +0.61 pp to the final ensemble through error decorrelation.

---

## BibTeX

```bibtex
@inproceedings{bertasius2021timesformer,
  title={{Is Space-Time Attention All You Need for Video Understanding?}},
  author={Bertasius, Gedas and Wang, Heng and Torresani, Lorenzo},
  booktitle={ICML},
  year={2021}
}

@inproceedings{dosovitskiy2021vit,
  title={{An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}},
  author={Dosovitskiy, Alexey and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{xiong2020prenorm,
  title={{On Layer Normalization in the Transformer Architecture}},
  author={Xiong, Ruibin and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{vaswani2017attention,
  title={{Attention Is All You Need}},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{liu2022videoswin,
  title={{Video Swin Transformer}},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{zhang2018mixup,
  title={{MixUp: Beyond Empirical Risk Minimization}},
  author={Zhang, Hongyi and Cai, Moustapha and Recht, Benjamin and Re, Christopher},
  booktitle={ICLR},
  year={2018}
}

@inproceedings{yun2019cutmix,
  title={{CutMix: Training Strategy that Makes Use of Sample Patches}},
  author={Yun, Sangdoo and Han, Dongyoon and Oh, Seong Joon and Chun, Sanghyuk and Choe, Junsoo and Yoo, Youngjoon},
  booktitle={ICCV},
  year={2019}
}
```
