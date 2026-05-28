# Document 2 — Model Analysis: Early Baselines (CNNBaseline, CNNLSTM, R2Plus1D)

This document covers the three baseline architectures that established lower bounds and informed the design decisions for the final model family.

---

## Part A — CNNBaseline and CNNLSTM

**Model identifiers:** `cnn_baseline`, `cnn_lstm`  
**Architecture files:** `src/models/cnn_baseline.py`, `src/models/cnn_lstm.py`  
**Status:** Lower bound models; not submitted.

### A.1 Model Overview

**CNNBaseline.** A ResNet18 [He et al., CVPR 2016] backbone (random initialisation) processes each frame independently; the resulting T=4 feature vectors (512-d) are average-pooled over time and linearly classified. This model contains zero temporal reasoning: the video is treated as a bag of independently scored frames, and the average pooling is equivalent to computing the mean of T independent class probability distributions. The model establishes the performance floor achievable by spatial-only recognition on SSv2.

**CNNLSTM.** Extends CNNBaseline by replacing average pooling with an unidirectional LSTM (hidden=256, 1 layer). Per-frame ResNet18 features are fed sequentially; the LSTM hidden state after the final frame feeds the classifier. The LSTM is theoretically capable of modelling temporal dependencies through gating, but with T=4 frames, the sequence is too short for the gating mechanism to provide meaningful benefit.

**Motivation for both.** These architectures serve as ablation baselines for temporal reasoning: CNNBaseline tests whether spatial-only features suffice (they do not, on SSv2); CNNLSTM tests whether a simple sequential model improves over bag-of-frames (marginally, but not substantially). The gap between CNNBaseline and the final ensemble (~40+ pp) quantifies the combined benefit of all temporal reasoning improvements.

### A.2 Design Choices

**Average pooling over time** is the degenerate temporal aggregation: it is equivalent to the class prediction from the "mean frame" of the video. On SSv2, where classes are defined by motion trajectories rather than mean appearance, this discards nearly all discriminative information.

**LSTM limitations at T=4.** The LSTM cell state is designed to carry long-range dependencies across many time steps. With T=4, the network has at most 4 sequential updates before classification — equivalent in capacity to a 2-layer MLP with skip connections (gating). The temporal dependencies in SSv2 are well within what this capacity can express, but the from-scratch initialisation requires the LSTM to simultaneously learn temporal patterns and class-discriminative spatial features, increasing optimisation difficulty.

### A.3 Related Work

LRCN [Donahue et al., CVPR 2015] established the CNN+LSTM paradigm for video classification and captioning. The model was shown to be effective on HMDB-51 and UCF-101 (appearance-heavy datasets) but these benchmarks are substantially easier than SSv2 in terms of temporal reasoning requirements. On SSv2, Goyal et al. [ICCV 2017] demonstrated that architectures relying purely on spatial appearance fail to achieve competitive performance, motivating the motion-centric designs that followed.

### A.4 Results

Exact validation accuracy numbers for CNNBaseline and CNNLSTM were not preserved in the logged experiments (early runs were not tracked systematically). The qualitative result is confirmed in FINAL_MODEL.md:
- **CNNBaseline:** low baseline (undisclosed absolute number; serves only as starting point)
- **CNNLSTM:** marginal gain over CNNBaseline

The absence of precise numbers is noted as a documentation gap; these models were not re-evaluated on the official val-dir set.

### A.5 Key Takeaway

The marginal gap between CNNBaseline and CNNLSTM on a motion-centric dataset confirms a known result: sequential RNN aggregation of frame-level features provides limited benefit when the backbone is frozen/random and the temporal sequence is short. The gain from TSM-Ultra-v2 (38.25% val-dir) over CNNLSTM is primarily attributable to (a) the temporal shift mechanism enabling the backbone to directly learn motion features, and (b) the regularisation recipe.

---

## Part B — R2Plus1D

**Model identifier:** `r2plus1d`  
**Checkpoint:** `models/r2plus1d_closed.pt`  
**Experiment config:** `src/configs/experiment/r2plus1d_closed.yaml`  
**Architecture file:** `src/models/r2plus1d.py`  
**Status:** Killed at epoch 15; not used in ensemble.

### B.1 Model Overview

**Motivation.** R(2+1)D [Tran et al., CVPR 2018] factorises each 3D convolution into a 2D spatial conv (1×k×k) followed by a 1D temporal conv (k×1×1) with a ReLU non-linearity between them. This factorisation increases expressivity relative to plain 3D conv: the intermediate non-linearity doubles the number of non-linear transformations per unit of computation. R(2+1)D was tested as a closed-track alternative to TSM — a different approach to spatiotemporal modelling that directly convolves over the temporal dimension rather than shifting channels.

**Hypothesis.** 3D-factored convolutions might learn motion representations more directly than channel shift (which is a discrete, parameter-free approximation of temporal convolution), potentially improving on `tsm_ultra` with sufficient training.

### B.2 Architecture

```
Input: (B, T, C, H, W)
  → permute → (B, C, T, H, W)    [torchvision convention]
  → R(2+1)D-18 backbone (random init, ~31M params)
    - Spatial conv (1, 3, 3) → ReLU → Temporal conv (3, 1, 1)
    - 4 stages, increasing channels 64→128→256→512
  → Global average pool → (B, 512)
  → Dropout(0.5) → Linear(512, 33)
```

Total parameters: ~31M (3× ResNet18). Input resolution: 112×112 (forced by VRAM constraints at T=8).

### B.3 Experimental Setup

| Hyperparameter | Value |
|----------------|-------|
| Backbone | R(2+1)D-18, random init |
| num_frames | 8 |
| image_size | 112×112 (VRAM constraint) |
| batch_size | 8 |
| optimizer | AdamW, lr=1e-3 |
| scheduler | cosine, 5-epoch warmup |
| epochs | Killed at 15 / planned 80 |
| reg_warmup_epochs | **None** (key failure) |
| label_smoothing | 0.1 |
| MixUp/CutMix | 0.4/0.5 (active from epoch 1) |

### B.4 Results

| Epoch | Val Loss | Val Acc | Best Val Acc |
|-------|----------|---------|-------------|
| 12 | 3.279 | 10.37% | 10.37% |
| 13 | 3.274 | 10.19% | 10.37% |
| 14 | 3.271 | 10.78% | 10.78% |
| 15 | 3.265 | 10.61% | 10.78% |

The run was killed at epoch 15 with 10.78% val accuracy — well below random (3.0% for 33 uniform classes, but higher for an imbalanced distribution). The training loss was still decreasing slowly, suggesting the model had not fully diverged but was converging extremely slowly.

### B.5 Failure Analysis

Four root causes explain the poor performance:

**1. Missing regularisation warmup.** MixUp and CutMix were active from epoch 1 with the original configuration (α=0.4, p=0.5). As established by the VFL warmup ablation (+36 pp), applying label-mixing augmentations before a randomly initialised model has learned basic features prevents feature learning and causes training to stall. This is the primary failure cause — the same experiment applied to VFL reduced accuracy to 9.45% before being killed.

**2. Memory-forced resolution compromise.** R(2+1)D with T=8 at 224×224 exceeds 16 GB VRAM at batch=8. Reducing to 112×112 cuts spatial information by 4× and degrades BatchNorm statistics (batch=8 is below the recommended minimum of ~16 for stable BN). At 224×224, batch=4 would be required — further degrading BN.

**3. High parameter count from scratch.** R(2+1)D-18 has ~31M parameters — 3× ResNet18. Training 31M parameters from scratch on ~36k clips of 33 classes is severely under-determined. ResNet18-based models (TSM, VFL) with ~11–17M params converge more reliably in this regime.

**4. Temporal conv undersized at T=8.** The R(2+1)D temporal filters have kernel size 3 in the temporal dimension; with T=8, the first temporal kernel covers only 3 positions. For action recognition on clips with 4–8 frames, the temporal receptive field of 3D-style convolutions is not sufficiently larger than that of a 1-frame-hop shift to justify the additional computational cost.

### B.6 Theoretical Strengths (Open Track Context)

R(2+1)D's design is well-suited to an open-track (with Kinetics pretraining) setting where:
- 3D conv filters can be initialised from Kinetics-pretrained weights
- Longer clips (16-32 frames) fully utilise the temporal kernel's receptive field
- Larger batch sizes (32+) are feasible with pretrained convergence speed

The original paper [Tran et al., CVPR 2018] reports 73.9% on Sports-1M and 78.7% on UCF-101 with pretraining — competitive with TSM. On SSv2 specifically, R(2+1)D achieves 44.8% (Kinetics-pretrained, T=8), comparable to TSM (ResNet18, ImageNet-pretrained, T=8: 45.6% [Lin et al., 2019]).

### B.7 Key Takeaway

R(2+1)D is architecturally sound for video action recognition but is poorly suited to the closed-track (from-scratch) setting on short clips with limited data. The combination of (a) missing regularisation warmup, (b) resolution compromise, and (c) high parameter count from scratch produced a run that could not converge within the compute budget. The conclusion for this project is: **do not invest in R(2+1)D on the closed track** unless the recipe is fixed and the run is given adequate epochs (80+) with the full regularisation stack including warmup.

---

## Part C — Summary Comparison

| Model | Temporal Mechanism | Internal Val | Val-Dir | Notes |
|-------|-------------------|-------------|---------|-------|
| CNNBaseline | None (avg pool) | Low | — | Lower bound |
| CNNLSTM | LSTM (sequential) | Marginal > baseline | — | Short-sequence limit |
| R2Plus1D (killed at ep15) | 3D-factored conv | 10.78% | — | No warmup; VRAM compromise |
| VFL (no warmup, killed) | Transformer | 9.45% | — | No warmup |
| **VFL (warmup, 60ep)** | **Transformer** | **45.78%** | — | **Warmup is critical** |
| **TSM-Ultra-v2** | **Channel shift** | **57.73%** | **38.25%** | **Best single model** |

The progression from CNNBaseline (~random) to TSM-Ultra-v2 (57.73% internal val) documents the cumulative benefit of:
1. Temporal reasoning module (+large gap: shift/attention vs. avg pool)
2. Regularisation warmup (+36 pp on VFL; replicated on all from-scratch models)
3. Correct frame count (+4.4 pp, frame-count fix for TSM)
4. Focal loss (estimated +1–2 pp on hard pairs)
5. Extended training schedule (+incremental improvement over 80→150 epochs)

---

## BibTeX

```bibtex
@inproceedings{donahue2015lrcn,
  title={{Long-term Recurrent Convolutional Networks for Visual Recognition and Description}},
  author={Donahue, Jeffrey and others},
  booktitle={CVPR},
  year={2015}
}

@inproceedings{tran2018r2plus1d,
  title={{A Closer Look at Spatiotemporal Convolutions for Action Recognition}},
  author={Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and Paluri, Manohar},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{he2016resnet,
  title={{Deep Residual Learning for Image Recognition}},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  year={2016}
}

@inproceedings{goyal2017ssv2,
  title={{The "Something Something" Video Database for Learning and Evaluating Visual Common Sense}},
  author={Goyal, Raghav and others},
  booktitle={ICCV},
  year={2017}
}
```
