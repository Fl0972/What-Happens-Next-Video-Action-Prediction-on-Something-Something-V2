# Document 1 — Model Inventory, Comparison & Literature Anchoring

**Project:** Temporal Action Prediction on "What Happens Next?" (SSv2-33)  
**Track:** Closed World (no ImageNet pretraining)  
**Dataset:** 33-class subset of Something-Something V2 [CITE: Goyal et al., 2017]; 4 JPEG frames per clip.

---

## 1. Comparison Table

| Model Name | Architecture Family | Core Innovation | Key Hyperparameters | Inspired By | Val-Dir Top-1 | Notes |
|---|---|---|---|---|---|---|
| **CNNBaseline** | 2D CNN | ResNet18 + temporal average pool | T=4, B=32 | He et al., CVPR 2016 [1] | — | No temporal reasoning; serves as lower bound |
| **CNNLSTM** | 2D CNN + RNN | ResNet18 per-frame features → LSTM | T=4, B=32, hidden=256 | Donahue et al., CVPR 2015 [2] | — | Marginal gain over CNNBaseline |
| **VideoFormerLite** (VFL, closed) | 2D CNN + Transformer | ResNet18 backbone + 2-layer pre-LN Transformer + [CLS] token | T=4, d=512, n_heads=8, n_layers=2 | Dosovitskiy et al., ICLR 2021 [3]; Bertasius et al., ICML 2021 [4] | — | 45.78% internal val (60 ep); +36 pp from reg warmup |
| **VideoFormerLite-Ultra** (VFL-Ultra) | 2D CNN + Transformer | Same as VFL + 100 epochs + tuned lr | T=4, lr=5e-4, n_layers=2 | Dosovitskiy et al. [3]; Bertasius et al. [4] | 32.11% | 53.37% internal val (100 ep); ensemble diversity source |
| **R2Plus1D** | Factored 3D CNN | Spatial (2D) + temporal (1D) conv factorisation | T=8, B=8, 112×112 | Tran et al., CVPR 2018 [5] | — | Killed at epoch 15; poor under closed-track constraints |
| **TSMResNet-Ultra** | 2D CNN + temporal shift | Zero-param channel shift ±1 frame at every ResNet block | T=5, fold_div=4, 80 ep | Lin et al., ICCV 2019 [6] | 34.91% | 53.32% internal val; LB 34.46% |
| **TSMResNet-Ultra-50** | 2D CNN + temporal shift | Scale-up: ResNet50 backbone + 8 frames | T=8, fold_div=8, ep=55 (early stop) | Lin et al. [6]; He et al. [1] | — | 45.26% internal val; capacity wasted without fixing frame count |
| **TSMResNet-Ultra-v2** | 2D CNN + temporal shift | Correct frame count (T=4) + focal loss + 150 epochs | T=4, fold_div=4, focal γ=2, 150 ep | Lin et al. [6]; Lin et al., ICCV 2017 [7] | **38.25%** | **57.73% internal val; best single model** |
| **TSMBiGRU** | 2D CNN + temporal shift + RNN | TSM-ResNet50 features → BiGRU for ordered temporal reasoning | T=4, gru_hidden=256, fold_div=8 | Lin et al. [6]; Schuster & Paliwal 1997 [8] | — | ~7.6% internal val (rotating fold); poor from-scratch convergence |
| **MaxViTVideo** | Hierarchical ViT + mean pool | MaxViT-T per-frame (from scratch) + frame interpolation (4→8) | T=8 (interp.) | Tu et al., ECCV 2022 [11] | — | Frame interpolation as temporal augmentation |
| **Final Ensemble** | Late fusion | Log-softmax avg of TSM-v2 + TSM-v2(rot) + VFL-Ultra(rot) + 10-crop TTA | — | [6][3][4] | **44.03%** | best closed-track result |

**Notes on val-dir accuracy:** All val-dir scores are top-1 on the full official validation set (6,745 clips, 33 classes), measured without TTA unless stated. Internal val accuracy is measured on a held-out 20% split of the training data and is consistently ~15–19 pp higher than val-dir due to distribution shift.

---

## 2. Per-Model Entries

### 2.1 CNNBaseline

**Architecture.** A standard ResNet18 [1] backbone is applied independently to each of the T=4 frames; the resulting per-frame feature vectors (512-d) are average-pooled over time to produce a single clip-level representation, which feeds a linear classifier. This model contains no temporal reasoning beyond the implicit spatial correlations between independently processed frames.

**Design choices.** The architecture serves as a sanity-check lower bound: it establishes that the task requires more than per-frame recognition. Average pooling over time is equivalent to treating the video as a bag of frames, discarding all ordering information — a known weakness for motion-defined labels such as those in SSv2.

**Related work.** ResNet18 [1] is the standard lightweight backbone for this parameter regime. Similar bag-of-frames approaches are used in TSN [12] as a degenerate case ("no temporal module"). On SSv2, these perform significantly below motion-aware models [6][12].

**How this differs from the cited baseline.** Unlike TSN [12], which uses sparse temporal segment sampling, this baseline uses uniform linspace frame sampling and does not apply any temporal segment jitter.

**Position in progression.** Step 1 in the experimental series; establishes lower bound.

---

### 2.2 CNNLSTM

**Architecture.** Per-frame ResNet18 features (512-d, shared weights) are fed as a sequence into a single-layer unidirectional LSTM (hidden=256). The final hidden state is passed to a linear classifier. This is a direct extension of the CNNBaseline with sequential temporal integration replacing temporal average pooling.

**Design choices.** LSTMs provide ordered sequential memory through gated recurrence, making them theoretically capable of capturing temporal ordering (e.g., distinguishing folding from unfolding). However, with only T=4 frames, backpropagation through time is trivial and the gating mechanism adds parameters without substantial capacity benefit. The LSTM is particularly sensitive to vanishing gradients when trained from scratch without pretraining on the backbone.

**Related work.** The CNN+LSTM paradigm for video classification was popularised by Donahue et al. [2] (LRCN) and Yue-Heng et al. Empirically, LSTMs on short clips (T≤8) rarely outperform attention-based alternatives [4] because the sequence is too short for the gating mechanism to provide meaningful benefit over simpler mean pooling.

**How this differs from the cited baseline.** LRCN [2] used deeper LSTM stacks (2 layers) and longer sequences. Here the LSTM operates on only 4 frames, limiting its sequential capacity.

**Position in progression.** Step 2; establishes that sequential RNN integration provides marginal improvement over temporal average pooling on 4-frame clips.

---

### 2.3 VideoFormerLite (VFL)

**Architecture.** Per-frame ResNet18 (512-d features) feeds a sequence of T frame tokens. A learnable [CLS] token is prepended, learnable temporal positional embeddings are added, and the sequence passes through L pre-Layer-Normalisation Transformer encoder blocks [3] (MHSA + MLP, d=512, n_heads=8, d_ff=2048). The [CLS] output is normalised, dropped, and projected to class logits. The standard variant uses L=2 blocks (17.5M total params); the "ultra" variant retains L=2 but uses 100 epochs and a tuned learning rate.

**Design choices.** Global temporal self-attention is the key inductive bias: every frame token can attend directly to every other frame, regardless of temporal distance. With T=4, this means any pair (frame 1, frame 4) is only one attention step apart, unlike TSM where distant frames communicate only through multiple shift layers. The [CLS] aggregation scheme follows ViT [3]. Pre-LN placement [13] improves training stability for from-scratch models by avoiding gradient explosion in the first epochs.

**Related work.** TimeSformer [4] applies divided space-time attention within a pure ViT for video; VFL hybridises this by keeping 2D CNN spatial feature extraction (cheaper and more data-efficient) and replacing the temporal aggregator with a small Transformer. This hybrid "CNN + self-attention" pattern was discussed in [3] (hybrid ViT) and is more sample-efficient than a pure ViT from scratch. The pre-LN Transformer follows Xiong et al. [13].

**How this differs from the cited baseline.** TimeSformer [4] uses a full ViT backbone; VFL uses ResNet18 as the spatial encoder to reduce parameter count and improve convergence speed from scratch. VFL also operates on the [CLS] token (ViT-style) rather than on a CLS-pooled attention over temporal tokens (TimeSformer-style).

**Key experimental finding.** Enabling MixUp/CutMix from epoch 1 resulted in 9.45% val accuracy (killed at epoch 4). Deferring these augmentations by 10 epochs (regularisation warm-up) produced 45.78% at 60 epochs — a +36.3 pp improvement. This confirms the cold-start sensitivity of Transformer-based models to label-mixing augmentations, consistent with the observed training instability in ViT fine-tuning when regularisation is too aggressive early [3].

**Position in progression.** Steps 3 and 6; demonstrates that Transformer temporal aggregation is competitive with TSM and provides ensemble diversity (different inductive bias: global attention vs. local channel shift).

---

### 2.4 R2Plus1D

**Architecture.** A torchvision `r2plus1d_18` backbone [5] (no pretrained weights) factorises each 3D convolution into a 2D spatial conv (1×k×k) followed by a 1D temporal conv (k×1×1) with a non-linearity in between. The input is permuted from (B,T,C,H,W) to (B,C,T,H,W) as required by torchvision. Global average pooling followed by dropout and a linear classifier produce class logits. Total parameters: ~31M (3× ResNet18).

**Design choices.** The factored design of R(2+1)D [5] recovers most of the expressivity of full 3D convolutions at lower FLOPs. However, from scratch on a 33-class 45k-clip dataset with only 4–8 frames, several constraints made this model poorly suited: (1) 3D conv weights require substantially more training data or epochs to converge from random initialisation; (2) at T≤8, the 1D temporal kernel has too few positions to model meaningful temporal patterns; (3) the larger parameter count requires smaller batch sizes and image resolution, degrading BatchNorm statistics.

**Related work.** R(2+1)D [5] was validated on Kinetics-400 (306k clips) with pre-training; SlowFast [14] also relies on Kinetics pretraining for competitive SSv2 scores. Both architectures are primarily open-track solutions.

**How this differs from the cited baseline.** R(2+1)D [5] was originally validated with Kinetics pretraining; this implementation trains from scratch. The temporal conv filter has T_max=8 positions vs. the 16-frame inputs used in [5].

**Position in progression.** Parallel experiment to VFL; demonstrated that 3D-factored convolutions from scratch are significantly weaker than shift-based or attention-based approaches under closed-track constraints.

---

### 2.5 TSMResNet-Ultra

**Architecture.** A ResNet18 backbone [1] where every residual block is wrapped with `TemporalShift` [6]: 1/fold_div of channels are shifted one position backward in time (allowing frame t to read from frame t-1), and 1/fold_div shifted forward — with `fold_div=4`, 50% of channels participate in temporal communication per block. No parameters are added. After the backbone, per-frame features (512-d) are averaged over time and projected to 33 classes. Stochastic depth [15] (max rate 0.1, linear schedule) and head dropout (0.3) provide regularisation. Total parameters: ~11.2M.

**Design choices.** The key property of TSM is parameter-free temporal mixing: all parameters learn spatial features (maximally useful from scratch) while temporal integration is provided "for free" by the channel shift. The `fold_div=4` setting (vs. the default 8 in Lin et al. [6]) is more aggressive, necessary to propagate temporal context across the shallow 5-frame sequence; at 16 frames, `fold_div=8` is sufficient because the shift percolates through more frames per forward pass. Training with T=5 uses one interpolated/duplicate frame, which was later identified as corrupting the shift signal.

**Related work.** TSM [6] was specifically validated on SSv2 (achieving 59.1% top-1 with ResNet50 pretrained on Kinetics). The zero-parameter design is shared with TEA [CITE: Li et al., CVPR 2020], which adds element-wise temporal excitation. The from-scratch adaptation follows the spirit of MobileNet-TSM [6] but without ImageNet pretraining.

**How this differs from the cited baseline.** Lin et al. [6] report SSv2 results with Kinetics-pretrained ResNet50 and T=8/16. This experiment trains ResNet18 from scratch with T=5 and `fold_div=4`, the latter compensating for the shorter sequence. The full regularisation recipe (reg warmup, weighted sampling, focal-free CE) represents a non-trivial closed-track adaptation.

**Position in progression.** Step 5; first model to demonstrate 50%+ internal validation accuracy (53.32%).

---

### 2.6 TSMResNet-Ultra-50

**Architecture.** Same as TSMResNet-Ultra but with a ResNet50 backbone (2048-d features, 23.6M params) and T=8 frames, using the default `fold_div=8`. The wider bottleneck blocks provide 4× the feature capacity of ResNet18. Stochastic depth (max rate 0.2), head dropout (0.5), and SGDR scheduling [16] (T₀=28, 4 cycles) were applied.

**Design choices.** The TSM-Ultra training curve showed a plateau after epoch 68, suggesting spatial capacity (ResNet18, 11M params) as the bottleneck. Scaling to ResNet50 was the natural response. However, T=8 was chosen based on SSv2 literature conventions [6], not based on the actual clip length (4 frames). This introduced 4 duplicated/interpolated frames, corrupting the shift signal — the same issue that had been identified for TSM-Ultra but re-introduced by the scale-up.

**Related work.** The original TSM paper [6] benchmarked SSv2 at T=8 and T=16 with ResNet50. SGDR [16] provides cyclic LR schedules that escape sharp minima.

**Key finding.** Despite 4× more parameters, TSM-Ultra-50 reached only 45.26% internal val at epoch 55 (early stopped) — 8 pp below TSM-Ultra-v2 (ResNet18, T=4). This confirms that **scaling capacity without fixing the data preparation bug is wasted compute** and that the frame-count mismatch is the primary bottleneck.

**Position in progression.** Step after TSM-Ultra; demonstrates the priority of correct data handling over architectural scale.

---

### 2.7 TSMResNet-Ultra-v2 — *Best single model*

**Architecture.** TSMResNet-Ultra with three targeted changes: (1) `num_frames=4` matching the actual clip length; (2) focal loss [7] (γ=2) replacing standard cross-entropy; (3) extended training to 150 epochs. Architecture remains ResNet18 (11.2M params), `fold_div=4`, stochastic depth 0.1, head dropout 0.3.

**Design choices.** The frame-count fix is the primary driver: with T=4, every frame is a unique temporal sample and the shift operates on independent information. Focal loss [7] (originally proposed for one-stage object detection) reweights each example by (1−p_t)^γ, concentrating gradient on hard examples. In SSv2, the hardest confusions are within-group pairs (e.g. "poking so it falls" vs. "poking so it slightly moves") where the correct class probability p_t is low but the wrong prediction is confidently wrong. Focal loss with γ=2 directly addresses this by amplifying gradient on these pairs.

**Related work.** Focal loss [7] was introduced for class imbalance in detection; its application to multi-class action recognition with near-duplicate classes was motivated by the SSv2 class structure [CITE: Goyal et al., 2017]. The AdamW + cosine annealing + warmup schedule follows the standard ViT fine-tuning recipe [3] adapted for from-scratch training.

**Key finding.** +4.4 pp over TSM-Ultra (57.73% vs. 53.32% internal val) purely from fixing num_frames. Val-dir: 38.25% vs. 34.91%, a +3.34 pp improvement. The focal loss contributes an additional gain on top of the frame-count fix that is not individually ablated but is consistent with the expected improvement on hard pairs.

**Position in progression.** Step 7; the culmination of the TSM experimental line.

---

### 2.8 TSMBiGRU

**Architecture.** TSM-ResNet50 backbone (2048-d per-frame features, `fold_div=8`) followed by a Bidirectional GRU (BiGRU) [8] with 256 hidden units per direction. Per-frame features are dropout-regularised before the GRU; GRU outputs are mean-pooled over time and passed through a head dropout layer before the classifier. Total temporal head parameters: ~2× the number in plain mean pooling. Trained with SGDR (T₀=23, 4 cycles), backbone LLRD (backbone LR × 0.1), and a rotating 5-fold scheme.

**Design choices.** The BiGRU is motivated by SSv2's hardest confusions: "folding" vs. "unfolding" and "real action" vs. "pretending", which differ primarily in the temporal direction of the motion trajectory. A bidirectional sequential model can in principle distinguish left-to-right from right-to-left trajectories within a 4-frame sequence. LSTM was rejected in favour of GRU because the cell state gate adds overfit risk with only 4 frames and no pretraining.

**Related work.** BiGRU temporal heads are used in hybrid video models [CITE: Zhu et al., CVPR 2018, VideoLSTM]; Schuster & Paliwal [8] introduced bidirectional RNNs. LLRD for large pretrained backbones follows the ViT fine-tuning recipe from Dosovitskiy et al. [3].

**Key finding.** Val accuracy of ~7.6% under the rotating-fold training scheme — far below expectations. Root causes: (1) from-scratch ResNet50 combined with the BiGRU head requires significantly more epochs to converge than the single-pass TSM; (2) the rotating-fold scheme further divides the training set, reducing the effective per-epoch sample count; (3) SGDR restarts amplify variance in early training. The model did not converge within 100 epochs.

**Position in progression.** Exploratory experiment; confirms that from-scratch training of large backbones (ResNet50) paired with sequential temporal heads requires fundamentally different training schedules.

---

### 2.9 MaxViTVideo

**Architecture.** MaxViT-T [11] (Multi-Axis ViT, ~31M params; hierarchical window+grid attention) processes each frame independently, producing a 512-dim feature per frame (after a 4-stage hierarchical encoder, AdaptiveAvgPool, LayerNorm, Linear, Tanh). Temporal aggregation is by mean pooling over the T frame features. The model is optionally combined with frame interpolation (dataset.interpolate_frames=true): 4 real JPEG frames are expanded to 8 by inserting linear pixel-blend midpoints, doubling the number of temporal positions at zero additional disk I/O.

**Design choices.** MaxViT [11] improves on standard ViT by combining window attention (local) and grid attention (global dilated) in each stage, providing multi-scale spatial reasoning at lower computational cost than full self-attention. Frame interpolation was proposed as a temporal augmentation strategy for fixed-length 4-frame clips: inserting synthetic midpoints doubles the temporal density of the mean-pool aggregator without requiring real additional frames.

**Related work.** MaxViT [11] outperforms both ViT-B and Swin Transformer [CITE: Liu et al., ICCV 2021] at similar parameter counts on image benchmarks. Frame interpolation for video is conceptually related to temporal super-resolution for action recognition [CITE: Li et al., 2023].

**Position in progression.** Designed but not trained within this experiment series; intended as an ensemble complement to TSM-Ultra-v2 (orthogonal inductive biases: multi-scale window/grid attention vs. local channel shift).

---

### 2.10 Final Ensemble

**Architecture.** The best closed-track submission is a **log-softmax** late-fusion of `tsm_ultra_v2`, its rotating-fold variant, and `video_former_lite_ultra` (rotating fold), each with 10-crop TTA — **44.03%** val-dir (see `ENSEMBLE_EXP.md`). The component study below uses an earlier **weighted-logit** three-model sub-ensemble: `tsm_ultra_v2` (weight 0.50), `tsm_ultra` (weight 0.25), `video_former_lite_ultra` (weight 0.25). Each model runs its own independent forward pass; pre-softmax logits are combined as a weighted average; argmax gives the final prediction. Test-time augmentation (TTA) applies 10 views per clip (5 spatial crops × 2 flips) before combining, with logits averaged across TTA views per model.

**Design choices.** Weights are proportional to validation accuracy: `tsm_ultra_v2` (57.73% internal val) dominates at 0.50; the other two contribute equally at 0.25. The key selection criterion is error decorrelation: `tsm_ultra_v2` and `tsm_ultra` differ in frame count (4 vs. 5) and loss function (focal vs. CE), while VFL introduces a radically different temporal inductive bias (global attention vs. local shift). These differences ensure that errors made by one model are not systematically shared by others.

**Related work.** Late-fusion ensembling for action recognition is standard practice [CITE: Simonyan & Zisserman, NIPS 2014; Feichtenhofer et al., CVPR 2016]. TTA for video classification is used in TSM [6] (multi-crop inference) and TimeSformer [4]. The weighted logit averaging follows [CITE: Guzman-Rivera et al., 2012].

**Key result.** The best closed-track ensemble reaches **44.03% val-dir top-1 / 75.37% top-5** (log-softmax of `tsm_v2` + `tsm_v2_rot` + `vfl_rot`, +TTA; see `ENSEMBLE_EXP.md`). On the three-model weighted sub-ensemble (40.43%), leave-one-out ablation confirms every component contributes positively: removing `tsm_ultra_v2` costs −3.17 pp; removing `tsm_ultra` costs −0.87 pp; removing VFL costs −0.61 pp.

---

## 3. Model Family Summary

```
Experiment progression (internal val accuracy):

[CNNBaseline] → [CNNLSTM] → [VFL (no warmup)] → [VFL (warmup, 60ep)] → [TSM-Ultra]  →  [VFL-Ultra]  →  [TSM-Ultra-v2]
  low           ~marginal       9.45%               45.78%               53.32%           53.37%          57.73%
                                                     ↑                      ↑                ↑
                               +36 pp from reg warmup            best TSM variant   best from-scratch model

Best ensemble: 44.03% val-dir (log-softmax of tsm_v2 + tsm_v2_rot + vfl_rot, +TTA)
```

---

## 4. BibTeX References

```bibtex
@inproceedings{he2016resnet,
  title={{Deep Residual Learning for Image Recognition}},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  year={2016}
}

@inproceedings{donahue2015lrcn,
  title={{Long-term Recurrent Convolutional Networks for Visual Recognition and Description}},
  author={Donahue, Jeffrey and others},
  booktitle={CVPR},
  year={2015}
}

@inproceedings{dosovitskiy2021vit,
  title={{An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}},
  author={Dosovitskiy, Alexey and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{bertasius2021timesformer,
  title={{Is Space-Time Attention All You Need for Video Understanding?}},
  author={Bertasius, Gedas and Wang, Heng and Torresani, Lorenzo},
  booktitle={ICML},
  year={2021}
}

@inproceedings{tran2018r2plus1d,
  title={{A Closer Look at Spatiotemporal Convolutions for Action Recognition}},
  author={Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and Paluri, Manohar},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{lin2019tsm,
  title={{TSM: Temporal Shift Module for Efficient Video Understanding}},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={ICCV},
  year={2019}
}

@inproceedings{lin2017focal,
  title={{Focal Loss for Dense Object Detection}},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={ICCV},
  year={2017}
}

@article{schuster1997bilstm,
  title={{Bidirectional Recurrent Neural Networks}},
  author={Schuster, Mike and Paliwal, Kuldip K.},
  journal={IEEE Transactions on Signal Processing},
  year={1997}
}

@inproceedings{bahdanau2015attention,
  title={{Neural Machine Translation by Jointly Learning to Align and Translate}},
  author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
  booktitle={ICLR},
  year={2015}
}

@inproceedings{jaegle2021perceiver,
  title={{Perceiver: General Perception with Iterative Attention}},
  author={Jaegle, Andrew and others},
  booktitle={ICML},
  year={2021}
}

@inproceedings{tu2022maxvit,
  title={{MaxViT: Multi-Axis Vision Transformer}},
  author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and Yang, Feng and Milanfar, Peyman and Bovik, Alan and Li, Yinxiao},
  booktitle={ECCV},
  year={2022}
}

@inproceedings{wang2016tsn,
  title={{Temporal Segment Networks: Towards Good Practices for Deep Action Recognition}},
  author={Wang, Limin and Xiong, Yuanjun and Wang, Zhe and Qiao, Yu and Lin, Dahua and Tang, Xiaoou and Van Gool, Luc},
  booktitle={ECCV},
  year={2016}
}

@inproceedings{xiong2020prenorm,
  title={{On Layer Normalization in the Transformer Architecture}},
  author={Xiong, Ruibin and Yang, Yunchang and He, Di and Zheng, Kai and Zheng, Shuxin and Xing, Chen and Zhang, Huishuai and Lan, Yanyan and Wang, Liwei and Liu, Tie-Yan},
  booktitle={ICML},
  year={2020}
}

@inproceedings{feichtenhofer2019slowfast,
  title={{SlowFast Networks for Video Recognition}},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={ICCV},
  year={2019}
}

@inproceedings{huang2016stochdepth,
  title={{Deep Networks with Stochastic Depth}},
  author={Huang, Gao and Sun, Yu and Liu, Zhuang and Sedra, Daniel and Weinberger, Kilian Q.},
  booktitle={ECCV},
  year={2016}
}

@inproceedings{loshchilov2017sgdr,
  title={{SGDR: Stochastic Gradient Descent with Warm Restarts}},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2017}
}

@inproceedings{goyal2017ssv2,
  title={{The "Something Something" Video Database for Learning and Evaluating Visual Common Sense}},
  author={Goyal, Raghav and Ebrahimi Kahou, Samira and Michalski, Vincent and others},
  booktitle={ICCV},
  year={2017}
}

@inproceedings{loshchilov2019adamw,
  title={{Decoupled Weight Decay Regularization}},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
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

@inproceedings{cubuk2020randaugment,
  title={{RandAugment: Practical Automated Data Augmentation with a Reduced Search Space}},
  author={Cubuk, Ekin D. and Zoph, Barret and Shlens, Jonathon and Le, Quoc V.},
  booktitle={CVPR Workshops},
  year={2020}
}
```

---

[1] He et al., CVPR 2016 | [2] Donahue et al., CVPR 2015 | [3] Dosovitskiy et al., ICLR 2021 | [4] Bertasius et al., ICML 2021 | [5] Tran et al., CVPR 2018 | [6] Lin et al., ICCV 2019 | [7] Lin et al., ICCV 2017 | [8] Schuster & Paliwal 1997 | [9] Bahdanau et al., ICLR 2015 | [10] Jaegle et al., ICML 2021 | [11] Tu et al., ECCV 2022 | [12] Wang et al., ECCV 2016 | [13] Xiong et al., ICML 2020 | [14] Feichtenhofer et al., ICCV 2019 | [15] Huang et al., ECCV 2016 | [16] Loshchilov & Hutter, ICLR 2017
