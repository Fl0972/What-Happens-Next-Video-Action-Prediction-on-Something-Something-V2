# CoAtNet vs EfficientFormer for Video Classification (Closed Track)

> **Closed-track constraint:** `pretrained=false` on all models.
> timm is used for the architecture only — no external weights are loaded.

## Papers

### CoAtNet
**"CoAtNet: Marrying Convolution and Attention for All Data Sizes"**
Zihang Dai, Hanxiao Liu, Quoc V. Le, Mingxing Tan — Google Brain.
NeurIPS 2021. arXiv:2106.04803 — https://arxiv.org/abs/2106.04803

CoAtNet unifies depthwise convolution and relative self-attention through a shared
pre-normalization formulation. The key insight is a **C-C-T-T stage layout**: two
MBConv (inverted bottleneck) stages process local features, then two Transformer
stages with relative position encodings handle global context. This ordering gives
the network the CNN's inductive bias for small datasets while retaining the
Transformer's expressiveness for global reasoning.

| Variant | Params | IN-1K top-1 (pretrained ref.) |
|---|---|---|
| CoAtNet-0 | 25 M | 81.6% |
| **CoAtNet-RMLP-1 (used here)** | **42 M** | **~83–84%** |
| CoAtNet-2 | 75 M | 87.1% |

timm model string (architecture only): `coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k`

---

### EfficientFormer
**"EfficientFormer: Vision Transformers at MobileNet Speed"**
Yanyu Li, Geng Yuan, Yang Wen, Evan Shelhamer, Jonathan Huang, Sergey Tulyakov — Snap Research.
NeurIPS 2022. arXiv:2206.01191 — https://arxiv.org/abs/2206.01191

EfficientFormer proves that a pure transformer can match MobileNet latency.
The design uses **dimension-consistent MetaFormer blocks**: pooling-based mixing in
the first three stages (hardware-friendly, no self-attention cost), then full MHSA
only in the final stage. The result is a very efficient architecture that trains
well from scratch due to its structured, progressive design.

| Variant | Params | IN-1K top-1 (pretrained ref.) |
|---|---|---|
| EfficientFormer-L1 | 12 M | 79.2% |
| **EfficientFormer-L3 (used here)** | **31 M** | **82.4%** |
| EfficientFormer-L7 | 82 M | 83.3% |

timm model string (architecture only): `efficientformer_l3.snap_dist_in1k`

---

## Architecture comparison

| Aspect | CoAtNet-RMLP-1 | EfficientFormer-L3 |
|---|---|---|
| Block types | MBConv + Transformer | Pooling Mixer + MHSA |
| Attention stages | Last 2 of 5 | Last 1 of 4 |
| Params | ~42 M | ~31 M |
| timm `num_features` | 1152 | 512 |
| Image size | 224 × 224 | 224 × 224 |
| Inductive bias | CNN → Attn (gradual) | Pool → Attn (abrupt) |

---

## Implementation in this repo

Both models use a timm backbone for per-frame feature extraction, then temporal
mean-pooling and a linear classifier:

```
(B, T, C, H, W) → backbone(B*T frames, random init) → (B*T, feat_dim)
               → reshape (B, T, feat_dim) → mean over T
               → Dropout → Linear → (B, 33)
```

`self.backbone` is exposed on both models so `backbone_lr_scale` works in
`build_optimizer()`, though it is set to `1.0` (uniform LR) for scratch training.

### Files

| File | Purpose |
|---|---|
| `src/models/coatnet.py` | `CoAtNetVideo` nn.Module |
| `src/models/efficientformer.py` | `EfficientFormerVideo` nn.Module |
| `src/configs/model/coatnet.yaml` | Model config (`pretrained: false`) |
| `src/configs/model/efficientformer.yaml` | Model config (`pretrained: false`) |
| `src/configs/experiment/efficientformer_scratch.yaml` | Scratch training recipe |
| `src/configs/experiment/coatnet_scratch.yaml` | Scratch training recipe |

`timm>=0.9.0` added to `pyproject.toml`. Run `uv sync` before training.

---

## From-scratch training recipe

Training transformers from scratch requires stronger regularisation and more epochs
than fine-tuning:

| Hyperparameter | From-scratch value | Fine-tuning typical |
|---|---|---|
| `backbone_lr_scale` | 1.0 (uniform) | 0.05 (protect pretrained) |
| `lr` | 3e-4 (EF) / 1e-4 (CoAtNet) | 1e-3 |
| `epochs` | 100 | 50 |
| `weight_decay` | 0.1 | 0.05 |
| `warmup_epochs` | 10 | 5 |
| `reg_warmup_epochs` | 10 | 5 |
| `label_smoothing` | 0.2 | 0.15 |
| `mixup_alpha` | 0.4 | 0.2 |
| `cutmix_prob` | 0.5 | 0.2 |

---

## Which experiment to run first

**Run EfficientFormer first (`efficientformer_scratch`).**

1. **Lighter (31M vs 42M params)** — shorter epoch time; quicker feedback.
2. **Simpler architecture** — only one attention stage, less risk of training
   instability from scratch than CoAtNet's two Transformer stages.
3. **Orthogonal inductive bias** — pure pooling + attention is different from
   TSM (channel shifts), ViT-BiGRU (global flat attention), and MaxViT
   (window+grid attention), so it adds ensemble diversity even at moderate accuracy.
4. **Validates the timm pipeline** — if the architecture loads and trains correctly,
   CoAtNet will work too.

Run CoAtNet second. Its C-C-T-T hybrid design is more expressive (~42M, deeper) but
the two Transformer stages may need more warmup epochs to stabilise from random
initialisation. Start it only once EfficientFormer confirms the pipeline is healthy.

### Commands

```bash
# Step 0 — install timm
uv sync

# Step 1 — EfficientFormer from scratch (run first)
cd src && python train.py experiment=efficientformer_scratch

# Step 2 — CoAtNet from scratch
cd src && python train.py experiment=coatnet_scratch

# Evaluate
cd src && python evaluate.py training.checkpoint_path=../models/efficientformer_scratch.pt
cd src && python evaluate.py training.checkpoint_path=../models/coatnet_scratch.pt

# Submission
cd src && python create_submission.py training.checkpoint_path=../models/efficientformer_scratch.pt
```

---

## Tuning suggestions

**Scale the architecture** (change `model.variant`, weights are never loaded):

```yaml
# Lighter EfficientFormer — faster experiments
model:
  variant: "efficientformer_l1.snap_dist_in1k"   # 12 M params

# Larger CoAtNet — higher capacity
model:
  variant: "coatnet_2_rw_224.sw_in12k"           # 75 M params
```

**More frames** — mean-pool head handles any T without architecture changes:

```yaml
dataset:
  num_frames: 8
training:
  batch_size: 8   # halve batch if VRAM is tight
```

**Ensemble** — both models have complementary inductive biases to the existing
TSM-ResNet and ViT-BiGRU checkpoints. Average softmax outputs for further gains.
