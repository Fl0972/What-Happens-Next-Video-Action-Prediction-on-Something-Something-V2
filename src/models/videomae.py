"""
VideoMAE wrapper that adapts HuggingFace's ``VideoMAEForVideoClassification``
to this repo's ``(B, T, C, H, W) -> (B, num_classes)`` convention.

Default checkpoint: ``MCG-NJU/videomae-base-finetuned-ssv2`` — VideoMAE
ViT-B/16 with tubelet embeddings, pretrained on Kinetics-400 with masked
reconstruction and finetuned on Something-Something v2 (174 classes).

Head warm-start
---------------
This challenge's classes are a subset of SSv2's 174 templates (same wording).
When ``class_names`` is supplied, we don't throw the pretrained 174-class head
away: each new class whose (normalized) name matches an SSv2 template inherits
that template's classifier row (weight + bias). Unmatched classes fall back to
the usual small-normal initialization. This gives a strong starting point and
faster convergence than a random head.

References:
    Tong, Song, Wang, Wang, "VideoMAE: Masked Autoencoders are Data-Efficient
    Learners for Self-Supervised Video Pre-Training", NeurIPS 2022,
    arXiv:2203.12602.

    HuggingFace transformers implementation:
    https://huggingface.co/docs/transformers/model_doc/videomae
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _normalize_label(s: str) -> str:
    """Canonicalize a class/template string for matching.

    Strips a leading ``"012_"`` folder prefix, turns ``_`` into spaces,
    drops the SSv2 ``[ ]`` placeholders and punctuation, lowercases, and
    collapses whitespace.
    """
    s = re.sub(r"^\d+_", "", s)
    s = s.replace("_", " ")
    s = s.lower()
    s = s.replace("[", "").replace("]", "")
    s = re.sub(r"[,\.;:]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_classes_to_ssv2(
    class_names: List[Optional[str]],
    id2label: Dict[int, str],
) -> Dict[int, int]:
    """Map ``new_class_index -> ssv2_class_index`` by (normalized) name.

    Exact normalized match takes priority. If none, a unique prefix match is
    accepted (handles truncated folder names such as
    ``..._but_something_`` for ``..., but [something] is empty``). Ambiguous
    or missing entries are simply left unmapped.
    """
    norm_to_idx: Dict[str, int] = {}
    for idx, lbl in id2label.items():
        norm_to_idx.setdefault(_normalize_label(str(lbl)), int(idx))

    mapping: Dict[int, int] = {}
    for new_i, name in enumerate(class_names):
        if not name:
            continue
        q = _normalize_label(str(name))
        if q in norm_to_idx:
            mapping[new_i] = norm_to_idx[q]
            continue
        cands = [i for t, i in norm_to_idx.items() if t == q or t.startswith(q + " ")]
        if len(cands) == 1:
            mapping[new_i] = cands[0]
    return mapping


class VideoMAE(nn.Module):
    """ViT-B/16 video classifier built on top of HF's VideoMAE."""

    DEFAULT_CHECKPOINT = "MCG-NJU/videomae-base-finetuned-ssv2"
    PRETRAIN_NUM_FRAMES = 16  # the number the checkpoint was finetuned with

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 16,
        pretrained: bool = True,
        checkpoint: str = DEFAULT_CHECKPOINT,
        gradient_checkpointing: bool = False,
        class_names: Optional[List[Optional[str]]] = None,
    ) -> None:
        super().__init__()
        # Lazy import so other models don't pay for transformers being absent.
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification

        # VideoMAE's tubelet embedding has temporal stride 2, so num_frames
        # must be even. Position embeddings are sinusoidal and recomputed
        # from `num_frames` at construction, so any even value works — but
        # values far from the pretraining setting (16) are out-of-distribution
        # for the attention patterns the model learned.
        if num_frames % 2 != 0:
            raise ValueError(
                f"VideoMAE num_frames must be even (tubelet stride is 2); got {num_frames}."
            )
        if num_frames != self.PRETRAIN_NUM_FRAMES:
            print(
                f"[VideoMAE] num_frames={num_frames} != pretraining default "
                f"({self.PRETRAIN_NUM_FRAMES}). Position embeddings will be "
                f"recomputed sinusoidally; this is OOD for the pretrained "
                f"attention patterns — verify on val_dir."
            )

        if not pretrained:
            cfg = VideoMAEConfig(num_labels=num_classes, num_frames=num_frames)
            self.backbone = VideoMAEForVideoClassification(cfg)
        else:
            # Load the full pretrained model *with its original 174-class head*
            # so we can reuse the relevant rows below. Override num_frames so
            # the position-embedding buffer is sized for our actual input;
            # ignore_mismatched_sizes lets the size-changed buffer (and the
            # 174→num_classes head) be reinitialised rather than blocking load.
            net = VideoMAEForVideoClassification.from_pretrained(
                checkpoint,
                num_frames=num_frames,
                ignore_mismatched_sizes=True,
            )
            hidden = int(net.config.hidden_size)
            old_w = net.classifier.weight.data.clone()   # (num_old, hidden)
            old_b = net.classifier.bias.data.clone()     # (num_old,)
            id2label = {int(k): str(v) for k, v in net.config.id2label.items()}

            new_head = nn.Linear(hidden, num_classes)
            # Mirror HF's classifier init (normal, std = config.initializer_range).
            nn.init.normal_(new_head.weight, std=float(getattr(net.config, "initializer_range", 0.02)))
            nn.init.zeros_(new_head.bias)

            if class_names is not None:
                mapping = match_classes_to_ssv2(class_names, id2label)
                for new_i, ssv2_i in mapping.items():
                    new_head.weight.data[new_i] = old_w[ssv2_i]
                    new_head.bias.data[new_i] = old_b[ssv2_i]
                unmatched = [
                    f"{i}:{class_names[i]}" for i in range(num_classes)
                    if class_names[i] and i not in mapping
                ]
                print(
                    f"VideoMAE head warm-start: copied {len(mapping)}/{num_classes} rows "
                    f"from the SSv2 head."
                    + (f" Unmatched: {unmatched}" if unmatched else "")
                )
            else:
                print("VideoMAE: random head init (no class_names provided for warm-start).")

            net.classifier = new_head
            net.num_labels = num_classes
            net.config.num_labels = num_classes
            if class_names is not None:
                net.config.id2label = {i: (class_names[i] or f"class_{i}") for i in range(num_classes)}
                net.config.label2id = {v: k for k, v in net.config.id2label.items()}
            self.backbone = net

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """video_batch: (B, T, C, H, W) -> logits (B, num_classes)."""
        return self.backbone(pixel_values=video_batch).logits


# ---------------------------------------------------------------------------
# Layer-wise LR decay (LLRD)
# ---------------------------------------------------------------------------
#
# Standard ViT finetuning trick: deeper layers (closer to the head) get higher
# LR, shallower layers (closer to input) get smaller LR by an exponential
# decay factor. Background:
#   Clark et al., "ELECTRA" (ICLR 2020), §3.1 — introduced for BERT.
#   Bao et al., "BEiT" (ICLR 2022), arXiv:2106.08254 — popularised for ViT.
# Generic ViT/MAE finetuning recipes (e.g. He et al., "MAE", CVPR 2022, Tab 9)
# use decay 0.65–0.75. Bias / LayerNorm weights are excluded from weight decay,
# also a standard ViT recipe.

def _videomae_layer_id(name: str, num_encoder_layers: int) -> int:
    """Return the layer index for an LLRD bucket.

    Convention (small index = closer to input, large = closer to head):
      0: patch + position embeddings
      1..num_encoder_layers: transformer encoder block index + 1
      num_encoder_layers + 1: final layernorm / fc_norm / classifier head
    """
    if "videomae.embeddings" in name:
        return 0
    if "videomae.encoder.layer." in name:
        block_idx = int(name.split("encoder.layer.")[1].split(".")[0])
        return block_idx + 1
    return num_encoder_layers + 1


def build_videomae_param_groups(
    model: nn.Module,
    base_lr: float,
    base_max_lr: float,
    weight_decay: float,
    decay_rate: float = 0.75,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Build AdamW param groups + the matching ``max_lr`` list for OneCycleLR.

    Returns ``(opt_groups, max_lrs)`` such that the *i*-th element of
    ``max_lrs`` is the OneCycleLR peak LR for the *i*-th param group.
    """
    backbone = model.backbone  # VideoMAEForVideoClassification
    num_encoder = len(backbone.videomae.encoder.layer)
    top = num_encoder + 1

    # Standard ViT no-decay set: biases, all LayerNorm gains, position/CLS
    # embeddings (token-style parameters that should not be shrunk).
    NO_DECAY_SUBSTRINGS = (
        "bias",
        "LayerNorm.weight",
        "layernorm.weight",
        "position_embeddings",
        "cls_token",
    )

    buckets: Dict[Tuple[int, bool], List[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = _videomae_layer_id(name, num_encoder)
        nd = any(s in name for s in NO_DECAY_SUBSTRINGS)
        buckets.setdefault((lid, nd), []).append(p)

    opt_groups: List[Dict[str, Any]] = []
    max_lrs: List[float] = []
    # Sort by layer id so the order is reproducible and easy to inspect.
    for (lid, nd), params in sorted(buckets.items()):
        scale = decay_rate ** (top - lid)
        lr_g = base_lr * scale
        max_lr_g = base_max_lr * scale
        wd = 0.0 if nd else weight_decay
        opt_groups.append({
            "params": params,
            "lr": lr_g,
            "weight_decay": wd,
            "_layer_id": lid,
            "_no_decay": nd,
        })
        max_lrs.append(max_lr_g)
    return opt_groups, max_lrs
