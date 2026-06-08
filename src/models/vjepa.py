"""
V-JEPA 2 wrapper, adapting HuggingFace's ``VJEPA2ForVideoClassification`` to
this repo's ``(B, T, C, H, W) -> (B, num_classes)`` convention, with support for
**progressive unfreezing** and **layer-wise LR decay (LLRD)** finetuning.

Backbone choice (integrity)
---------------------------
For a *rules-clean* run use the self-supervised checkpoint
``facebook/vjepa2-vitl-fpc64-256`` (no SSv2 labels). The SSv2-finetuned
``...-ssv2`` checkpoint saw the test videos' labels and must not be used for a
competition submission.

Architecture (HF ``VJEPA2ForVideoClassification``)
--------------------------------------------------
``vjepa2`` (VJEPA2Model: ``encoder`` = patch-embed + 24 transformer layers +
final layernorm, plus a ``predictor`` used only by the JEPA objective) ->
``pooler`` (attentive pooler) -> ``classifier`` (Linear 1024->num_classes).
The predictor is irrelevant to classification and is always frozen.

Freezing / progressive unfreezing (``unfreeze_top_k``)
------------------------------------------------------
  unfreeze_top_k = 0   -> frozen encoder, only pooler + head train (attentive probe)
  unfreeze_top_k = K   -> top K encoder layers + final layernorm + pooler + head train
  unfreeze_top_k = -1  -> whole encoder + pooler + head train (full finetune)
The patch embedding and predictor stay frozen. Frozen submodules are kept in
eval() (no droppath/dropout drift) while trainable ones run in train().

Use ``build_vjepa_param_groups`` (mirrors the VideoMAE LLRD helper) so deeper
layers get higher LR — the standard BEiT/MAE finetuning recipe — over only the
currently-trainable parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class VJEPA2(nn.Module):
    DEFAULT_CHECKPOINT = "facebook/vjepa2-vitl-fpc64-256"  # SSL (label-clean)
    PRETRAIN_NUM_FRAMES = 16
    IMAGE_SIZE = 256

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 16,
        pretrained: bool = True,
        checkpoint: str = DEFAULT_CHECKPOINT,
        unfreeze_top_k: int = 0,
        gradient_checkpointing: bool = False,
        class_names: Optional[List[Optional[str]]] = None,
    ) -> None:
        super().__init__()
        from transformers import VJEPA2ForVideoClassification

        if num_frames % 2 != 0:
            raise ValueError(f"V-JEPA num_frames must be even (tubelet stride 2); got {num_frames}.")

        if not pretrained:
            from transformers import VJEPA2Config
            net = VJEPA2ForVideoClassification(VJEPA2Config(num_labels=num_classes, frames_per_clip=num_frames))
        else:
            net = VJEPA2ForVideoClassification.from_pretrained(checkpoint, ignore_mismatched_sizes=True)
            hidden = int(net.config.hidden_size)
            old_w = net.classifier.weight.data.clone()
            old_b = net.classifier.bias.data.clone()
            id2label = {int(k): str(v) for k, v in net.config.id2label.items()}
            new_head = nn.Linear(hidden, num_classes)
            nn.init.normal_(new_head.weight, std=float(getattr(net.config, "initializer_range", 0.02)))
            nn.init.zeros_(new_head.bias)
            # Warm-start only makes sense for the SSv2-finetuned checkpoint (which
            # has matching template rows). With an SSL backbone there is no such
            # head, so this is a no-op and the head stays random.
            if class_names is not None and old_w.shape[0] == len(id2label) and len(id2label) > num_classes:
                from models.videomae import match_classes_to_ssv2
                mapping = match_classes_to_ssv2(class_names, id2label)
                for new_i, src_i in mapping.items():
                    new_head.weight.data[new_i] = old_w[src_i]
                    new_head.bias.data[new_i] = old_b[src_i]
                print(f"VJEPA2 head warm-start: copied {len(mapping)}/{num_classes} rows.")
            else:
                print("VJEPA2: random head init.")
            net.classifier = new_head
            net.num_labels = num_classes
            net.config.num_labels = num_classes

        self.backbone = net
        self.unfreeze_top_k = int(unfreeze_top_k)
        self._trainable_layers: set = set()
        self._apply_freeze()

        if self.unfreeze_top_k != 0 and gradient_checkpointing:
            try:
                self.backbone.vjepa2.gradient_checkpointing_enable()
            except Exception as e:  # pragma: no cover
                print(f"VJEPA2: gradient_checkpointing_enable() failed ({e}); continuing.")

    # ------------------------------------------------------------------
    def _apply_freeze(self) -> None:
        enc = self.backbone.vjepa2
        for p in enc.parameters():            # freeze encoder + predictor
            p.requires_grad = False
        n = len(enc.encoder.layer)
        k = n if self.unfreeze_top_k < 0 else min(self.unfreeze_top_k, n)
        self._trainable_layers = set(range(n - k, n)) if k > 0 else set()
        for i in self._trainable_layers:
            for p in enc.encoder.layer[i].parameters():
                p.requires_grad = True
        if k > 0:
            for p in enc.encoder.layernorm.parameters():
                p.requires_grad = True
        for p in self.backbone.pooler.parameters():
            p.requires_grad = True
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_fz = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        mode = "frozen probe" if k == 0 else (f"full finetune ({n} layers)" if k == n else f"top-{k} unfrozen")
        print(f"VJEPA2 {mode}: {n_fz/1e6:.0f}M frozen, {n_tr/1e6:.1f}M trainable.")

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            enc = self.backbone.vjepa2
            enc.predictor.eval()
            enc.encoder.embeddings.eval()                 # patch embed always frozen
            for i, layer in enumerate(enc.encoder.layer):
                layer.train(i in self._trainable_layers)
            enc.encoder.layernorm.train(bool(self._trainable_layers))
        return self

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values_videos=video_batch).logits


# ---------------------------------------------------------------------------
# Layer-wise LR decay for V-JEPA (mirrors models.videomae.build_videomae_param_groups)
# ---------------------------------------------------------------------------

def _vjepa2_layer_id(name: str, num_layers: int) -> int:
    """0 = patch embed; 1..L = encoder block; L+1 = final norm / pooler / head."""
    if "vjepa2.encoder.embeddings" in name:
        return 0
    if "vjepa2.encoder.layer." in name:
        return int(name.split("encoder.layer.")[1].split(".")[0]) + 1
    return num_layers + 1


def build_vjepa_param_groups(
    model: nn.Module,
    base_lr: float,
    base_max_lr: float,
    weight_decay: float,
    decay_rate: float = 0.75,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """AdamW param groups + matching OneCycle max_lr list, over *trainable* params
    only, with deeper layers getting higher LR."""
    num_layers = len(model.backbone.vjepa2.encoder.layer)
    top = num_layers + 1
    NO_DECAY = ("bias", "norm", "query_tokens", "mask_token", "embeddings")

    buckets: Dict[Tuple[int, bool], List[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = _vjepa2_layer_id(name, num_layers)
        nd = any(s in name.lower() for s in NO_DECAY)
        buckets.setdefault((lid, nd), []).append(p)

    opt_groups: List[Dict[str, Any]] = []
    max_lrs: List[float] = []
    for (lid, nd), params in sorted(buckets.items()):
        scale = decay_rate ** (top - lid)
        opt_groups.append({
            "params": params,
            "lr": base_lr * scale,
            "weight_decay": 0.0 if nd else weight_decay,
            "_layer_id": lid,
            "_no_decay": nd,
        })
        max_lrs.append(base_max_lr * scale)
    return opt_groups, max_lrs
