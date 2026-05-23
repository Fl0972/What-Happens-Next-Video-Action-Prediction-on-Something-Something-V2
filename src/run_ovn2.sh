#!/bin/bash
#
# Overnight batch 2: train VideoMAE-Large (SSv2 pretrained) and assemble
# two submissions:
#
#   Submission A — VideoMAE-Large (new) × VideoMAE-K400-Base (existing)
#                  6 checkpoints, per-class weighted + uniform.
#                  The biggest single-model upgrade (307M params vs 86M)
#                  combined with the existing K400-pretrained diversity.
#
#   Submission B — TSM-ResNet50 (existing ovn1) only
#                  3 checkpoints. A pure CNN baseline for comparison —
#                  useful for the report and as a sanity reference.
#
# Sequencing: train → submission A → submission B. Each step blocks the next
# (set -e). Softmax is cached so the K400 ckpts in submission A reuse the
# ovn1 work without re-inferring.
#
# Usage (inside tmux, then Ctrl+b d to detach):
#     bash run_ovn2.sh

set -euo pipefail

# ─── knobs ────────────────────────────────────────────────────────────────
SRC=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/src
M=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/models
S=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/submissions
LOGS=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/logs
PYTHON=/usr/bin/python

# Large-model checkpoints + the ~5 GB resume file live on LOCAL disk, not the
# NFS home: the home has a 30 GB quota that a 307M-param model (top-k ≈ 1.2 GB
# each + a ~5 GB optimizer/EMA resume) blows past, which silently truncated
# checkpoints to 0 bytes. /Data is a 561 GB local NVMe partition that persists
# across reboots and isn't auto-cleaned, so resume-after-reboot still works.
# Only the Large model uses this; reused inputs (k400/tsm) and the tiny
# submission CSVs stay on NFS.
LM=/Data/florian.guillaumey/challenge_models

TAG=ovn2
NUM_FRAMES=4

# VideoMAE-Large training — heavier than Base (307M vs 86M).
# NOTE: MCG-NJU did *not* publish Large finetuned on SSv2 (only on K400). So
# we use the K400-finetuned Large checkpoint and disable warm_start_head
# (K400's head doesn't have matching SSv2-template rows for our classes).
# Trade-off: no SSv2-specific finetuning bias, but cleaner generalisation
# (no leakage) and a much bigger backbone.
LARGE_CKPT=$LM/videomae_${TAG}_large.pt
LARGE_HF_NAME=MCG-NJU/videomae-large-finetuned-kinetics
LARGE_BATCH=2          # batch_size=2 with gradient_checkpointing to fit on 19GB
LARGE_EPOCHS=15        # cut from Base's 20 — Large converges faster + needs more time per step

# Pseudo-labels (reuse existing CSV from videomae_4)
PSEUDO_CSV=$M/videomae_4_pseudo.csv
PSEUDO_THRESHOLD=0.85

# Existing checkpoints we're reusing (from ovn1 batch)
K400_LIST=(
    $M/videomae_ovn1_k400_top1.pt
    $M/videomae_ovn1_k400_top2.pt
    $M/videomae_ovn1_k400_top3.pt
)
TSM_LIST=(
    $M/tsm_r50_ovn1_top1.pt
    $M/tsm_r50_ovn1_top2.pt
    $M/tsm_r50_ovn1_top3.pt
)

# Submission outputs
SUB_A_PERCLASS=$S/ensemble_${TAG}_large_plus_k400_perclass.csv
SUB_A_UNIFORM=$S/ensemble_${TAG}_large_plus_k400_uniform.csv
SUB_B_PERCLASS=$S/ensemble_${TAG}_tsm_perclass.csv
SUB_B_UNIFORM=$S/ensemble_${TAG}_tsm_uniform.csv

mkdir -p "$M" "$S" "$LOGS" "$LM"
cd "$SRC"

echo "==================================================================="
echo "[$(date)] Starting overnight batch 2 (tag=$TAG, num_frames=$NUM_FRAMES)"
echo "  VideoMAE-Large checkpoint: $LARGE_CKPT"
echo "  Submission A → $SUB_A_PERCLASS"
echo "  Submission B → $SUB_B_PERCLASS"
echo "==================================================================="

DONE_DIR=$LOGS/.done_${TAG}
mkdir -p "$DONE_DIR"

# run_step <name> <cmd…> — runs the command (tee'd to a log) and writes a
# marker file on success. If the marker already exists, the step is skipped.
# After a reboot, re-launching the orchestration script picks up here.
run_step () {
    local name=$1; shift
    local log=$LOGS/${TAG}_${name}.log
    local marker=$DONE_DIR/$name
    if [ -f "$marker" ]; then
        echo
        echo "[$(date)] SKIP $name (marker $marker exists — already completed)"
        return 0
    fi
    echo
    echo "──────────────────────────────────────────────────────────────────"
    echo "[$(date)] STEP: $name  →  log: $log"
    echo "──────────────────────────────────────────────────────────────────"
    "$@" 2>&1 | tee "$log"
    # `set -e` would have aborted the script if "$@" failed; reaching here
    # means success. Mark done so we don't redo on the next launch.
    touch "$marker"
}

# ─── 1. Train VideoMAE-Large (SSv2 pretrained) ────────────────────────────
PSEUDO_ARGS=""
if [ -n "$PSEUDO_CSV" ] && [ -f "$PSEUDO_CSV" ]; then
    PSEUDO_ARGS="dataset.pseudo_labels_path=$PSEUDO_CSV dataset.pseudo_threshold=$PSEUDO_THRESHOLD"
fi

run_step train_videomae_large $PYTHON train.py experiment=videomae \
    model.checkpoint=$LARGE_HF_NAME \
    model.warm_start_head_from_ssv2=false \
    model.gradient_checkpointing=true \
    training.checkpoint_path=$LARGE_CKPT \
    training.batch_size=$LARGE_BATCH \
    training.epochs=$LARGE_EPOCHS \
    dataset.num_frames=$NUM_FRAMES \
    $PSEUDO_ARGS

# ─── 2. Sanity-eval VideoMAE-Large alone ──────────────────────────────────
run_step eval_videomae_large $PYTHON evaluate.py \
    training.checkpoint_path=$LARGE_CKPT training.ensemble_top_k=3 \
    dataset.num_frames=$NUM_FRAMES

# ─── 3. Submission A — VideoMAE-Large × VideoMAE-K400-Base ────────────────
LARGE_TOP=(
    $LM/videomae_${TAG}_large_top1.pt
    $LM/videomae_${TAG}_large_top2.pt
    $LM/videomae_${TAG}_large_top3.pt
)
LIST_A="[$(IFS=,; echo "${LARGE_TOP[*]},${K400_LIST[*]}")]"

# Pin the softmax cache to the NFS models dir. It otherwise defaults to the
# *first* checkpoint's parent — and since the Large checkpoints live on /Data
# while ssv2/k400 are cached under models/, that split would miss the Base
# caches and force a needless (and previously crash-prone) re-inference.
run_step ensemble_A_perclass $PYTHON ensemble_per_class.py \
    "training.checkpoint_paths=$LIST_A" \
    dataset.num_frames=$NUM_FRAMES \
    training.softmax_cache_dir=$M/_softmax_cache \
    dataset.submission_output=$SUB_A_PERCLASS

run_step ensemble_A_uniform $PYTHON create_submission.py \
    "training.checkpoint_paths=$LIST_A" \
    dataset.num_frames=$NUM_FRAMES \
    dataset.submission_output=$SUB_A_UNIFORM

# ─── 4. Submission B — TSM only ───────────────────────────────────────────
LIST_B="[$(IFS=,; echo "${TSM_LIST[*]}")]"

run_step ensemble_B_perclass $PYTHON ensemble_per_class.py \
    "training.checkpoint_paths=$LIST_B" \
    dataset.num_frames=$NUM_FRAMES \
    training.softmax_cache_dir=$M/_softmax_cache \
    dataset.submission_output=$SUB_B_PERCLASS

run_step ensemble_B_uniform $PYTHON create_submission.py \
    "training.checkpoint_paths=$LIST_B" \
    dataset.num_frames=$NUM_FRAMES \
    dataset.submission_output=$SUB_B_UNIFORM

echo
echo "==================================================================="
echo "[$(date)] ALL STEPS DONE"
echo "  Submission A (Large + K400):"
echo "    per-class: $SUB_A_PERCLASS"
echo "    uniform  : $SUB_A_UNIFORM"
echo "  Submission B (TSM only):"
echo "    per-class: $SUB_B_PERCLASS"
echo "    uniform  : $SUB_B_UNIFORM"
echo
echo "  Compare val-acc numbers:"
echo "    grep -E 'Top-1 accuracy|uniform soft-vote|per-class weighted' $LOGS/${TAG}_*.log"
echo "==================================================================="
