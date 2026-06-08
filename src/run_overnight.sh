#!/bin/bash
#
# Ensemble + submission only (no training).
#
# Reuses the 9 checkpoints from the ovn1 batch (3 × VideoMAE-SSv2,
# 3 × VideoMAE-K400, 3 × TSM-R50, all at num_frames=4) and writes two
# submission CSVs:
#   - per-class weighted ensemble (recommended)
#   - uniform soft-vote (baseline for comparison)
#
# Softmax tensors are cached, so the uniform run after the per-class one
# is effectively free.
#
# Usage (inside tmux, then Ctrl+b d to detach):
#     bash run_overnight.sh

set -euo pipefail

# ─── knobs ────────────────────────────────────────────────────────────────
SRC=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/src
M=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/models
S=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/submissions
LOGS=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/logs
PYTHON=/usr/bin/python

TAG=ovn1                # which trained batch to ensemble (matches checkpoint names)
NUM_FRAMES=4            # must match what the checkpoints were trained with
SUB_PERCLASS=$S/ensemble_${TAG}_perclass.csv
SUB_UNIFORM=$S/ensemble_${TAG}_uniform.csv

mkdir -p "$S" "$LOGS"
cd "$SRC"

# ─── checkpoint list ──────────────────────────────────────────────────────
EXPLICIT_LIST="[\
$M/videomae_${TAG}_ssv2_top1.pt,\
$M/videomae_${TAG}_ssv2_top2.pt,\
$M/videomae_${TAG}_ssv2_top3.pt,\
$M/videomae_${TAG}_k400_top1.pt,\
$M/videomae_${TAG}_k400_top2.pt,\
$M/videomae_${TAG}_k400_top3.pt,\
$M/tsm_r50_${TAG}_top1.pt,\
$M/tsm_r50_${TAG}_top2.pt,\
$M/tsm_r50_${TAG}_top3.pt]"

echo "==================================================================="
echo "[$(date)] Ensemble + submission (tag=$TAG, num_frames=$NUM_FRAMES)"
echo "==================================================================="

# ─── helper ───────────────────────────────────────────────────────────────
run_step () {
    local name=$1; shift
    local log=$LOGS/${TAG}_${name}.log
    echo
    echo "──────────────────────────────────────────────────────────────────"
    echo "[$(date)] STEP: $name  →  log: $log"
    echo "──────────────────────────────────────────────────────────────────"
    "$@" 2>&1 | tee "$log"
}

# ─── 1. Per-class weighted ensemble ───────────────────────────────────────
run_step ensemble_perclass $PYTHON ensemble_per_class.py \
    "training.checkpoint_paths=$EXPLICIT_LIST" \
    dataset.num_frames=$NUM_FRAMES \
    dataset.submission_output=$SUB_PERCLASS

# ─── 2. Uniform ensemble (cheap once softmax cache is warm) ───────────────
run_step ensemble_uniform $PYTHON create_submission.py \
    "training.checkpoint_paths=$EXPLICIT_LIST" \
    dataset.num_frames=$NUM_FRAMES \
    dataset.submission_output=$SUB_UNIFORM

echo
echo "==================================================================="
echo "[$(date)] DONE"
echo "  Per-class submission: $SUB_PERCLASS"
echo "  Uniform   submission: $SUB_UNIFORM"
echo
echo "  Compare the two val numbers from the per-class log:"
echo "    grep -E 'uniform soft-vote|per-class weighted' $LOGS/${TAG}_ensemble_perclass.log"
echo "==================================================================="
