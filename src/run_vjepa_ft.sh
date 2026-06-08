#!/usr/bin/env bash
# Progressive-unfreezing + LLRD finetuning of the *self-supervised* V-JEPA 2
# ViT-L on the window-capped (rules-clean) frames, then eval + submission.
#
# Stage 0 (already done): frozen attentive probe -> vjepa_ssl_probe_top1.pt
# Stage 1: unfreeze top 6 encoder layers + LLRD, warm-started from the probe
# Stage 2: unfreeze the whole encoder + LLRD, warm-started from stage 1
#
# Durability: run via loop_until_done.sh under the systemd user service. Each
# step writes a .done marker (skip on restart); train.py resumes from its
# <ckpt>.resume.pt on /Data, so a kill/reboot mid-stage is rescued.
set -euo pipefail

SRC=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/src
S=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/submissions
LOGS=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/logs
LM=/Data/florian.guillaumey/challenge_models
DATA=/Data/florian.guillaumey/val2_win16
PYTHON=/usr/bin/python
export HF_HOME=/Data/florian.guillaumey/hf_cache   # keep V-JEPA downloads off the NFS quota

SSL=facebook/vjepa2-vitl-fpc64-256                 # self-supervised, label-clean
PROBE=$LM/vjepa_ssl_probe_top1.pt                  # stage-0 frozen probe (warm-start)
S1=$LM/vjepa_ft_s1.pt
S2=$LM/vjepa_ft_s2.pt

TAG=vjepaft
DONE_DIR=$LOGS/.done_${TAG}
mkdir -p "$DONE_DIR" "$S" "$LM"
cd "$SRC"

run_step () {
    local name=$1; shift
    local marker="$DONE_DIR/$name"
    local log="$LOGS/${TAG}_${name}.log"
    if [ -f "$marker" ]; then echo "[skip] $name (done)"; return 0; fi
    echo "[$(date)] [run] $name"
    "$@" 2>&1 | tee "$log"
    touch "$marker"
}

DATA_ARGS="dataset.train_dir=$DATA/train dataset.val_dir=$DATA/val dataset.test_dir=$DATA/test"

# ─── Stage 1: top-6 unfrozen + LLRD (warm-start from frozen probe) ──────────
run_step ft_stage1 $PYTHON train.py experiment=vjepa \
    model.checkpoint=$SSL model.unfreeze_top_k=6 model.gradient_checkpointing=true \
    training.init_weights_from=$PROBE training.checkpoint_path=$S1 \
    training.epochs=4 training.batch_size=4 \
    training.lr=0.00005 training.max_lr=0.0005 \
    training.layerwise_lr_decay=true training.llrd_decay=0.75 \
    $DATA_ARGS

# ─── Stage 2: full encoder unfrozen + LLRD (warm-start from stage 1) ────────
run_step ft_stage2 $PYTHON train.py experiment=vjepa \
    model.checkpoint=$SSL model.unfreeze_top_k=-1 model.gradient_checkpointing=true \
    training.init_weights_from=$S1 training.checkpoint_path=$S2 \
    training.epochs=4 training.batch_size=2 \
    training.lr=0.00002 training.max_lr=0.0002 \
    training.layerwise_lr_decay=true training.llrd_decay=0.75 \
    $DATA_ARGS

# ─── Eval final model on val_dir (snapshot ensemble of top-3) ───────────────
run_step ft_eval $PYTHON evaluate.py \
    training.checkpoint_path=$S2 training.ensemble_top_k=3 \
    dataset.val_dir=$DATA/val

# ─── Submission: V-JEPA finetuned, 10-view TTA, top-3 snapshots ────────────
run_step ft_submit $PYTHON create_submission.py \
    training.checkpoint_path=$S2 training.ensemble_top_k=3 \
    dataset.test_dir=$DATA/test \
    dataset.submission_output=$S/vjepa_ft_s2.csv

echo "[$(date)] run_vjepa_ft.sh ALL DONE"
