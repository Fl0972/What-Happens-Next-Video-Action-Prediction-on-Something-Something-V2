#!/usr/bin/env bash
# Time-budgeted: train one more diverse honest model + cache test + ensemble.
# Target: done well before today's 23:00 deadline.
#
# VideoMAE-Large K400 (no SSv2 leak) finetuned at 16f / 224 (native VideoMAE crop)
# on the window-capped data. Different architecture (pixel MAE) + different
# pretraining (K400) + different crop (224) from V-JEPA. Should add real
# diversity to the existing 14-model honest ensemble.
#
# NO eval step (per user request) — train.py's internal per-epoch train-split
# val stays (it's integral to top-k snapshot selection, only ~10 min/epoch).
# All steps systemd-durable with run_step markers + train.py resume.
set -euo pipefail

SRC=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/src
S=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/submissions
LOGS=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/logs
LM=/Data/florian.guillaumey/challenge_models
NFS_CACHE=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/models/_softmax_cache
DATA_WIN=/Data/florian.guillaumey/val2_win16
DATA_SHIP=/Data/florian.guillaumey/val2
PYTHON=/usr/bin/python
export HF_HOME=/Data/florian.guillaumey/hf_cache

CKPT=$LM/vmae_l_k400_win.pt
TAG=vmaewin
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

# 1. Train VideoMAE-Large K400 16f/224 on win-capped, full finetune + LLRD, 3 epochs
#    Time budget: ~5h. Native 224 res = ~25% fewer tokens than 256 -> faster than V-JEPA stage 2.
run_step train $PYTHON train.py experiment=videomae \
    model.checkpoint=MCG-NJU/videomae-large-finetuned-kinetics \
    model.warm_start_head_from_ssv2=false \
    model.gradient_checkpointing=true \
    training.checkpoint_path=$CKPT \
    training.epochs=3 training.batch_size=2 \
    training.lr=0.00002 training.max_lr=0.0002 \
    training.layerwise_lr_decay=true training.llrd_decay=0.75 \
    dataset.train_dir=$DATA_WIN/train dataset.val_dir=$DATA_WIN/val dataset.test_dir=$DATA_WIN/test \
    dataset.num_frames=16

# 2. Cache test softmax for top-3 snapshots (~25 min)
run_step cache_test $PYTHON cache_test_softmax.py \
    "training.checkpoint_paths=[$LM/vmae_l_k400_win_top1.pt,$LM/vmae_l_k400_win_top2.pt,$LM/vmae_l_k400_win_top3.pt]" \
    training.softmax_cache_dir=$NFS_CACHE \
    dataset.test_dir=$DATA_WIN/test

# 3. 17-model uniform ensemble (the 14-model v4 + 3 new VideoMAE-Large-win snapshots)
run_step ensemble_v5 $PYTHON honest_ensemble.py \
    --stems vjepa_ft_s2_top1 vjepa_ft_s2_top2 vjepa_ft_s2_top3 \
            vjepa_ft_s2_pseudo_top1 vjepa_ft_s2_pseudo_top2 vjepa_ft_s2_pseudo_top3 \
            videomae_ovn1_k400_top1 videomae_ovn1_k400_top2 videomae_ovn1_k400_top3 \
            tsm_r50_ovn1_top1 tsm_r50_ovn1_top2 tsm_r50_ovn1_top3 \
            videomae_ovn2_large_attempt1_top2 videomae_ovn2_large_attempt1_top3 \
            vmae_l_k400_win_top1 vmae_l_k400_win_top2 vmae_l_k400_win_top3 \
    --cache-dirs $NFS_CACHE \
    --test-dir $DATA_SHIP/test \
    --out $S/ensemble_honest_v5.csv

echo "[$(date)] run_vmae_l_win.sh ALL DONE"
