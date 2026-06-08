#!/usr/bin/env bash
# Honest ensemble (V-JEPA-ft + k400 + tsm, per-model preprocessing) followed by
# overnight pseudo-labeling retrain of V-JEPA. Runs under systemd via
# loop_until_done.sh; each step has a .done marker (skip on restart) and
# train.py resumes from /Data resume.pt — so kills/reboots are rescued.
set -euo pipefail

SRC=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/src
S=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/submissions
LOGS=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/logs
LM=/Data/florian.guillaumey/challenge_models
NFS=/users/eleves-b/2024/florian.guillaumey/Challenge-Modal/models
NFS_CACHE=$NFS/_softmax_cache
DATA_WIN=/Data/florian.guillaumey/val2_win16
DATA_SHIP=/Data/florian.guillaumey/val2
PYTHON=/usr/bin/python
export HF_HOME=/Data/florian.guillaumey/hf_cache

SSL=facebook/vjepa2-vitl-fpc64-256
S2=$LM/vjepa_ft_s2.pt                    # current best V-JEPA finetune (Kaggle 0.6361)
PSEUDO_CSV=$LM/vjepa_pseudo.csv
S2P=$LM/vjepa_ft_s2_pseudo.pt            # after pseudo-label retrain

TAG=vjepapseudo
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

# ─── ENSEMBLE PHASE ─────────────────────────────────────────────────────────
# 1. Cache V-JEPA-ft test softmax (3 top-k on window-capped data)
run_step cache_vjepa_test $PYTHON cache_test_softmax.py \
    "training.checkpoint_paths=[$LM/vjepa_ft_s2_top1.pt,$LM/vjepa_ft_s2_top2.pt,$LM/vjepa_ft_s2_top3.pt]" \
    training.softmax_cache_dir=$NFS_CACHE \
    dataset.test_dir=$DATA_WIN/test

# 2. Honest ensemble (V-JEPA-ft + k400 + tsm, uniform mean of 9 cached softmax)
run_step honest_ensemble $PYTHON honest_ensemble.py \
    --stems vjepa_ft_s2_top1 vjepa_ft_s2_top2 vjepa_ft_s2_top3 \
            videomae_ovn1_k400_top1 videomae_ovn1_k400_top2 videomae_ovn1_k400_top3 \
            tsm_r50_ovn1_top1 tsm_r50_ovn1_top2 tsm_r50_ovn1_top3 \
    --cache-dirs $NFS_CACHE \
    --test-dir $DATA_SHIP/test \
    --out $S/ensemble_honest.csv

# ─── OVERNIGHT PSEUDO-LABELING PHASE ───────────────────────────────────────
# 3. Generate pseudo-labels from V-JEPA-ft top-3 + TTA (conf >= 0.85)
run_step pseudo_label $PYTHON pseudo_label.py \
    "training.checkpoint_paths=[$LM/vjepa_ft_s2_top1.pt,$LM/vjepa_ft_s2_top2.pt,$LM/vjepa_ft_s2_top3.pt]" \
    dataset.test_dir=$DATA_WIN/test \
    dataset.pseudo_threshold=0.85 \
    dataset.pseudo_labels_output=$PSEUDO_CSV

# 4. Continue-finetune V-JEPA with pseudo labels (init from s2, full unfreeze + LLRD)
run_step ft_pseudo $PYTHON train.py experiment=vjepa \
    model.checkpoint=$SSL model.unfreeze_top_k=-1 model.gradient_checkpointing=true \
    training.init_weights_from=$S2 training.checkpoint_path=$S2P \
    training.epochs=4 training.batch_size=2 \
    training.lr=0.00002 training.max_lr=0.0002 \
    training.layerwise_lr_decay=true training.llrd_decay=0.75 \
    dataset.train_dir=$DATA_WIN/train dataset.val_dir=$DATA_WIN/val dataset.test_dir=$DATA_WIN/test \
    dataset.pseudo_labels_path=$PSEUDO_CSV dataset.pseudo_threshold=0.85

# 5. Eval pseudo-trained model on val_dir (TTA, top-3 snapshots)
run_step ft_pseudo_eval $PYTHON evaluate.py \
    training.checkpoint_path=$S2P training.ensemble_top_k=3 \
    dataset.val_dir=$DATA_WIN/val

# 6. Cache pseudo-model test softmax (3 top-k)
run_step cache_pseudo_test $PYTHON cache_test_softmax.py \
    "training.checkpoint_paths=[$LM/vjepa_ft_s2_pseudo_top1.pt,$LM/vjepa_ft_s2_pseudo_top2.pt,$LM/vjepa_ft_s2_pseudo_top3.pt]" \
    training.softmax_cache_dir=$NFS_CACHE \
    dataset.test_dir=$DATA_WIN/test

# 7. Final honest ensemble v2 (V-JEPA-pseudo + k400 + tsm)
run_step honest_ensemble_v2 $PYTHON honest_ensemble.py \
    --stems vjepa_ft_s2_pseudo_top1 vjepa_ft_s2_pseudo_top2 vjepa_ft_s2_pseudo_top3 \
            videomae_ovn1_k400_top1 videomae_ovn1_k400_top2 videomae_ovn1_k400_top3 \
            tsm_r50_ovn1_top1 tsm_r50_ovn1_top2 tsm_r50_ovn1_top3 \
    --cache-dirs $NFS_CACHE \
    --test-dir $DATA_SHIP/test \
    --out $S/ensemble_honest_v2.csv

# 8. Stand-alone V-JEPA-pseudo submission (without other models)
run_step ft_pseudo_submit $PYTHON create_submission.py \
    training.checkpoint_path=$S2P training.ensemble_top_k=3 \
    dataset.test_dir=$DATA_WIN/test \
    dataset.submission_output=$S/vjepa_ft_s2_pseudo.csv

echo "[$(date)] run_honest_pseudo.sh ALL DONE"
