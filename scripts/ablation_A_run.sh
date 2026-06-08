#!/usr/bin/env bash
# Exp A — TSM without focal loss (rotating folds).
# Ablation axis: focal_gamma 2.0 → CE loss.
# Reference: tsm_ultra_v2_rotating (T3) = 39.45% val-dir.
#
# USAGE
#   Direct:   bash scripts/ablation_A_run.sh
#   Via cron: managed by scripts/setup_ablation_crons.sh
#
# RESUME
#   Automatic. The script checks models/ablation_tsm_no_focal_rotating.pt
#   (best checkpoint) and training.resume=true in the config restores from
#   models/ablation_tsm_no_focal_last.pt on every restart.
#
# DONE
#   A logs/.done_ablation_tsm_no_focal_rotating file is created on success.
#   Delete it to retrain from scratch.
#
# LOGS
#   Appended to logs/ablation_tsm_no_focal_rotating.log (timestamps included).

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ablation_lib.sh"

run_ablation "ablation_tsm_no_focal_rotating"
