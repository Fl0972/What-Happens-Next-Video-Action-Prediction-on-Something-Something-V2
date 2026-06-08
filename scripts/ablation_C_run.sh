#!/usr/bin/env bash
# Exp C — TSM with focal loss γ=1 (rotating folds).
# Ablation axis: focal_gamma 2.0 → 1.0.
# Reference: tsm_ultra_v2_rotating (T3) = 39.45% val-dir.
#
# USAGE
#   Direct:   bash scripts/ablation_C_run.sh
#   Via cron: managed by scripts/setup_ablation_crons.sh
#
# RESUME
#   Automatic. training.resume=true in the config restores from
#   models/ablation_tsm_focal_g1_last.pt on every restart.
#
# DONE
#   logs/.done_ablation_tsm_focal_g1_rotating created on success.
#
# LOGS
#   logs/ablation_tsm_focal_g1_rotating.log

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ablation_lib.sh"

run_ablation "ablation_tsm_focal_g1_rotating"
