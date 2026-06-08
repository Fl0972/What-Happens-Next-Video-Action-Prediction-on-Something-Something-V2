#!/usr/bin/env bash
# Exp D — VFL with 10-fold rotating validation (data ceiling).
# Ablation axis: n_folds 5 → 10, sgdr_t0 25 → 50.
# Reference: vfl_ultra_rotating (V3) = 36.44% val-dir.
#
# USAGE
#   Direct:   bash scripts/ablation_D_run.sh
#   Via cron: managed by scripts/setup_ablation_crons.sh
#
# RESUME
#   Automatic. training.resume=true in the config restores from
#   models/ablation_vfl_folds10_last.pt on every restart.
#   Note: the rolling_window (list of last 10 val accs) is also saved
#   in the _last.pt, so the rolling average continues correctly.
#
# DONE
#   logs/.done_ablation_vfl_folds10_rotating created on success.
#
# LOGS
#   logs/ablation_vfl_folds10_rotating.log

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/trackA_ablation_lib.sh"

run_ablation "trackA_ablation_vfl_folds10_rotating"
