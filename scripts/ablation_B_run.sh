#!/usr/bin/env bash
# Exp B — VFL with 2 Transformer layers (rotating folds).
# Ablation axis: n_layers 3 → 2.
# Reference: vfl_ultra_rotating (V3) = 36.44% val-dir.
#
# USAGE
#   Direct:   bash scripts/ablation_B_run.sh
#   Via cron: managed by scripts/setup_ablation_crons.sh
#
# RESUME
#   Automatic. training.resume=true in the config restores from
#   models/ablation_vfl_2layers_last.pt on every restart.
#
# DONE
#   logs/.done_ablation_vfl_2layers_rotating created on success.
#
# LOGS
#   logs/ablation_vfl_2layers_rotating.log

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ablation_lib.sh"

run_ablation "ablation_vfl_2layers_rotating"
