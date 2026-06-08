#!/usr/bin/env bash
# Exp E — TSM SGDR T_mult=2.
# Ablation config: src/configs/experiment/ablation_tsm_sgdr_t2_rotating.yaml
#
# USAGE
#   Direct:   bash scripts/ablation_E_run.sh
#   Via cron: managed by scripts/setup_ablation_crons.sh
#
# RESUME   Automatic (resume: true in config, _last.pt restored each run).
# DONE     logs/.done_ablation_tsm_sgdr_t2_rotating created on success.
# LOGS     logs/ablation_tsm_sgdr_t2_rotating.log

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ablation_lib.sh"

run_ablation "ablation_tsm_sgdr_t2_rotating"
