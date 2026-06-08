#!/usr/bin/env bash
# Exp F — VFL 2-layer SGDR T_mult=2.
# Ablation config: src/configs/experiment/ablation_vfl2_sgdr_t2_rotating.yaml
#
# USAGE
#   Direct:   bash scripts/ablation_F_run.sh
#   Via cron: managed by scripts/setup_ablation_crons.sh
#
# RESUME   Automatic (resume: true in config, _last.pt restored each run).
# DONE     logs/.done_ablation_vfl2_sgdr_t2_rotating created on success.
# LOGS     logs/ablation_vfl2_sgdr_t2_rotating.log

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/trackA_ablation_lib.sh"

run_ablation "trackA_ablation_vfl2_sgdr_t2_rotating"
