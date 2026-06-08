#!/usr/bin/env bash
# Exp H — VFL 2-layer one-cycle LR.
# Ablation config: src/configs/experiment/ablation_vfl2_onecycle_rotating.yaml
#
# USAGE
#   Direct:   bash scripts/ablation_H_run.sh
#   Via cron: managed by scripts/setup_ablation_crons.sh
#
# RESUME   Automatic (resume: true in config, _last.pt restored each run).
# DONE     logs/.done_ablation_vfl2_onecycle_rotating created on success.
# LOGS     logs/ablation_vfl2_onecycle_rotating.log

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/trackA_ablation_lib.sh"

run_ablation "trackA_ablation_vfl2_onecycle_rotating"
