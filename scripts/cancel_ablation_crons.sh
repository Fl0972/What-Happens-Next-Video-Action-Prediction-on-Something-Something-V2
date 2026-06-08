#!/usr/bin/env bash
# cancel_ablation_crons.sh — remove all cron entries added by setup_ablation_crons.sh.
#
# Matches lines containing 'ablation_A_run', 'ablation_B_run', 'ablation_C_run',
# 'ablation_D_run', or the 'ABLATION_EXP' tag.
#
# Does NOT kill any in-progress training process — use status_ablations.sh to
# identify PIDs and kill manually if needed.
#
# USAGE
#   bash scripts/cancel_ablation_crons.sh

set -euo pipefail

echo "=== Removing ablation cron entries ==="

current="$(crontab -l 2>/dev/null || true)"
if [[ -z "$current" ]]; then
    echo "Crontab is empty. Nothing to remove."
    exit 0
fi

filtered="$(echo "$current" | grep -vE '(ablation_[ABCD]_run|ABLATION_EXP)' || true)"

if [[ "$current" == "$filtered" ]]; then
    echo "No ablation entries found in crontab."
    exit 0
fi

removed="$(echo "$current" | grep -E '(ablation_[ABCD]_run|ABLATION_EXP)' || true)"
echo "Removing:"
echo "$removed" | sed 's/^/  /'

echo "$filtered" | crontab -
echo ""
echo "Done. Remaining crontab:"
crontab -l 2>/dev/null || echo "(empty)"
