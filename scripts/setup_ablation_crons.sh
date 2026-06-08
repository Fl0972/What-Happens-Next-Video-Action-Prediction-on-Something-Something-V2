#!/usr/bin/env bash
# setup_ablation_crons.sh — install cron entries for all 4 ablation experiments.
#
# Two entries per experiment:
#
#   @reboot   — restarts the watchdog immediately after a machine reboot.
#               A 30-second sleep lets network/filesystem mounts settle first.
#
#   */5 * * * * — heartbeat: every 5 minutes, check if the watchdog is still
#               alive. If it died unexpectedly (e.g., OOM-killed watchdog
#               process itself), the next heartbeat revives it.
#               If training is running normally, flock blocks the new instance
#               in under 1 ms and it exits silently — zero overhead.
#
# WHY TWO ENTRIES?
#   @reboot alone covers reboots but not "watchdog process died after boot".
#   */5 alone covers crash recovery but delays resume by up to 5 minutes
#   after a reboot (cron daemon may not fire immediately on boot).
#   Together they guarantee < 5 min downtime in all failure scenarios.
#
# IDEMPOTENT: running this script multiple times will not duplicate entries.
#
# USAGE
#   bash scripts/setup_ablation_crons.sh
#
# REMOVE
#   bash scripts/cancel_ablation_crons.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve absolute paths for the 4 run scripts.
declare -A RUNNERS=(
    [A]="$SCRIPTS_DIR/ablation_A_run.sh"
    [B]="$SCRIPTS_DIR/ablation_B_run.sh"
    [C]="$SCRIPTS_DIR/ablation_C_run.sh"
    [D]="$SCRIPTS_DIR/ablation_D_run.sh"
    [E]="$SCRIPTS_DIR/ablation_E_run.sh"
    [F]="$SCRIPTS_DIR/ablation_F_run.sh"
    [G]="$SCRIPTS_DIR/ablation_G_run.sh"
    [H]="$SCRIPTS_DIR/ablation_H_run.sh"
)

declare -A LABELS=(
    [A]="Exp A: TSM no focal"
    [B]="Exp B: VFL 2 layers"
    [C]="Exp C: TSM focal gamma=1"
    [D]="Exp D: VFL 10 folds"
    [E]="Exp E: TSM SGDR T_mult=2"
    [F]="Exp F: VFL 2-layer SGDR T_mult=2"
    [G]="Exp G: TSM one-cycle LR"
    [H]="Exp H: VFL 2-layer one-cycle LR"
)

# ---------------------------------------------------------------------------
# Helper: add a cron line only if not already present.
# ---------------------------------------------------------------------------
add_cron_line() {
    local line="$1"
    # Read current crontab (empty output if none set yet).
    local current
    current="$(crontab -l 2>/dev/null || true)"
    if echo "$current" | grep -qF "$line"; then
        echo "  [already present] $line"
    else
        ( echo "$current"; echo "$line" ) | crontab -
        echo "  [added] $line"
    fi
}

echo "=== Installing ablation cron entries ==="
echo ""

for exp in A B C D; do
    runner="${RUNNERS[$exp]}"
    label="${LABELS[$exp]}"

    if [[ ! -f "$runner" ]]; then
        echo "ERROR: runner not found: $runner" >&2
        exit 1
    fi
    chmod +x "$runner"

    echo "--- $label ---"

    # @reboot entry — start 30 s after boot to let filesystems settle.
    add_cron_line "@reboot sleep 30 && bash $runner"

    # Heartbeat entry — revive a crashed watchdog within 5 minutes.
    # The ABLATION_EXP tag makes it easy to grep for removal.
    add_cron_line "*/5 * * * * bash $runner  # ABLATION_EXP_${exp}"

    echo ""
done

echo "=== Done. Current ablation cron entries: ==="
crontab -l 2>/dev/null | grep -E '(ablation_[ABCD]_run|ABLATION_EXP)' || echo "(none found — check output above for errors)"
echo ""
echo "Run 'bash scripts/cancel_ablation_crons.sh' to remove all ablation entries."
echo "Run 'bash scripts/status_ablations.sh' to check experiment progress."
