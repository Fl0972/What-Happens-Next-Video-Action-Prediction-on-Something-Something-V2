#!/usr/bin/env bash
# status_ablations.sh — live status of all 4 ablation experiments.
#
# Shows for each experiment:
#   - Status: DONE / RUNNING / CRASHED / NOT STARTED
#   - Current epoch and best rolling-avg (read from _last.pt checkpoint)
#   - Last 3 lines of the experiment log
#
# USAGE
#   bash scripts/status_ablations.sh
#   watch -n 60 bash scripts/status_ablations.sh   # refresh every minute

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
LOG_DIR="$PROJ_ROOT/logs"
MODELS_DIR="$PROJ_ROOT/models"

# Find uv (same logic as trackA_ablation_lib.sh).
UV_BIN="$(command -v uv 2>/dev/null || true)"
[[ -z "$UV_BIN" ]] && UV_BIN="$HOME/.local/bin/uv"

# ---------------------------------------------------------------------------
# checkpoint_info LAST_PT TOTAL_EPOCHS
# Prints "epoch X / TOTAL_EPOCHS | best_rolling_avg Y" or an error.
# ---------------------------------------------------------------------------
checkpoint_info() {
    local last_pt="$1"
    local total_epochs="$2"
    if [[ ! -f "$last_pt" ]]; then
        echo "no checkpoint yet"
        return
    fi
    "$UV_BIN" run python3 - "$last_pt" "$total_epochs" 2>/dev/null <<'PYEOF'
import sys, torch
last_pt, total_epochs = sys.argv[1], int(sys.argv[2])
try:
    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", "?")
    best  = ckpt.get("best_val_accuracy", ckpt.get("val_accuracy", "?"))
    rw    = ckpt.get("rolling_window", [])
    rolling = f"{sum(rw)/len(rw):.4f}" if rw else "n/a"
    print(f"epoch {epoch}/{total_epochs} | best_rolling_avg {best:.4f} | "
          f"current_rolling {rolling}")
except Exception as e:
    print(f"(could not read checkpoint: {e})")
PYEOF
}

# ---------------------------------------------------------------------------
# Experiment definitions: NAME TOTAL_EPOCHS PROCESS_PATTERN
# ---------------------------------------------------------------------------
declare -a EXPS=(
    "A|trackA_ablation_tsm_no_focal_rotating|200|trackA_ablation_tsm_no_focal_rotating"
    "B|trackA_ablation_vfl_2layers_rotating|150|trackA_ablation_vfl_2layers_rotating"
    "C|trackA_ablation_tsm_focal_g1_rotating|200|trackA_ablation_tsm_focal_g1_rotating"
    "D|trackA_ablation_vfl_folds10_rotating|150|trackA_ablation_vfl_folds10_rotating"
    "E|trackA_ablation_tsm_sgdr_t2_rotating|200|trackA_ablation_tsm_sgdr_t2_rotating"
    "F|trackA_ablation_vfl2_sgdr_t2_rotating|200|trackA_ablation_vfl2_sgdr_t2_rotating"
    "G|trackA_ablation_tsm_onecycle_rotating|200|trackA_ablation_tsm_onecycle_rotating"
    "H|trackA_ablation_vfl2_onecycle_rotating|200|trackA_ablation_vfl2_onecycle_rotating"
)

declare -A DESCRIPTIONS=(
    [A]="TSM, CE loss (no focal)          — axis: focal γ 2→0"
    [B]="VFL, 2 Transformer layers        — axis: n_layers 3→2"
    [C]="TSM, focal γ=1                  — axis: focal γ 2→1"
    [D]="VFL, 10-fold rotating            — axis: n_folds 5→10"
    [E]="TSM, SGDR T_mult=2              — axis: LR shape (growing cycles)"
    [F]="VFL 2-layer, SGDR T_mult=2      — axis: LR shape (growing cycles)"
    [G]="TSM, one-cycle 3e-3             — axis: LR shape (single sweep)"
    [H]="VFL 2-layer, one-cycle 2.4e-3   — axis: LR shape (single sweep)"
)

printf '\n%s\n' "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf '  ABLATION EXPERIMENT STATUS    %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
printf '%s\n\n' "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for entry in "${EXPS[@]}"; do
    IFS='|' read -r exp_id exp_name total_epochs proc_pattern <<< "$entry"

    done_marker="$LOG_DIR/.done_${exp_name}"
    log_file="$LOG_DIR/${exp_name}.log"
    last_pt="$MODELS_DIR/${exp_name}_last.pt"
    best_pt="$MODELS_DIR/${exp_name}.pt"

    # ---- Determine status ----
    if [[ -f "$done_marker" ]]; then
        status="✅ DONE"
        status_color="\033[32m"
    elif pgrep -f "experiment=${proc_pattern}" > /dev/null 2>&1; then
        status="🔄 RUNNING"
        status_color="\033[34m"
    elif [[ -f "$last_pt" ]]; then
        status="💥 CRASHED (last checkpoint exists)"
        status_color="\033[33m"
    else
        status="⏳ NOT STARTED"
        status_color="\033[37m"
    fi

    # ---- Print block ----
    printf "${status_color}Exp %s — %s\033[0m\n" "$exp_id" "${DESCRIPTIONS[$exp_id]}"
    printf "  Config  : %s\n" "$exp_name"
    printf "  Status  : %s\n" "$status"
    printf "  Progress: %s\n" "$(checkpoint_info "$last_pt" "$total_epochs")"

    if [[ -f "$best_pt" ]]; then
        best_size=$(du -sh "$best_pt" 2>/dev/null | cut -f1)
        printf "  Best .pt: %s (%s)\n" "$best_pt" "$best_size"
    fi

    if [[ -f "$log_file" ]]; then
        printf "  Last log:\n"
        tail -n 3 "$log_file" | sed 's/^/    /'
    else
        printf "  Last log: (not started yet)\n"
    fi

    printf '\n'
done

printf '%s\n' "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf 'Cron entries:\n'
crontab -l 2>/dev/null | grep -E '(ablation_[ABCD]_run|ABLATION_EXP)' \
    | sed 's/^/  /' \
    || printf '  (none installed — run trackA_setup_ablation_crons.sh)\n'
printf '%s\n\n' "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
