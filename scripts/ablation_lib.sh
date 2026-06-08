#!/usr/bin/env bash
# ablation_lib.sh — shared watchdog core for ablation experiments.
#
# SOURCE this file from each experiment's run script:
#   source "$(dirname "${BASH_SOURCE[0]}")/ablation_lib.sh"
#   run_ablation "ablation_tsm_no_focal_rotating"
#
# What this provides:
#   run_ablation EXPERIMENT_NAME
#     - Acquires an exclusive flock so only one instance runs per experiment.
#       Duplicate invocations (from cron heartbeat) exit immediately.
#     - Checks a done-marker file; skips re-running a finished experiment.
#     - Loops indefinitely until training exits 0, with exponential backoff
#       (60 s → 120 s → 240 s → 300 s cap) between crash recoveries.
#     - Logs every action with timestamps to logs/<experiment>.log.
#     - Creates logs/.done_<experiment> on success.

# ---------------------------------------------------------------------------
# Resolve paths relative to this lib file (works with absolute and relative
# invocations because we cd into the directory first).
# ---------------------------------------------------------------------------
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$_LIB_DIR/.." && pwd)"
SRC_DIR="$PROJ_ROOT/src"
LOG_DIR="$PROJ_ROOT/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Locate uv: prefer the one on PATH, fall back to the standard user-local
# install location so cron (which has a minimal PATH) still works.
# ---------------------------------------------------------------------------
UV_BIN="$(command -v uv 2>/dev/null || true)"
if [[ -z "$UV_BIN" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
fi
if [[ ! -x "$UV_BIN" ]]; then
    echo "ERROR: cannot find uv at '$UV_BIN'. Set PATH or install uv." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# log TAG FILE MESSAGE  — timestamp-prefixed, write to file AND stdout.
# ---------------------------------------------------------------------------
log() {
    local tag="$1" log_file="$2"; shift 2
    printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$tag" "$*" \
        | tee -a "$log_file"
}

# ---------------------------------------------------------------------------
# run_ablation EXPERIMENT_NAME
# ---------------------------------------------------------------------------
run_ablation() {
    local experiment="$1"
    local tag="$experiment"

    local lock_file="/tmp/ablation_${experiment}.lock"
    local done_marker="$LOG_DIR/.done_${experiment}"
    local log_file="$LOG_DIR/${experiment}.log"

    # ------------------------------------------------------------------
    # Acquire exclusive lock (non-blocking).
    # If another instance of this experiment is already running — including
    # a previous cron invocation that is still in its training loop — we
    # exit silently in under 1 ms.  The kernel releases the lock
    # automatically when the holding process dies for any reason.
    # ------------------------------------------------------------------
    exec 9>"$lock_file"
    if ! flock -n 9; then
        # Completely silent: this is the normal case when cron fires
        # while training is already in progress.
        exit 0
    fi
    # Release lock on any exit (normal, error, or signal).
    trap 'flock -u 9 2>/dev/null; exec 9>&-' EXIT

    log "$tag" "$log_file" "=== Watchdog started (PID $$, host $(hostname)) ==="
    log "$tag" "$log_file" "Proj root : $PROJ_ROOT"
    log "$tag" "$log_file" "uv binary : $UV_BIN"
    log "$tag" "$log_file" "Lock file : $lock_file"
    log "$tag" "$log_file" "Done mark : $done_marker"

    # ------------------------------------------------------------------
    # Skip if already finished.
    # ------------------------------------------------------------------
    if [[ -f "$done_marker" ]]; then
        log "$tag" "$log_file" "Already completed (found $done_marker). Nothing to do."
        log "$tag" "$log_file" "Delete that file and re-run this script to retrain."
        exit 0
    fi

    # ------------------------------------------------------------------
    # Training retry loop.
    # ------------------------------------------------------------------
    cd "$SRC_DIR"

    local attempt=0
    local delay=60   # start: 1 min; doubles each retry up to 5 min cap

    while true; do
        attempt=$(( attempt + 1 ))
        log "$tag" "$log_file" "--- Attempt $attempt ---"

        # Run training; capture exit code even under set -e.
        set +e
        "$UV_BIN" run python train_kfold.py "experiment=${experiment}" \
            2>&1 | tee -a "$log_file"
        local exit_code="${PIPESTATUS[0]}"
        set -e

        if [[ "$exit_code" -eq 0 ]]; then
            log "$tag" "$log_file" "SUCCESS — training finished normally after $attempt attempt(s)."
            touch "$done_marker"
            log "$tag" "$log_file" "Done marker written: $done_marker"
            exit 0
        fi

        log "$tag" "$log_file" "FAILED (exit code $exit_code). Retrying in ${delay}s …"
        sleep "$delay"

        # Exponential backoff: 60 → 120 → 240 → 300 (hard cap).
        delay=$(( delay * 2 > 300 ? 300 : delay * 2 ))
    done
}
