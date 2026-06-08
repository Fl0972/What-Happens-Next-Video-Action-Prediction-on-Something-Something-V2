#!/usr/bin/env bash
# Auto-restart wrapper for tsm_ultra_resnet50_rotating training.
#
# Retries on any non-zero exit (crash, OOM, etc.).
# Survives machine reboots when registered as a @reboot cron job:
#
#   crontab -e
#   @reboot /path/to/scripts/train_rotating.sh
#
# The script is idempotent: if training has already finished (exit 0 from a
# previous run), it exits immediately without re-training.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_ROOT/src"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/train_rotating.log"
DONE_MARKER="$LOG_DIR/.train_rotating_done"

MAX_RETRIES=9999
RETRY_DELAY=60   # seconds between restart attempts

mkdir -p "$LOG_DIR"

# Guard: if previous run succeeded, skip entirely.
if [[ -f "$DONE_MARKER" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training already completed. Remove $DONE_MARKER to re-run." | tee -a "$LOG_FILE"
    exit 0
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== train_rotating.sh started (PID $$) ==="
log "Repo: $REPO_ROOT"
log "Max retries: $MAX_RETRIES, retry delay: ${RETRY_DELAY}s"

cd "$SRC_DIR"

for attempt in $(seq 1 $MAX_RETRIES); do
    log "--- Attempt $attempt/$MAX_RETRIES ---"

    set +e
    uv run python train_kfold.py \
        experiment=tsm_ultra_resnet50_rotating \
        2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [[ $EXIT_CODE -eq 0 ]]; then
        log "Training completed successfully (attempt $attempt)."
        touch "$DONE_MARKER"
        exit 0
    fi

    log "Training exited with code $EXIT_CODE. Retrying in ${RETRY_DELAY}s…"
    sleep "$RETRY_DELAY"
done

log "ERROR: Reached max retries ($MAX_RETRIES). Giving up."
exit 1
