#!/usr/bin/env bash
set -Eeuo pipefail

# Resume the same dataset, model and optimizer; apply the approved minimum-epoch policy.
ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$ROOT"
PYTHON="${PRIVACY_FILTER_PYTHON:-$ROOT/.venv/bin/python}"
CHECKPOINT="${1:-experiments/runs/causal/checkpoint-8000}"
POLICY="experiments/runs/causal/continuation_policy.json"
LOG="experiments/runs/full-causal-continuation.log"
export PYTHONUNBUFFERED=1

test -x "$PYTHON"
test -f "$CHECKPOINT/model.safetensors"
test -f "$CHECKPOINT/optimizer.pt"
test -f "$CHECKPOINT/scheduler.pt"
test -f "$CHECKPOINT/rng_state.pth"
test -f "$CHECKPOINT/trainer_state.json"
mkdir -p experiments/runs
exec >>"$LOG" 2>&1
printf '\n[%s] Resuming full causal BIOES training from %s\n' "$(date -u +%FT%TZ)" "$CHECKPOINT"

ARGS=(train --config configs/train-causal.yaml --resume-from-checkpoint "$CHECKPOINT")
if [[ ! -f "$POLICY" ]]; then
  ARGS+=(--min-early-stopping-epoch 1.0)
fi
"$PYTHON" -m valendra_privacy_filter.cli "${ARGS[@]}"
printf '[%s] Continued causal BIOES training process finished\n' "$(date -u +%FT%TZ)"
