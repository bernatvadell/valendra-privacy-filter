#!/usr/bin/env bash
set -Eeuo pipefail

# Start a new optimizer/scheduler on the isolated mixed dataset from saved weights.
ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$ROOT"
PYTHON="${PRIVACY_FILTER_PYTHON:-$ROOT/.venv/bin/python}"
CHECKPOINT="${1:?Pass the complete prior checkpoint directory}"
RUN_DIR="experiments/runs/openpii-nemotron-causal-warmstart"
LOG="experiments/runs/mixed-warmstart.log"
export PYTHONUNBUFFERED=1

test -x "$PYTHON"
test -f "$CHECKPOINT/model.safetensors"
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/trainer_state.json"
test -f "$CHECKPOINT/../run_identity.json"
test ! -e "$RUN_DIR/run_identity.json" || { printf 'Run exists; use same-phase checkpoint resume.\n'; exit 1; }
mkdir -p experiments/runs
exec >>"$LOG" 2>&1
printf '\n[%s] Starting mixed-data BIOES fine-tuning from saved weights\n' "$(date -u +%FT%TZ)"
"$PYTHON" -c 'import shutil,torch; free=shutil.disk_usage(".").free; print("free_disk_bytes=",free,"cuda=",torch.cuda.is_available(),"bf16=",torch.cuda.is_bf16_supported()); assert free >= 22 * 1024**3; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
"$PYTHON" -c 'from importlib.metadata import distributions; print("\n".join(sorted("{}=={}".format(d.metadata["Name"], d.version) for d in distributions() if d.metadata.get("Name"))))' > experiments/runs/mixed-warmstart-environment.txt
"$PYTHON" -m valendra_privacy_filter.cli train --config configs/train-openpii-nemotron-warmstart.yaml \
  --initialize-from-checkpoint "$CHECKPOINT"
printf '[%s] Mixed-data BIOES training process finished\n' "$(date -u +%FT%TZ)"
