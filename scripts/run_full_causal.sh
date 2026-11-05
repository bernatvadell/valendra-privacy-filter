#!/usr/bin/env bash
set -Eeuo pipefail

# Run from a dedicated checkout. Override PRIVACY_FILTER_PYTHON if needed.
ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$ROOT"
PYTHON="${PRIVACY_FILTER_PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1
DATA_DIR="data/processed"
SYNTHETIC_DIR="data/raw/synthetic"
RUN_DIR="experiments/runs/causal"
LOG="experiments/runs/full-causal.log"

mkdir -p experiments/runs
exec >>"$LOG" 2>&1
printf '\n[%s] Starting full causal BIOES preparation and training\n' "$(date -u +%FT%TZ)"
test -x "$PYTHON"
test ! -e "$RUN_DIR/run_identity.json" || { printf 'Run already exists; resume it explicitly instead.\n'; exit 1; }
test ! -e "$DATA_DIR/manifest.json" || { printf 'Processed dataset already exists; choose a new directory.\n'; exit 1; }

"$PYTHON" -c 'import shutil,torch; free=shutil.disk_usage(".").free; print("free_disk_bytes=",free,"cuda=",torch.cuda.is_available(),"bf16=",torch.cuda.is_bf16_supported()); assert free >= 30 * 1024**3; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
"$PYTHON" -m pip freeze > experiments/runs/full-causal-environment.txt
"$PYTHON" -m valendra_privacy_filter.cli generate-synthetic --output-dir "$SYNTHETIC_DIR" --seed 42 --per-split 20000
"$PYTHON" -m valendra_privacy_filter.cli prepare-dataset --synthetic-dir "$SYNTHETIC_DIR" \
  --output-dir "$DATA_DIR" --max-length 512 --overlap 64 --taxonomy configs/taxonomy.yaml --seed 42
printf '[%s] Dataset complete; starting BF16 training\n' "$(date -u +%FT%TZ)"
"$PYTHON" -m valendra_privacy_filter.cli train --config configs/train-causal.yaml
printf '[%s] Full causal BIOES training process finished\n' "$(date -u +%FT%TZ)"
