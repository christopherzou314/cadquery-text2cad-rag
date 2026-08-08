#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CADQUERY_PYTHON:-python}"
RESULTS="$ROOT/experiments/cadquery_benchmark_v1/results/full_b2"
LOG="$RESULTS/experiment.log"

mkdir -p "$RESULTS"
printf '\n=== persistent launch %s ===\n' "$(date --iso-8601=seconds)" >> "$LOG"
cd "$ROOT"
exec "$PYTHON" scripts/run_benchmark.py \
  --cases experiments/cadquery_benchmark_v1/full_cases.json \
  --experiment-dir experiments/cadquery_benchmark_v1/results/full_b2 \
  --max-repairs 2 \
  --resume >> "$LOG" 2>&1
