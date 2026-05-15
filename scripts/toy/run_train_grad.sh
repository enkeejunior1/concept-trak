#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/toy"

NUM_SPLIT="${NUM_SPLIT:-8}"
F="${F:-dps}"
NFE="${NFE:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

for split_idx in $(seq 0 $((NUM_SPLIT - 1))); do
  python "$EXP_DIR/train_grad.py" \
    --split_idx "$split_idx" \
    --num_split "$NUM_SPLIT" \
    --f "$F" \
    --NFE "$NFE" \
    --batch_size "$BATCH_SIZE" \
    --guidance_scale "$GUIDANCE_SCALE" \
    $EXTRA_FLAGS
done
