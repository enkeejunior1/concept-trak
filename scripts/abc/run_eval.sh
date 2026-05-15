#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/abc"

TASK_IDX="${TASK_IDX:-0}"
LAYER="${LAYER:-attn2}"
F="${F:-dps}"
NUM_SPLIT="${NUM_SPLIT:-8}"
NFE="${NFE:-10}"
TRAIN_GUIDANCE_SCALE="${TRAIN_GUIDANCE_SCALE:-7.5}"
CONCEPT_GUIDANCE_SCALE="${CONCEPT_GUIDANCE_SCALE:-1.0}"
ETA="${ETA:-0.1}"
TOP_K="${TOP_K:-10}"

if [[ "${DDIM_INVERSION:-1}" == "1" ]]; then
  DDIM_FLAG="--ddim_inversion"
else
  DDIM_FLAG=""
fi

python "$EXP_DIR/eval.py" \
  --task_idx "$TASK_IDX" \
  --layer "$LAYER" \
  --num_split "$NUM_SPLIT" \
  --NFE "$NFE" \
  --f "$F" \
  --train_guidance_scale "$TRAIN_GUIDANCE_SCALE" \
  --concept_guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
  --eta "$ETA" \
  --top_k "$TOP_K" \
  $DDIM_FLAG
