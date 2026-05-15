#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/toy"

F="${F:-dps}"
NUM_SPLIT="${NUM_SPLIT:-8}"
SHAPE_IDX="${SHAPE_IDX:-0}"
COLOR_IDX="${COLOR_IDX:-1}"
TARGET_CONCEPT_DIM="${TARGET_CONCEPT_DIM:-0}"
TARGET_CONCEPT_IDX="${TARGET_CONCEPT_IDX:-$SHAPE_IDX}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
NFE="${NFE:-10}"
TRAIN_GUIDANCE_SCALE="${TRAIN_GUIDANCE_SCALE:-7.5}"
TEST_GUIDANCE_SCALE="${TEST_GUIDANCE_SCALE:-7.5}"
ETA="${ETA:-0.1}"
TOP_K="${TOP_K:-10}"

if [[ "${DDIM_INVERSION:-1}" == "1" ]]; then
  DDIM_FLAG="--ddim_inversion"
else
  DDIM_FLAG=""
fi

python "$EXP_DIR/eval.py" \
  --sample_idx "$SAMPLE_IDX" \
  --shape_idx "$SHAPE_IDX" \
  --color_idx "$COLOR_IDX" \
  --target_concept_dim "$TARGET_CONCEPT_DIM" \
  --target_concept_idx "$TARGET_CONCEPT_IDX" \
  --num_split "$NUM_SPLIT" \
  --NFE "$NFE" \
  --f "$F" \
  --train_guidance_scale "$TRAIN_GUIDANCE_SCALE" \
  --test_guidance_scale "$TEST_GUIDANCE_SCALE" \
  --eta "$ETA" \
  --top_k "$TOP_K" \
  $DDIM_FLAG
