#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/toy"

SHAPE_IDX="${SHAPE_IDX:-0}"
COLOR_IDX="${COLOR_IDX:-1}"
TARGET_CONCEPT_DIM="${TARGET_CONCEPT_DIM:-0}"
TARGET_CONCEPT_IDX="${TARGET_CONCEPT_IDX:-$SHAPE_IDX}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
NFE="${NFE:-10}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
ETA="${ETA:-0.1}"

if [[ "${DDIM_INVERSION:-1}" == "1" ]]; then
  DDIM_FLAG="--ddim_inversion"
else
  DDIM_FLAG=""
fi

python "$EXP_DIR/test_grad.py" \
  --sample_idx "$SAMPLE_IDX" \
  --shape_idx "$SHAPE_IDX" \
  --color_idx "$COLOR_IDX" \
  --target_concept_dim "$TARGET_CONCEPT_DIM" \
  --target_concept_idx "$TARGET_CONCEPT_IDX" \
  --NFE "$NFE" \
  --eta "$ETA" \
  --guidance_scale "$GUIDANCE_SCALE" \
  $DDIM_FLAG
