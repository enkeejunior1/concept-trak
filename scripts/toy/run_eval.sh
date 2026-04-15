#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/toy"

MODE="${MODE:-global}"
F="${F:-dpsv1}"
NUM_SPLIT="${NUM_SPLIT:-8}"
SHAPE_IDX="${SHAPE_IDX:-0}"
COLOR_IDX="${COLOR_IDX:-9}"
TARGET_CONCEPT_DIM="${TARGET_CONCEPT_DIM:-0}"
TARGET_CONCEPT_IDX="${TARGET_CONCEPT_IDX:-$SHAPE_IDX}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
NFE="${NFE:-10}"
TRAIN_GUIDANCE_SCALE="${TRAIN_GUIDANCE_SCALE:-7.5}"
CONCEPT_GUIDANCE_SCALE="${CONCEPT_GUIDANCE_SCALE:-7.5}"
ETA="${ETA:-0.1}"
TOP_K="${TOP_K:-10}"

if [[ "${DDIM_INVERSION:-1}" == "1" ]]; then
  DDIM_FLAG="--ddim_inversion"
else
  DDIM_FLAG=""
fi

if [[ "$MODE" == "global" ]]; then
  CONCEPT_F="slider"
  NUM_SAMPLES=0
  EXTRA_ARGS=()
elif [[ "$MODE" == "local" ]]; then
  CONCEPT_F="slider_local_1"
  NUM_SAMPLES=1
  EXTRA_ARGS=(--sample_idx "$SAMPLE_IDX" --eta "$ETA")
else
  echo "Unsupported MODE: $MODE" >&2
  echo "Use MODE=global or MODE=local" >&2
  exit 1
fi

python "$EXP_DIR/influence.py" \
  --num_samples "$NUM_SAMPLES" \
  --shape_idx "$SHAPE_IDX" \
  --color_idx "$COLOR_IDX" \
  --target_concept_dim "$TARGET_CONCEPT_DIM" \
  --target_concept_idx "$TARGET_CONCEPT_IDX" \
  --num_split "$NUM_SPLIT" \
  --NFE "$NFE" \
  --f "$F" \
  --concept_f "$CONCEPT_F" \
  --train_gs "$TRAIN_GUIDANCE_SCALE" \
  --concept_gs "$CONCEPT_GUIDANCE_SCALE" \
  "${EXTRA_ARGS[@]}" \
  $DDIM_FLAG

python "$EXP_DIR/eval.py" \
  --num_samples "$NUM_SAMPLES" \
  --shape_idx "$SHAPE_IDX" \
  --color_idx "$COLOR_IDX" \
  --target_concept_dim "$TARGET_CONCEPT_DIM" \
  --target_concept_idx "$TARGET_CONCEPT_IDX" \
  --NFE "$NFE" \
  --f "$F" \
  --concept_f "$CONCEPT_F" \
  --train_gs "$TRAIN_GUIDANCE_SCALE" \
  --concept_gs "$CONCEPT_GUIDANCE_SCALE" \
  --top_k "$TOP_K" \
  "${EXTRA_ARGS[@]}" \
  $DDIM_FLAG
