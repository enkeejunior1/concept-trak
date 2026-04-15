#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/abc"

TASK_IDX="${TASK_IDX:-0}"
LAYER="${LAYER:-attn2}"
F="${F:-dpsv1}"
MODE="${MODE:-global}"
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

case "$MODE" in
  global) CONCEPT_F="slider" ;;
  local_1) CONCEPT_F="slider_local_1" ;;
  local_2) CONCEPT_F="slider_local_2" ;;
  local_seed) CONCEPT_F="slider_seed" ;;
  local_ti) CONCEPT_F="slider_ti" ;;
  baseline_global) CONCEPT_F="global" ;;
  baseline_local) CONCEPT_F="local" ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    exit 1
    ;;
esac

EXTRA_ARGS=()
if [[ "$CONCEPT_F" == "slider_local_1" || "$CONCEPT_F" == "slider_local_2" || "$CONCEPT_F" == "slider_seed" ]]; then
  EXTRA_ARGS+=(--eta "$ETA")
fi

python "$EXP_DIR/influence.py" \
  --task_idx "$TASK_IDX" \
  --layer "$LAYER" \
  --num_split "$NUM_SPLIT" \
  --NFE "$NFE" \
  --f "$F" \
  --concept_f "$CONCEPT_F" \
  --train_guidance_scale "$TRAIN_GUIDANCE_SCALE" \
  --concept_guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
  "${EXTRA_ARGS[@]}" \
  $DDIM_FLAG

python "$EXP_DIR/eval.py" \
  --task_idx "$TASK_IDX" \
  --layer "$LAYER" \
  --NFE "$NFE" \
  --f "$F" \
  --concept_f "$CONCEPT_F" \
  --train_guidance_scale "$TRAIN_GUIDANCE_SCALE" \
  --concept_guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
  --top_k "$TOP_K" \
  "${EXTRA_ARGS[@]}" \
  $DDIM_FLAG
