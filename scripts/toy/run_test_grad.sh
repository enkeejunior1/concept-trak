#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/toy"

MODE="${MODE:-global}"
SHAPE_IDX="${SHAPE_IDX:-0}"
COLOR_IDX="${COLOR_IDX:-9}"
TARGET_CONCEPT_DIM="${TARGET_CONCEPT_DIM:-0}"
TARGET_CONCEPT_IDX="${TARGET_CONCEPT_IDX:-$SHAPE_IDX}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
NFE="${NFE:-10}"
CONCEPT_GUIDANCE_SCALE="${CONCEPT_GUIDANCE_SCALE:-7.5}"
ETA="${ETA:-0.1}"

if [[ "${DDIM_INVERSION:-1}" == "1" ]]; then
  DDIM_FLAG="--ddim_inversion"
else
  DDIM_FLAG=""
fi

case "$MODE" in
  global)
    python "$EXP_DIR/test_global_grad.py" \
      --shape_idx "$SHAPE_IDX" \
      --color_idx "$COLOR_IDX" \
      --target_concept_dim "$TARGET_CONCEPT_DIM" \
      --target_concept_idx "$TARGET_CONCEPT_IDX" \
      --NFE "$NFE" \
      --guidance_scale "$CONCEPT_GUIDANCE_SCALE"
    ;;
  local)
    python "$EXP_DIR/generate_samples.py" \
      --base_dir "$EXP_DIR" \
      --model_path "$EXP_DIR/weights/model.bin" \
      --classifier_path "$EXP_DIR/weights/classifier.bin"

    python "$EXP_DIR/test_local_grad.py" \
      --sample_idx "$SAMPLE_IDX" \
      --shape_idx "$SHAPE_IDX" \
      --color_idx "$COLOR_IDX" \
      --target_concept_dim "$TARGET_CONCEPT_DIM" \
      --target_concept_idx "$TARGET_CONCEPT_IDX" \
      --num_samples 1 \
      --NFE "$NFE" \
      --eta "$ETA" \
      --guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
      $DDIM_FLAG
    ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    echo "Use MODE=global or MODE=local" >&2
    exit 1
    ;;
esac
