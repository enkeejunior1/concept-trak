#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/abc"

TASK_IDX="${TASK_IDX:-0}"
LAYER="${LAYER:-attn2}"
NFE="${NFE:-10}"
CONCEPT_GUIDANCE_SCALE="${CONCEPT_GUIDANCE_SCALE:-1.0}"
ETA="${ETA:-0.1}"
SD_MODEL_PATH="${SD_MODEL_PATH:-CompVis/stable-diffusion-v1-4}"
PROJ_TYPE="${PROJ_TYPE:-random_mask}"

python "$EXP_DIR/test_grad.py" \
  --task_idx "$TASK_IDX" \
  --layer "$LAYER" \
  --NFE "$NFE" \
  --eta "$ETA" \
  --guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
  --proj_type "$PROJ_TYPE" \
  --sd_model_path "$SD_MODEL_PATH"
