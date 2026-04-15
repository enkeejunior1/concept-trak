#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/abc"

TASK_IDX="${TASK_IDX:-0}"
LAYER="${LAYER:-attn2}"
MODE="${MODE:-global}"
NFE="${NFE:-10}"
CONCEPT_GUIDANCE_SCALE="${CONCEPT_GUIDANCE_SCALE:-1.0}"
ETA="${ETA:-0.1}"
SD_MODEL_PATH="${SD_MODEL_PATH:-/home/yonghyun.park/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b}"
TI_STEPS="${TI_STEPS:-5000}"

if [[ "${DDIM_INVERSION:-1}" == "1" ]]; then
  DDIM_FLAG="--ddim_inversion"
else
  DDIM_FLAG=""
fi

case "$MODE" in
  global)
    python "$EXP_DIR/test_global_grad.py" \
      --task_idx "$TASK_IDX" \
      --layer "$LAYER" \
      --f slider \
      --NFE "$NFE" \
      --guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
      --sd_model_path "$SD_MODEL_PATH"
    ;;
  local_1)
    python "$EXP_DIR/test_local_grad.py" \
      --task_idx "$TASK_IDX" \
      --layer "$LAYER" \
      --f slider_local_1 \
      --NFE "$NFE" \
      --eta "$ETA" \
      --guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
      --sd_model_path "$SD_MODEL_PATH" \
      $DDIM_FLAG
    ;;
  local_2)
    python "$EXP_DIR/test_local_alt_grad.py" \
      --task_idx "$TASK_IDX" \
      --layer "$LAYER" \
      --f slider_local_2 \
      --NFE "$NFE" \
      --eta "$ETA" \
      --guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
      --sd_model_path "$SD_MODEL_PATH" \
      $DDIM_FLAG
    ;;
  local_seed)
    python "$EXP_DIR/test_local_seed_grad.py" \
      --task_idx "$TASK_IDX" \
      --layer "$LAYER" \
      --f slider_seed \
      --NFE "$NFE" \
      --eta "$ETA" \
      --guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
      --sd_model_path "$SD_MODEL_PATH"
    ;;
  local_ti)
    if [[ "${SKIP_TI:-0}" != "1" ]]; then
      python "$EXP_DIR/ti.py" \
        --task_idx "$TASK_IDX" \
        --sd_model_path "$SD_MODEL_PATH" \
        --max_train_steps "$TI_STEPS"
    fi

    python "$EXP_DIR/test_local_ti_grad.py" \
      --task_idx "$TASK_IDX" \
      --layer "$LAYER" \
      --f slider_ti \
      --NFE "$NFE" \
      --guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
      --sd_model_path "$SD_MODEL_PATH"
    ;;
  baseline_global)
    python "$EXP_DIR/baseline_global_grad.py" \
      --task_idx "$TASK_IDX" \
      --layer "$LAYER" \
      --f global \
      --NFE "$NFE" \
      --guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
      --sd_model_path "$SD_MODEL_PATH"
    ;;
  baseline_local)
    python "$EXP_DIR/baseline_local_grad.py" \
      --task_idx "$TASK_IDX" \
      --layer "$LAYER" \
      --f local \
      --NFE "$NFE" \
      --guidance_scale "$CONCEPT_GUIDANCE_SCALE" \
      --sd_model_path "$SD_MODEL_PATH"
    ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    echo "Use MODE=global, local_1, local_2, local_seed, local_ti, baseline_global, or baseline_local" >&2
    exit 1
    ;;
esac
