#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/abc"

TASK_IDX="${TASK_IDX:-0}"
LAYER="${LAYER:-attn2}"
F="${F:-dpsv1}"
NUM_SPLIT="${NUM_SPLIT:-8}"
NFE="${NFE:-10}"
TRAIN_GUIDANCE_SCALE="${TRAIN_GUIDANCE_SCALE:-7.5}"
SD_MODEL_PATH="${SD_MODEL_PATH:-/home/yonghyun.park/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b}"

if [[ "${DDIM_INVERSION:-1}" == "1" ]]; then
  DDIM_FLAG="--ddim_inversion"
else
  DDIM_FLAG=""
fi

if [[ "$F" == "dasv1" ]]; then
  for split_idx in $(seq 0 $((NUM_SPLIT - 1))); do
    python "$EXP_DIR/train_loss.py" \
      --split_idx "$split_idx" \
      --num_split "$NUM_SPLIT" \
      --NFE "$NFE" \
      --guidance_scale "$TRAIN_GUIDANCE_SCALE" \
      --sd_model_path "$SD_MODEL_PATH" \
      $DDIM_FLAG
  done

  python "$EXP_DIR/task_loss.py" \
    --task_idx "$TASK_IDX" \
    --layer "$LAYER" \
    --f "$F" \
    --NFE "$NFE" \
    --guidance_scale "$TRAIN_GUIDANCE_SCALE" \
    --sd_model_path "$SD_MODEL_PATH" \
    $DDIM_FLAG
fi

for split_idx in $(seq 0 $((NUM_SPLIT - 1))); do
  python "$EXP_DIR/train_grad.py" \
    --split_idx "$split_idx" \
    --num_split "$NUM_SPLIT" \
    --layer "$LAYER" \
    --f "$F" \
    --NFE "$NFE" \
    --guidance_scale "$TRAIN_GUIDANCE_SCALE" \
    --sd_model_path "$SD_MODEL_PATH" \
    $DDIM_FLAG
done

python "$EXP_DIR/task_grad.py" \
  --task_idx "$TASK_IDX" \
  --layer "$LAYER" \
  --f "$F" \
  --NFE "$NFE" \
  --guidance_scale "$TRAIN_GUIDANCE_SCALE" \
  --sd_model_path "$SD_MODEL_PATH" \
  $DDIM_FLAG
