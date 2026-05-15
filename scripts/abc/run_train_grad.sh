#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/abc"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_IDX="${TASK_IDX:-0}"
LAYER="${LAYER:-attn2}"
F="${F:-dps}"
NUM_SPLIT="${NUM_SPLIT:-16}"
NFE="${NFE:-10}"
TRAIN_GUIDANCE_SCALE="${TRAIN_GUIDANCE_SCALE:-7.5}"
SD_MODEL_PATH="${SD_MODEL_PATH:-CompVis/stable-diffusion-v1-4}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/abc}"
GRAD_DIR="${GRAD_DIR:-$EXP_DIR/results/grads}"
DTYPE="${DTYPE:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PROJ_TYPE="${PROJ_TYPE:-random_mask}"
DDIM_INVERSION="${DDIM_INVERSION:-1}"
NORMALIZE="${NORMALIZE:-1}"
RUN_TASK_GRAD="${RUN_TASK_GRAD:-0}"

if [[ "$DDIM_INVERSION" == "1" ]]; then
  DDIM_FLAG="--ddim_inversion"
else
  DDIM_FLAG=""
fi

if [[ "$NORMALIZE" == "1" ]]; then
  NORMALIZE_FLAG="--normalize"
else
  NORMALIZE_FLAG=""
fi

for split_idx in $(seq 0 $((NUM_SPLIT - 1))); do
  "$PYTHON_BIN" "$EXP_DIR/train_grad.py" \
    --split_idx "$split_idx" \
    --num_split "$NUM_SPLIT" \
    --layer "$LAYER" \
    --f "$F" \
    --NFE "$NFE" \
    --dtype "$DTYPE" \
    --batch_size "$BATCH_SIZE" \
    --guidance_scale "$TRAIN_GUIDANCE_SCALE" \
    --proj_type "$PROJ_TYPE" \
    --sd_model_path "$SD_MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$GRAD_DIR" \
    $DDIM_FLAG \
    $NORMALIZE_FLAG
done

if [[ "$RUN_TASK_GRAD" == "1" ]]; then
  "$PYTHON_BIN" "$EXP_DIR/task_grad.py" \
    --task_idx "$TASK_IDX" \
    --layer "$LAYER" \
    --f "$F" \
    --NFE "$NFE" \
    --dtype "$DTYPE" \
    --batch_size "$BATCH_SIZE" \
    --guidance_scale "$TRAIN_GUIDANCE_SCALE" \
    --proj_type "$PROJ_TYPE" \
    --sd_model_path "$SD_MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$GRAD_DIR" \
    $DDIM_FLAG \
    $NORMALIZE_FLAG
fi
