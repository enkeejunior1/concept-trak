#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/abc"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROMPT="${PROMPT:-Pikachu in the style of The Simpsons}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
TARGET_CONCEPTS="${TARGET_CONCEPTS:-Pikachu,The Simpsons}"
NEGATIVE_PROMPTS="${NEGATIVE_PROMPTS:-in the style of The Simpsons,Pikachu}"
SEED="${SEED:-0}"
LAYER="${LAYER:-attn2}"
F="${F:-dps}"
NUM_SPLIT="${NUM_SPLIT:-16}"
NFE="${NFE:-10}"
TRAIN_GUIDANCE_SCALE="${TRAIN_GUIDANCE_SCALE:-7.5}"
CONCEPT_GUIDANCE_SCALE="${CONCEPT_GUIDANCE_SCALE:-1.0}"
GEN_GUIDANCE_SCALE="${GEN_GUIDANCE_SCALE:-7.5}"
GEN_NUM_INFERENCE_STEPS="${GEN_NUM_INFERENCE_STEPS:-50}"
ETA="${ETA:-0.1}"
TOP_K="${TOP_K:-10}"
SD_MODEL_PATH="${SD_MODEL_PATH:-CompVis/stable-diffusion-v1-4}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/abc}"
GRAD_DIR="${GRAD_DIR:-$EXP_DIR/results/grads}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_DIR/results/qual}"
DTYPE="${DTYPE:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-256}"
PROJ_TYPE="${PROJ_TYPE:-random_mask}"
NORMALIZE="${NORMALIZE:-1}"
GPU_ID="${GPU_ID:-4}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

derive_negative_prompt() {
  local prompt="$1"
  local target="$2"
  local neg="${prompt//$target/}"
  while [[ "$neg" == *"  "* ]]; do
    neg="${neg//  / }"
  done
  trim "$neg"
}

run_one_target() {
  local target_concept="$1"
  local negative_prompt="$2"

  local cmd=(
    "$PYTHON_BIN" "$EXP_DIR/qual.py"
    --prompt "$PROMPT"
    --seed "$SEED"
    --layer "$LAYER"
    --f "$F"
    --num_split "$NUM_SPLIT"
    --NFE "$NFE"
    --dtype "$DTYPE"
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --proj_type "$PROJ_TYPE"
    --top_k "$TOP_K"
    --train_guidance_scale "$TRAIN_GUIDANCE_SCALE"
    --concept_guidance_scale "$CONCEPT_GUIDANCE_SCALE"
    --gen_guidance_scale "$GEN_GUIDANCE_SCALE"
    --gen_num_inference_steps "$GEN_NUM_INFERENCE_STEPS"
    --eta "$ETA"
    --sd_model_path "$SD_MODEL_PATH"
    --data_dir "$DATA_DIR"
    --grad_dir "$GRAD_DIR"
    --results_dir "$RESULTS_DIR"
  )

  if [[ -n "$target_concept" ]]; then
    cmd+=(--target_concept "$target_concept")
  fi

  if [[ "${DDIM_INVERSION:-1}" == "1" ]]; then
    cmd+=(--ddim_inversion)
  fi

  if [[ "${NORMALIZE:-0}" == "1" ]]; then
    cmd+=(--normalize)
  fi

  if [[ "${RENDER_LEASTK:-0}" == "1" ]]; then
    cmd+=(--render_leastk)
  fi

  if [[ "${FORCE_RECOMPUTE_CONCEPT_GRAD:-0}" == "1" ]]; then
    cmd+=(--force_recompute_concept_grad)
  fi

  if [[ -n "$negative_prompt" ]]; then
    cmd+=(--negative_prompt "$negative_prompt")
  fi

  if [[ -n "${TI_MODEL_PATH:-}" ]]; then
    cmd+=(--ti_model_path "$TI_MODEL_PATH")
    cmd+=(--ti_weight_name "${TI_WEIGHT_NAME:-new1.bin}")
  fi

  echo "Running qual: prompt='$PROMPT' target='$target_concept' negative='$negative_prompt'"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}"
}

if [[ -n "$TARGET_CONCEPTS" ]]; then
  IFS=',' read -r -a target_array <<< "$TARGET_CONCEPTS"
  neg_array=()
  if [[ -n "$NEGATIVE_PROMPTS" ]]; then
    IFS=',' read -r -a neg_array <<< "$NEGATIVE_PROMPTS"
  fi

  for idx in "${!target_array[@]}"; do
    target_concept="$(trim "${target_array[$idx]}")"
    if [[ -z "$target_concept" ]]; then
      continue
    fi

    negative_prompt=""
    if [[ "${neg_array[$idx]+set}" == "set" ]]; then
      negative_prompt="$(trim "${neg_array[$idx]}")"
    elif [[ "${#target_array[@]}" == "1" && -n "$NEGATIVE_PROMPT" ]]; then
      negative_prompt="$NEGATIVE_PROMPT"
    else
      negative_prompt="$(derive_negative_prompt "$PROMPT" "$target_concept")"
    fi

    run_one_target "$target_concept" "$negative_prompt"
  done
else
  run_one_target "" "$NEGATIVE_PROMPT"
fi
