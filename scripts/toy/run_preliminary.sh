#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/toy"

CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-25}"
MODEL_EPOCHS="${MODEL_EPOCHS:-100}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-512}"
MODEL_BATCH_SIZE="${MODEL_BATCH_SIZE:-128}"

python "$EXP_DIR/preliminary/generate_data.py" --base_dir "$EXP_DIR"
python "$EXP_DIR/preliminary/train_classifier.py" \
  --base_dir "$EXP_DIR" \
  --output_path "$EXP_DIR/weights/classifier.bin" \
  --batch_size "$CLASSIFIER_BATCH_SIZE" \
  --num_epochs "$CLASSIFIER_EPOCHS"
python "$EXP_DIR/preliminary/train_model.py" \
  --base_dir "$EXP_DIR" \
  --classifier_path "$EXP_DIR/weights/classifier.bin" \
  --output_path "$EXP_DIR/weights/model.bin" \
  --batch_size "$MODEL_BATCH_SIZE" \
  --num_epochs "$MODEL_EPOCHS"
