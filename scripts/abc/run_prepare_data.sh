#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/abc"

SOURCE_DIR="${SOURCE_DIR:?Set SOURCE_DIR to the original AbC benchmark asset directory}"
COPY_MODE="${COPY_MODE:-symlink}"

python "$EXP_DIR/preliminary/prepare_data.py" \
  --source_dir "$SOURCE_DIR" \
  --dest_dir "$EXP_DIR" \
  --mode "$COPY_MODE"
