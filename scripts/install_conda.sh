#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-concept-trak}"

cd "$ROOT_DIR"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  conda env create -f environment.yml
fi

echo
echo "Environment ready."
echo "Run: conda activate $ENV_NAME"
