#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/nvidia/miniconda3/envs/concept-trak/bin/python}"
HF_NAME="${HF_NAME:-YongYong}"
HF_REPO_ID="${HF_REPO_ID:-${HF_NAME}/concept-trak-abc-train-grad}"
REPO_TYPE="${REPO_TYPE:-dataset}"
TRAIN_GRAD_NAME="${TRAIN_GRAD_NAME:-attn2-dps-NFE10-norm-ddim-gs_7.5}"
PATH_IN_REPO="${PATH_IN_REPO:-$TRAIN_GRAD_NAME}"
GRAD_ROOT="${GRAD_ROOT:-$ROOT_DIR/experiments/abc/results/grads}"
MAX_WORKERS="${MAX_WORKERS:-8}"

mkdir -p "$GRAD_ROOT"

export HF_REPO_ID REPO_TYPE TRAIN_GRAD_NAME PATH_IN_REPO GRAD_ROOT MAX_WORKERS

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ["REPO_TYPE"]
path_in_repo = os.environ["PATH_IN_REPO"].strip("/")
grad_root = Path(os.environ["GRAD_ROOT"]).resolve()
max_workers = int(os.environ["MAX_WORKERS"])
token = os.environ.get("HF_TOKEN") or None

allow_patterns = [f"{path_in_repo}/*"] if path_in_repo else None
snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=str(grad_root),
    allow_patterns=allow_patterns,
    max_workers=max_workers,
    token=token,
)

target_dir = grad_root / path_in_repo if path_in_repo else grad_root
expected = [target_dir / f"train_grad-{idx}.npy" for idx in range(16)]
missing = [str(path) for path in expected if not path.exists()]
if missing:
    raise FileNotFoundError("Download finished but files are missing:\n" + "\n".join(missing))

print(f"Downloaded train gradients from https://huggingface.co/datasets/{repo_id}")
print(f"Local path: {target_dir}")
PY
