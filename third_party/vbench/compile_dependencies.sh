#!/bin/bash
set -exo pipefail

eval "$(conda shell.bash hook)"
export CONDA_ENV="${CONDA_ENV:-base}"
conda activate "$CONDA_ENV"

uv pip compile third_party/vbench/requirements.txt \
    -o third_party/vbench/requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
