#!/bin/bash
set -exo pipefail

eval "$(conda shell.bash hook)"
export CONDA_ENV="${CONDA_ENV:-base}"
conda activate "$CONDA_ENV"

PREFIX="${PREFIX:-$HOME}/.cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PREFIX/uv}"

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
CUDA_TARGET_INCLUDES="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/include)"
CUDA_TARGET_LIBS="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/lib)"
export CPATH="${CUDA_TARGET_INCLUDES#:}${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CUDA_TARGET_LIBS#:}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_TARGET_LIBS#:}:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export MAX_JOBS=${MAX_JOBS:-32}
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

[ ! -d "third_party/vbench/venv" ] && uv venv "third_party/vbench/venv" --python="$(command -v python)"
uv pip sync third_party/vbench/requirements.lock \
    --python "third_party/vbench/venv/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    --link-mode=hardlink
source "third_party/vbench/venv/bin/activate"

pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@a25898a09d6ee232767647e92c6177fb1c642369 --no-build-isolation

python - <<'PY'
from vbench.utils import init_submodules

init_submodules([
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
], local=True, read_frame=False)
PY

mkdir -p "$TORCH_HOME/hub/checkpoints"
ln -sfn "$VBENCH_CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth" "$TORCH_HOME/hub/checkpoints/dino_vitbase16_pretrain.pth"
