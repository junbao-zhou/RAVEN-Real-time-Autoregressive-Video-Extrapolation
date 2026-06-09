#!/bin/bash
set -exo pipefail

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-/usr/local/cuda}}"
CUDA_TARGET_INCLUDES="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/include)"
CUDA_TARGET_LIBS="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/lib)"
export CPATH="${CUDA_TARGET_INCLUDES#:}${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CUDA_TARGET_LIBS#:}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_TARGET_LIBS#:}${CONDA_PREFIX:+:$CONDA_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export MAX_JOBS=${MAX_JOBS:-32}
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;9.0a}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
export FLASH_ATTENTION_DISABLE_SM80="${FLASH_ATTENTION_DISABLE_SM80:-TRUE}"

"$PYTHON_BIN" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r tools/requirements.txt

if ! ls assets/flash_attn-2*.whl >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip wheel -v git+https://github.com/Dao-AILab/flash-attention.git@060c9188beec3a8b62b33a3bfa6d5d2d44975fab --no-build-isolation --no-deps --wheel-dir=./assets --no-cache-dir
fi
"$PYTHON_BIN" -m pip install assets/flash_attn-2*.whl

if [ "${INSTALL_FLASH_ATTN_3:-0}" = "1" ]; then
    if ! ls assets/flash_attn_3*.whl >/dev/null 2>&1; then
        "$PYTHON_BIN" -m pip wheel -v git+https://github.com/Dao-AILab/flash-attention.git@e2743ab5b3803bb672b16437ba98a3b1d4576c50#subdirectory=hopper --no-build-isolation --no-deps --wheel-dir=./assets --no-cache-dir
    fi
    "$PYTHON_BIN" -m pip install assets/flash_attn_3*.whl
fi

if [ "${INSTALL_MAGI_ATTENTION:-0}" = "1" ]; then
    if ! ls assets/magi_attention-*.whl >/dev/null 2>&1; then
        export MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY="${MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY:-90}"
        export MAGI_ATTENTION_PREBUILD_FFA_JOBS=$MAX_JOBS
        "$PYTHON_BIN" -m pip wheel -v git+https://github.com/SandAI-org/MagiAttention.git@e08bea8a051031978dbfcd069e2d876b36559bb9 --no-build-isolation --no-deps --wheel-dir=./assets --no-cache-dir
    fi
    "$PYTHON_BIN" -m pip install assets/magi_attention-*.whl
fi
