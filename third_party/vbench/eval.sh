#!/bin/bash
set -exo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <videos_path>" >&2
    exit 1
fi

SOURCE_DIR_INPUT="$1"
if [ ! -d "$SOURCE_DIR_INPUT" ]; then
    echo "Input directory not found: $SOURCE_DIR_INPUT" >&2
    exit 1
fi
SOURCE_DIR="$(cd "$SOURCE_DIR_INPUT" && pwd -P)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-/usr/local/cuda}}"
CUDA_TARGET_INCLUDES="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/include)"
CUDA_TARGET_LIBS="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/lib)"
export CPATH="${CUDA_TARGET_INCLUDES#:}${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CUDA_TARGET_LIBS#:}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_TARGET_LIBS#:}${CONDA_PREFIX:+:$CONDA_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PROMPT_FILE="$REPO_ROOT/assets/vbench_all_dimension.txt"
VBENCH_SAMPLES_PER_PROMPT="${VBENCH_SAMPLES_PER_PROMPT:-5}"
STATIC_FILTER_SAMPLES_PER_PROMPT="${STATIC_FILTER_SAMPLES_PER_PROMPT:-25}"
SYNC_POLL_SECONDS="${SYNC_POLL_SECONDS:-5}"
SYNC_TIMEOUT_SECONDS="${SYNC_TIMEOUT_SECONDS:-0}"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Prompt file not found: $PROMPT_FILE" >&2
    exit 1
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    DETECTED_GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    GPU_IDS=()
    for ((i = 0; i < DETECTED_GPU_COUNT; i++)); do
        GPU_IDS+=("$i")
    done
fi

VISIBLE_GPU_COUNT="${#GPU_IDS[@]}"
if [ "$VISIBLE_GPU_COUNT" -le 0 ]; then
    echo "Detected invalid GPU count: $VISIBLE_GPU_COUNT" >&2
    exit 1
fi

N="${N:-$VISIBLE_GPU_COUNT}"
STATIC_FILTER_N="${STATIC_FILTER_N:-$VISIBLE_GPU_COUNT}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
LOCAL_ADDR="${LOCAL_ADDR:-$(python -c "import socket; print(socket.gethostbyname('$(hostname)'))")}"

if [ -z "${MASTER_PORT:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        MASTER_PORT="$(expr 10000 + "$SLURM_JOB_ID" % 20000)"
    else
        MASTER_PORT=7890
    fi
fi

for value_name in VBENCH_SAMPLES_PER_PROMPT STATIC_FILTER_SAMPLES_PER_PROMPT SYNC_POLL_SECONDS N STATIC_FILTER_N NNODES MASTER_PORT; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer: ${value}" >&2
        exit 1
    fi
done

if ! [[ "$SYNC_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "SYNC_TIMEOUT_SECONDS must be a non-negative integer: ${SYNC_TIMEOUT_SECONDS}" >&2
    exit 1
fi

if [ "$N" -gt "$VISIBLE_GPU_COUNT" ]; then
    echo "Requested evaluation GPU count exceeds visible GPU count: $N > $VISIBLE_GPU_COUNT" >&2
    exit 1
fi
NPROC_PER_NODE="$N"

if [ "$STATIC_FILTER_N" -gt "$VISIBLE_GPU_COUNT" ]; then
    echo "Requested static filter GPU count exceeds visible GPU count: $STATIC_FILTER_N > $VISIBLE_GPU_COUNT" >&2
    exit 1
fi
MAX_STATIC_FILTER_WORKERS="$((NNODES * STATIC_FILTER_N))"

NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-}}"
if [ "$NNODES" -eq 1 ]; then
    NODE_RANK="${NODE_RANK:-0}"
    MASTER_ADDR="${MASTER_ADDR:-localhost}"
    RDZV_BACKEND="${RDZV_BACKEND:-static}"
else
    if [ -z "$NODE_RANK" ]; then
        echo "NODE_RANK must be set for multi-node runs outside SLURM" >&2
        exit 1
    fi
    if [ -z "${MASTER_ADDR:-}" ] && [ -n "${SLURM_JOB_NODELIST:-}" ]; then
        if python -c "import hostlist" >/dev/null 2>&1; then
            MASTER_ADDR="$(python -c "import hostlist; print(hostlist.expand_hostlist('$SLURM_JOB_NODELIST')[0])")"
        elif command -v scontrol >/dev/null 2>&1; then
            MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
        fi
    fi
    if [ -z "${MASTER_ADDR:-}" ]; then
        echo "MASTER_ADDR must be set for multi-node runs outside SLURM" >&2
        exit 1
    fi
    RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
fi

if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]]; then
    echo "NODE_RANK must be a non-negative integer: ${NODE_RANK}" >&2
    exit 1
fi
if [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "Node rank must be smaller than node count: $NODE_RANK >= $NNODES" >&2
    exit 1
fi

STATIC_FILTER_GLOBAL_N="${STATIC_FILTER_GLOBAL_N:-$((NNODES * STATIC_FILTER_N))}"
STATIC_FILTER_NODE_OFFSET="${STATIC_FILTER_NODE_OFFSET:-$((NODE_RANK * STATIC_FILTER_N))}"

if ! [[ "$STATIC_FILTER_GLOBAL_N" =~ ^[1-9][0-9]*$ ]]; then
    echo "STATIC_FILTER_GLOBAL_N must be a positive integer: ${STATIC_FILTER_GLOBAL_N}" >&2
    exit 1
fi
if ! [[ "$STATIC_FILTER_NODE_OFFSET" =~ ^[0-9]+$ ]]; then
    echo "STATIC_FILTER_NODE_OFFSET must be a non-negative integer: ${STATIC_FILTER_NODE_OFFSET}" >&2
    exit 1
fi
if [ "$STATIC_FILTER_GLOBAL_N" -gt "$MAX_STATIC_FILTER_WORKERS" ]; then
    echo "Total static filter shard count exceeds total launched static filter workers: $STATIC_FILTER_GLOBAL_N > $MAX_STATIC_FILTER_WORKERS" >&2
    exit 1
fi

TOTAL_CPUS="$(nproc --all)"
OMP_THREADS="$((TOTAL_CPUS / NPROC_PER_NODE))"
if [ "$OMP_THREADS" -le 0 ]; then
    OMP_THREADS=1
fi

export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=31
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="$OMP_THREADS"
export TOKENIZERS_PARALLELISM=false

RESULT_DIR="${SOURCE_DIR}_filtered"
DEST_DIR="${RESULT_DIR}/filtered_videos"
SHARD_ROOT="${RESULT_DIR}/static_filter_shards"
RUN_ROOT="${RESULT_DIR}/multi_eval_sync"
SOURCE_DIR_REAL="$(realpath "$SOURCE_DIR")"

mkdir -p "$DEST_DIR" "$SHARD_ROOT" "${RESULT_DIR}/evaluation_results"

if [ "$NODE_RANK" -eq 0 ]; then
    if [ -e "$RUN_ROOT" ]; then
        rm -rf "$RUN_ROOT"
    fi
    mkdir -p "$RUN_ROOT"
    printf '%s\n' \
        "source_dir=${SOURCE_DIR_REAL}" \
        "prompt_file=${PROMPT_FILE}" \
        "vbench_samples_per_prompt=${VBENCH_SAMPLES_PER_PROMPT}" \
        "static_filter_samples_per_prompt=${STATIC_FILTER_SAMPLES_PER_PROMPT}" \
        "static_filter_global_n=${STATIC_FILTER_GLOBAL_N}" \
        "nnodes=${NNODES}" \
        "eval_gpus_per_node=${NPROC_PER_NODE}" \
        "static_filter_gpus_per_node=${STATIC_FILTER_N}" \
        > "${RUN_ROOT}/config"

    python - <<'PY' "$SOURCE_DIR" "$PROMPT_FILE" "$SHARD_ROOT" "$STATIC_FILTER_GLOBAL_N" "$STATIC_FILTER_SAMPLES_PER_PROMPT"
import pathlib
import sys

source_dir = pathlib.Path(sys.argv[1])
prompt_file = pathlib.Path(sys.argv[2])
shard_root = pathlib.Path(sys.argv[3])
shard_count = int(sys.argv[4])
samples_per_prompt = int(sys.argv[5])

prompts = [line.strip() for line in prompt_file.read_text().splitlines()[:75] if line.strip()]

for shard_index in range(shard_count):
    shard_input_dir = shard_root / f"inputs_{shard_index}"
    shard_input_dir.mkdir(parents=True, exist_ok=True)
    for existing_file in shard_input_dir.glob("*.mp4"):
        existing_file.unlink()

for prompt_index, prompt in enumerate(prompts):
    shard_index = prompt_index % shard_count
    shard_input_dir = shard_root / f"inputs_{shard_index}"
    for video_index in range(samples_per_prompt):
        source_file = source_dir / f"{prompt}-{video_index}.mp4"
        if source_file.exists():
            target_file = shard_input_dir / source_file.name
            if target_file.exists() or target_file.is_symlink():
                target_file.unlink()
            target_file.symlink_to(source_file.resolve())
PY

    touch "${RUN_ROOT}/static_filter_inputs.ready"
fi

WAIT_START="$(date +%s)"
while [ ! -f "${RUN_ROOT}/static_filter_inputs.ready" ]; do
    if [ "$SYNC_TIMEOUT_SECONDS" -gt 0 ]; then
        NOW="$(date +%s)"
        if [ $((NOW - WAIT_START)) -ge "$SYNC_TIMEOUT_SECONDS" ]; then
            echo "Timed out waiting for static filter shard preparation" >&2
            exit 1
        fi
    fi
    sleep "$SYNC_POLL_SECONDS"
done

for ((local_index = 0; local_index < STATIC_FILTER_N; local_index++)); do
    GLOBAL_SHARD_INDEX="$((STATIC_FILTER_NODE_OFFSET + local_index))"
    if [ "$GLOBAL_SHARD_INDEX" -ge "$STATIC_FILTER_GLOBAL_N" ]; then
        continue
    fi

    SHARD_INPUT_DIR="${SHARD_ROOT}/inputs_${GLOBAL_SHARD_INDEX}"
    SHARD_RESULT_DIR="${SHARD_ROOT}/result_${GLOBAL_SHARD_INDEX}"
    SHARD_DONE_FILE="${SHARD_RESULT_DIR}/done"
    SHARD_CONFIG_FILE="${SHARD_RESULT_DIR}/config"
    SHARD_CONFIG="source_dir=${SOURCE_DIR_REAL};static_filter_samples_per_prompt=${STATIC_FILTER_SAMPLES_PER_PROMPT};static_filter_global_n=${STATIC_FILTER_GLOBAL_N}"

    if [ -f "$SHARD_DONE_FILE" ] && [ -f "$SHARD_CONFIG_FILE" ] && grep -Fxq "$SHARD_CONFIG" "$SHARD_CONFIG_FILE"; then
        continue
    fi

    mkdir -p "$SHARD_RESULT_DIR"

    if ! find "$SHARD_INPUT_DIR" -maxdepth 1 -name '*.mp4' -print -quit | grep -q .; then
        printf '{}\n' > "${SHARD_RESULT_DIR}/filtered_static_video.json"
        mkdir -p "${SHARD_RESULT_DIR}/filtered_videos"
        find "${SHARD_RESULT_DIR}/filtered_videos" -maxdepth 1 -type f -name '*.mp4' -delete
        printf '%s\n' "$SHARD_CONFIG" > "$SHARD_CONFIG_FILE"
        touch "$SHARD_DONE_FILE"
        continue
    fi

    (
        CUDA_VISIBLE_DEVICES="${GPU_IDS[$local_index]}" \
        vbench static_filter \
            --videos_path "$SHARD_INPUT_DIR" \
            --result_path "$SHARD_RESULT_DIR" \
            --filter_scope all
        find "${SHARD_RESULT_DIR}/filtered_videos" -maxdepth 1 -type f -name '*.mp4' -delete
        printf '%s\n' "$SHARD_CONFIG" > "$SHARD_CONFIG_FILE"
        touch "$SHARD_DONE_FILE"
    ) &
done

wait
touch "${RUN_ROOT}/static_filter_node_${NODE_RANK}.done"

WAIT_START="$(date +%s)"
while true; do
    DONE_NODE_COUNT="$(find "$RUN_ROOT" -maxdepth 1 -type f -name 'static_filter_node_*.done' | wc -l)"
    if [ "$DONE_NODE_COUNT" -ge "$NNODES" ]; then
        break
    fi
    if [ "$SYNC_TIMEOUT_SECONDS" -gt 0 ]; then
        NOW="$(date +%s)"
        if [ $((NOW - WAIT_START)) -ge "$SYNC_TIMEOUT_SECONDS" ]; then
            echo "Timed out waiting for all nodes to finish static filtering" >&2
            exit 1
        fi
    fi
    sleep "$SYNC_POLL_SECONDS"
done

if [ "$NODE_RANK" -eq 0 ]; then
    python - <<'PY' "$SHARD_ROOT" "$RESULT_DIR" "$PROMPT_FILE" "$VBENCH_SAMPLES_PER_PROMPT" "$STATIC_FILTER_GLOBAL_N"
import json
import pathlib
import re
import sys

shard_root = pathlib.Path(sys.argv[1])
result_dir = pathlib.Path(sys.argv[2])
prompt_file = pathlib.Path(sys.argv[3])
samples_per_prompt = int(sys.argv[4])
shard_count = int(sys.argv[5])
dest_dir = result_dir / "filtered_videos"
dest_dir.mkdir(parents=True, exist_ok=True)

prompts = [line.strip() for line in prompt_file.read_text().splitlines()[:75] if line.strip()]
for prompt in prompts:
    for index in range(samples_per_prompt):
        target_file = dest_dir / f"{prompt}-{index}.mp4"
        if target_file.exists() or target_file.is_symlink():
            target_file.unlink()

merged = {}
for shard_index in range(shard_count):
    info_file = shard_root / f"result_{shard_index}" / "filtered_static_video.json"
    if not info_file.exists():
        raise FileNotFoundError(info_file)
    data = json.loads(info_file.read_text())
    for prompt, value in data.items():
        if prompt in merged:
            raise RuntimeError(f"duplicate prompt in static filter shards: {prompt}")
        value["static_path"] = sorted(
            value["static_path"],
            key=lambda path: (
                int(match.group(1)) if (match := re.search(r"-(\d+)\.[^.]+$", pathlib.Path(path).name)) else 10**9,
                path,
            ),
        )
        merged[prompt] = value
        for index, video_path in enumerate(value["static_path"][:samples_per_prompt]):
            target_file = dest_dir / f"{prompt}-{index}.mp4"
            if target_file.exists() or target_file.is_symlink():
                target_file.unlink()
            target_file.symlink_to(pathlib.Path(video_path).resolve())

(result_dir / "filtered_static_video.json").write_text(json.dumps(merged))
PY

    mkdir -p "$DEST_DIR"
    tail -n +76 "$PROMPT_FILE" | while IFS= read -r prompt || [[ -n "$prompt" ]]; do
        for index in $(seq 0 $((VBENCH_SAMPLES_PER_PROMPT - 1))); do
            filename="${prompt}-${index}.mp4"
            source_file="${SOURCE_DIR}/${filename}"
            target_file="${DEST_DIR}/${filename}"
            if [ -f "$target_file" ] || [ -L "$target_file" ]; then
                unlink "$target_file"
            fi
            if [ -f "$source_file" ]; then
                ln -sf "$(realpath "$source_file")" "$target_file"
            fi
        done
    done

    touch "${RUN_ROOT}/static_filter_merge.done"
fi

WAIT_START="$(date +%s)"
while [ ! -f "${RUN_ROOT}/static_filter_merge.done" ]; do
    if [ "$SYNC_TIMEOUT_SECONDS" -gt 0 ]; then
        NOW="$(date +%s)"
        if [ $((NOW - WAIT_START)) -ge "$SYNC_TIMEOUT_SECONDS" ]; then
            echo "Timed out waiting for merged static filter outputs" >&2
            exit 1
        fi
    fi
    sleep "$SYNC_POLL_SECONDS"
done

EVAL_GPU_IDS=("${GPU_IDS[@]:0:$N}")
EVAL_CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${EVAL_GPU_IDS[*]}")"

if ! find "${RESULT_DIR}/evaluation_results" -maxdepth 1 -name '*_eval_results.json' -print -quit | grep -q .; then
    CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" \
    torchrun \
        --nnodes "$NNODES" \
        --nproc_per_node "$NPROC_PER_NODE" \
        --node_rank "$NODE_RANK" \
        --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
        --rdzv-backend "$RDZV_BACKEND" \
        --local_addr="$LOCAL_ADDR" \
        -m vbench.launch.evaluate \
        --videos_path "$DEST_DIR" \
        --dimension \
            subject_consistency \
            background_consistency \
            temporal_flickering \
            motion_smoothness \
            dynamic_degree \
            aesthetic_quality \
            imaging_quality \
            object_class \
            multiple_objects \
            human_action \
            color \
            spatial_relationship \
            scene \
            temporal_style \
            appearance_style \
            overall_consistency \
        --load_ckpt_from_local True \
        --output_path "${RESULT_DIR}/evaluation_results"
fi

if [ "$NODE_RANK" -eq 0 ]; then
    python -m third_party.vbench.cal_final_score --model_name "${RESULT_DIR}/evaluation_results"
    touch "${RUN_ROOT}/final_score.done"
fi

WAIT_START="$(date +%s)"
while [ ! -f "${RUN_ROOT}/final_score.done" ]; do
    if [ "$SYNC_TIMEOUT_SECONDS" -gt 0 ]; then
        NOW="$(date +%s)"
        if [ $((NOW - WAIT_START)) -ge "$SYNC_TIMEOUT_SECONDS" ]; then
            echo "Timed out waiting for final score aggregation" >&2
            exit 1
        fi
    fi
    sleep "$SYNC_POLL_SECONDS"
done
