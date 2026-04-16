#!/bin/bash

# ROOT_DEFAULT="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align"
# PYTHON_DEFAULT="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/.align_vq/bin/python"
# MODEL_DEFAULT="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf"
# SCIENCEQA_DEFAULT="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/datasets/ScienceQA"

# PYTHON_DEFAULT="/root/autodl-tmp/conda/envs/align_vq/bin/python"
# MODEL_DEFAULT="/root/autodl-tmp/models/llama3-llava-next-8b-hf"
# SCIENCEQA_DEFAULT="/root/autodl-tmp/datasets/ScienceQA"
ROOT_DEFAULT="/home/songxinhao/workspace/vq_lord"
PYTHON_DEFAULT="/mnt/shared-storage-gpfs2/evoagi-share-gpfs2/xhsong/miniconda3/envs/vq_lord/bin/python"
MODEL_DEFAULT="/mnt/shared-storage-gpfs2/evoagi-share-gpfs2/xhsong/models/llama3-llava-next-8b-hf"
SCIENCEQA_DEFAULT="/mnt/shared-storage-gpfs2/evoagi-share-gpfs2/xhsong/datasets/ScienceQA"
# SCIENCEQA_DEFAULT="/home/songxinhao/workspace/datasets/ScienceQA"

align_vq_init_paths() {
    ROOT_DIR="${ROOT_DIR:-$ROOT_DEFAULT}"
    ROOT_DIR="${ROOT_DIR%/}"

    SCRIPT_DIR="${ROOT_DIR}/scripts"
    PYTHON_BIN="${PYTHON_BIN:-$PYTHON_DEFAULT}"
    MODEL_PATH="${MODEL_PATH:-$MODEL_DEFAULT}"
    SCIENCEQA_PATH="${SCIENCEQA_PATH:-$SCIENCEQA_DEFAULT}"

    CKPT_DIR="${ROOT_DIR}/vq_lord_ckpts"
    DATA_DIR="${ROOT_DIR}/vq_lord_data"
    PREPROCESS_DIR="${DATA_DIR}/preprocess"
    TEST_RESULT_DIR="${ROOT_DIR}/vq_lord_test_results"
    LOG_DIR="${ROOT_DIR}/logs"

    export ROOT_DIR SCRIPT_DIR PYTHON_BIN MODEL_PATH SCIENCEQA_PATH
    export CKPT_DIR DATA_DIR PREPROCESS_DIR TEST_RESULT_DIR LOG_DIR
}

align_vq_setup_env() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export VICTIM_MODEL="${VICTIM_MODEL:-qwen3.5-flash-2026-02-23}"
    export TEACHER_API_BASE="${TEACHER_API_BASE:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}}}"
    export TEACHER_API_KEY="${TEACHER_API_KEY:-${OPENAI_API_KEY:-sk-abc8c59df2d64b7ba22718eae4fe80c2}}"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-121}"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
    export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
        export OMP_NUM_THREADS=8
    fi
}

align_vq_setup_distributed_env() {
    export DDP_NPROC="${DDP_NPROC:-1}"
    export TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
    export TORCHRUN_STANDALONE="${TORCHRUN_STANDALONE:-1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export ALIGN_VQ_TORCHRUN_MODE="${ALIGN_VQ_TORCHRUN_MODE:-bin}"

    if ! [[ "${DDP_NPROC}" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误: DDP_NPROC 必须是正整数，当前值: ${DDP_NPROC}"
        exit 1
    fi

    if [ "${DDP_NPROC}" -gt 1 ]; then
        if command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
            ALIGN_VQ_TORCHRUN_MODE="bin"
        elif "${PYTHON_BIN}" -c "import torch.distributed.run" >/dev/null 2>&1; then
            ALIGN_VQ_TORCHRUN_MODE="module"
            echo "提示: PATH 中未找到 ${TORCHRUN_BIN}，将回退为 ${PYTHON_BIN} -m torch.distributed.run"
        else
            echo "错误: 未找到 torchrun 启动器，且 ${PYTHON_BIN} 无法导入 torch.distributed.run"
            exit 1
        fi
        export ALIGN_VQ_TORCHRUN_MODE

        local visible_devices="${CUDA_VISIBLE_DEVICES:-}"
        if [ -n "${visible_devices}" ]; then
            local count=0
            local old_ifs="${IFS}"
            IFS=','
            # shellcheck disable=SC2206
            local device_items=(${visible_devices})
            IFS="${old_ifs}"
            count="${#device_items[@]}"
            if [ "${count}" -lt "${DDP_NPROC}" ]; then
                echo "错误: CUDA_VISIBLE_DEVICES=${visible_devices} 仅提供 ${count} 张卡，少于 DDP_NPROC=${DDP_NPROC}"
                exit 1
            fi
        fi
    fi
}

align_vq_make_train_launcher() {
    local out_name="$1"
    local -n out_ref="${out_name}"

    out_ref=("${PYTHON_BIN}")
    if [ "${DDP_NPROC:-1}" -le 1 ]; then
        return 0
    fi

    if [ "${ALIGN_VQ_TORCHRUN_MODE:-bin}" = "module" ]; then
        out_ref=("${PYTHON_BIN}" "-m" "torch.distributed.run")
    else
        out_ref=("${TORCHRUN_BIN}")
    fi
    if [ "${TORCHRUN_STANDALONE:-1}" = "1" ]; then
        out_ref+=("--standalone")
    fi
    out_ref+=("--nproc_per_node=${DDP_NPROC}")
    if [ -n "${MASTER_PORT:-}" ]; then
        out_ref+=("--master_port=${MASTER_PORT}")
    fi
}

align_vq_ensure_runtime_dirs() {
    mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${PREPROCESS_DIR}" "${TEST_RESULT_DIR}" "${LOG_DIR}"
}

align_vq_setup_logging() {
    local log_name="$1"
    # align_vq_ensure_runtime_dirs

    if [ "${ALIGN_VQ_LOGGING_CONFIGURED:-0}" = "1" ]; then
        return 0
    fi

    local timestamp
    timestamp="$(date -u +%Y%m%d_%H%M%S)"
    LOG_FILE="${LOG_FILE:-${LOG_DIR}/${log_name}_${timestamp}.log}"

    export LOG_FILE
    export ALIGN_VQ_LOGGING_CONFIGURED=1
    exec > >(tee -a "${LOG_FILE}") 2>&1
}

align_vq_print_header() {
    local title="$1"
    echo "======================================================"
    echo "${title}"
    echo "======================================================"
}

align_vq_require_path() {
    local path="$1"
    local label="$2"
    if [ ! -e "${path}" ]; then
        echo "错误: 未找到 ${label}: ${path}"
        exit 1
    fi
}

align_vq_require_file() {
    local path="$1"
    local label="$2"
    if [ ! -f "${path}" ]; then
        echo "错误: 未找到 ${label}: ${path}"
        exit 1
    fi
}

align_vq_require_dir() {
    local path="$1"
    local label="$2"
    if [ ! -d "${path}" ]; then
        echo "错误: 未找到 ${label}: ${path}"
        exit 1
    fi
}

align_vq_assert_scienceqa_path() {
    if [ ! -d "${SCIENCEQA_PATH}" ] && [ "${HF_DATASETS_OFFLINE:-0}" = "1" ]; then
        echo "错误: HF_DATASETS_OFFLINE=1，但本地 ScienceQA 目录不存在: ${SCIENCEQA_PATH}"
        exit 1
    fi
}

align_vq_prepare_scienceqa_preprocess() {
    local output_path="$1"
    local split="$2"
    local train_num="$3"
    local seed="$4"
    local bucket_by="$5"
    local bucket_batch_size="$6"
    local bucket_drop_last="$7"
    local shuffle="${8:-1}"
    local preview_buckets="${9:-10}"
    local preview_batches="${10:-10}"

    local preprocess_entry="${ROOT_DIR}/data_preprocess/sciqa_preprocess.py"

    align_vq_require_file "${preprocess_entry}" "预处理入口"
    align_vq_require_path "${MODEL_PATH}" "模型路径"
    align_vq_assert_scienceqa_path
    mkdir -p "$(dirname "${output_path}")"

    if [ -f "${output_path}" ] && [ "${PREPROCESS_FORCE_REBUILD:-0}" != "1" ]; then
        echo "复用预处理文件: ${output_path}"
    else
        echo "生成预处理文件: ${output_path}"
        "${PYTHON_BIN}" "${preprocess_entry}" \
            --dataset-path="${SCIENCEQA_PATH}" \
            --model-path="${MODEL_PATH}" \
            --split="${split}" \
            --train-num="${train_num}" \
            --seed="${seed}" \
            --bucket-by="${bucket_by}" \
            --bucket-batch-size="${bucket_batch_size}" \
            --bucket-drop-last="${bucket_drop_last}" \
            --shuffle="${shuffle}" \
            --preview-buckets="${preview_buckets}" \
            --preview-batches="${preview_batches}" \
            --save-json="${output_path}"
    fi

    align_vq_require_file "${output_path}" "预处理输出"
}

align_vq_require_stage1_artifacts() {
    local codebook_path="$1"
    align_vq_require_file "${codebook_path}" "Stage1 vq_codebook"
}

align_vq_require_stage2_artifacts() {
    local ckpt_path="$1"
    align_vq_require_dir "${ckpt_path}" "Stage2 checkpoint 目录"
    align_vq_require_file "${ckpt_path}/vq_codebook.pt" "Stage2 vq_codebook"
    align_vq_require_file "${ckpt_path}/adapter_config.json" "Stage2 adapter_config.json"
    align_vq_require_file "${ckpt_path}/projector.pt" "Stage2 projector.pt"
}

align_vq_extract_eval_metrics() {
    local json_path="$1"
    "${PYTHON_BIN}" - "${json_path}" <<'PY'
import json
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

metrics = payload.get("metrics", {})
acc = float(metrics.get("accuracy", 0.0))
results = payload.get("results", [])

pat = re.compile(r"(?:answer|答案)\s*[:：]\s*\(?\s*([A-D])\s*\)?", re.IGNORECASE)
head_pat = re.compile(r"^\s*\(?\s*([A-D])\s*\)?\b", re.IGNORECASE)

fmt_hits = 0
for item in results:
    text = str(item.get("output", "") or "")
    if pat.search(text) or head_pat.search(text):
        fmt_hits += 1

total = max(1, len(results))
fmt_rate = fmt_hits / total

print(f"ACCURACY={acc:.6f}")
print(f"FORMAT_RATE={fmt_rate:.6f}")
print(f"N={len(results)}")
PY
}
