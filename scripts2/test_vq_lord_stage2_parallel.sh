#!/bin/bash
set -euo pipefail

SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "test_vq_lord_stage2_parallel"

# Evaluation
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
EVAL_ANSWER_MODE="${EVAL_ANSWER_MODE:-logits}"
USE_4BIT="${USE_4BIT:-0}"
USE_VQ="${USE_VQ:-0}"
VQ_CODEBOOK_SIZE="${VQ_CODEBOOK_SIZE:-1024}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-0}"

# Parallel
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"

# Paths
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/sciqa_process_parallel.py"
STAGE2_CKPT_PATH="${STAGE2_CKPT_PATH:-${CKPT_DIR}/stage2/stage2_vision_epoch13}"
SHARD_RESULT_DIR="${SHARD_RESULT_DIR:-${TEST_RESULT_DIR}/stage2_${EVAL_SPLIT}_${EVAL_ANSWER_MODE}_vq${USE_VQ}_parallel_shards}"
RESULT_PATH="${RESULT_PATH:-${TEST_RESULT_DIR}/stage2_${EVAL_SPLIT}_${EVAL_ANSWER_MODE}_vq${USE_VQ}_parallel.json}"

align_vq_print_header "Stage2 产物并行评测"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SCIENCEQA_PATH: ${SCIENCEQA_PATH}"
echo "EVAL_ENTRY: ${EVAL_ENTRY}"
echo "STAGE2_CKPT_PATH: ${STAGE2_CKPT_PATH}"
echo "NUM_SHARDS: ${NUM_SHARDS}"
echo "GPU_IDS: ${GPU_IDS}"
echo "EVAL_MAX_NEW_TOKENS: ${EVAL_MAX_NEW_TOKENS}"
echo "SHARD_RESULT_DIR: ${SHARD_RESULT_DIR}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${EVAL_ENTRY}" "并行评测入口"
align_vq_require_stage2_artifacts "${STAGE2_CKPT_PATH}"
align_vq_assert_scienceqa_path

mkdir -p "${SHARD_RESULT_DIR}" "$(dirname "${RESULT_PATH}")"

read -r -a GPU_ID_ARR <<< "${GPU_IDS}"
if [ "${#GPU_ID_ARR[@]}" -lt "${NUM_SHARDS}" ]; then
    echo "错误: GPU_IDS 数量不足，至少需要 ${NUM_SHARDS} 个 GPU id，当前只有 ${#GPU_ID_ARR[@]}"
    exit 1
fi

PIDS=()
SHARD_LOGS=()

for (( shard_id=0; shard_id<NUM_SHARDS; shard_id++ )); do
    gpu_id="${GPU_ID_ARR[$shard_id]}"
    shard_result_path="${SHARD_RESULT_DIR}/shard_$(printf '%02d' "${shard_id}").json"
    shard_log_path="${LOG_DIR}/test_vq_lord_stage2_parallel_shard$(printf '%02d' "${shard_id}")_$(date -u +%Y%m%d_%H%M%S).log"
    SHARD_LOGS+=("${shard_log_path}")

    echo "[Launch] shard=${shard_id}/${NUM_SHARDS} gpu=${gpu_id} result=${shard_result_path}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        "${PYTHON_BIN}" "${EVAL_ENTRY}" \
            --model_path="${MODEL_PATH}" \
            --adapter_path="${STAGE2_CKPT_PATH}" \
            --scienceqa_path="${SCIENCEQA_PATH}" \
            --split="${EVAL_SPLIT}" \
            --max_samples="${EVAL_MAX_SAMPLES}" \
            --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
            --use_4bit="${USE_4BIT}" \
            --use_vq="${USE_VQ}" \
            --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
            --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
            --vq_codebook_path="${STAGE2_CKPT_PATH}/vq_codebook.pt" \
            --answer_mode="${EVAL_ANSWER_MODE}" \
            --num_shards="${NUM_SHARDS}" \
            --shard_id="${shard_id}" \
            --save_path="${shard_result_path}"
    ) > "${shard_log_path}" 2>&1 &
    PIDS+=("$!")
done

FAILED=0
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    shard_id="${idx}"
    if wait "${pid}"; then
        echo "[Done] shard=${shard_id} log=${SHARD_LOGS[$idx]}"
    else
        echo "[Error] shard=${shard_id} failed, log=${SHARD_LOGS[$idx]}"
        FAILED=1
    fi
done

if [ "${FAILED}" -ne 0 ]; then
    echo "错误: 至少一个 shard 评测失败，请检查对应 shard 日志"
    exit 1
fi

"${PYTHON_BIN}" "${EVAL_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --adapter_path="${STAGE2_CKPT_PATH}" \
    --scienceqa_path="${SCIENCEQA_PATH}" \
    --split="${EVAL_SPLIT}" \
    --max_samples="${EVAL_MAX_SAMPLES}" \
    --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    --use_4bit="${USE_4BIT}" \
    --use_vq="${USE_VQ}" \
    --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
    --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    --vq_codebook_path="${STAGE2_CKPT_PATH}/vq_codebook.pt" \
    --answer_mode="${EVAL_ANSWER_MODE}" \
    --num_shards="${NUM_SHARDS}" \
    --shard_result_dir="${SHARD_RESULT_DIR}" \
    --merge_only=1 \
    --save_path="${RESULT_PATH}"

align_vq_require_file "${RESULT_PATH}" "Stage2 并行 test 结果"
eval "$(align_vq_extract_eval_metrics "${RESULT_PATH}")"

align_vq_print_header "Stage2 产物并行评测完成"
echo "ACCURACY=${ACCURACY}"
echo "FORMAT_RATE=${FORMAT_RATE}"
echo "N=${N}"
echo "Shard 目录: ${SHARD_RESULT_DIR}"
echo "结果文件: ${RESULT_PATH}"
