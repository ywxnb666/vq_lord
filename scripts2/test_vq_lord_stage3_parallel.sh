#!/bin/bash
set -euo pipefail

# shellcheck source=./common.sh
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "test_vq_lord_stage3_parallel"

# Evaluation
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-16}"
ENABLE_SECOND_PASS="${ENABLE_SECOND_PASS:-1}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-1024}"
EVAL_ANSWER_MODE="${EVAL_ANSWER_MODE:-generate}"
USE_4BIT="${USE_4BIT:-0}"
USE_VQ="${USE_VQ:-1}"
if [ "${REMOVE_VQ_CODEBOOK}" = "1" ]; then
    USE_VQ=0
fi
VQ_CODEBOOK_SIZE="${VQ_CODEBOOK_SIZE:-1024}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-0}"

# Stable conservative batching
SCIENCEQA_SEED="${SCIENCEQA_SEED:-20240306}"
EVAL_BUCKET_BATCH_SIZE="${EVAL_BUCKET_BATCH_SIZE:-8}"
PREPROCESS_SHUFFLE="${PREPROCESS_SHUFFLE:-1}"

# Parallel
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"

# Paths
PREPROCESS_ENTRY="${ROOT_DIR}/data_preprocess/sciqa_preprocess.py"
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/sciqa_process2_parallel.py"
# STAGE3_FINAL_ADAPTER_PATH="${STAGE3_FINAL_ADAPTER_PATH:-${CKPT_DIR}/stage3/stage3_sub1_period7}"
STAGE3_FINAL_ADAPTER_PATH="/home/songxinhao/workspace/vq_lord/vq_lord_ckpts/Qwen3.5flash/stage3/stage3_sub1_period7"
# STAGE3_FINAL_ADAPTER_PATH="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts_stage3_tune/run_20260323_140152/stage3_sub1_period7"
BUCKET_PLAN_PATH="${BUCKET_PLAN_PATH:-${PREPROCESS_DIR}/scienceqa_${EVAL_SPLIT}_n${EVAL_MAX_SAMPLES}_seed${SCIENCEQA_SEED}_patches_bs${EVAL_BUCKET_BATCH_SIZE}.json}"
SHARD_RESULT_DIR="${SHARD_RESULT_DIR:-${TEST_RESULT_DIR}/stage3_${EVAL_SPLIT}_${EVAL_ANSWER_MODE}_vq${USE_VQ}_bucketed_shards}"
RESULT_PATH="${RESULT_PATH:-${TEST_RESULT_DIR}/stage3_${EVAL_SPLIT}_${EVAL_ANSWER_MODE}_vq${USE_VQ}_bucketed_parallel.json}"

align_vq_print_header "Stage3 产物并行分桶评测"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SCIENCEQA_PATH: ${SCIENCEQA_PATH}"
echo "PREPROCESS_ENTRY: ${PREPROCESS_ENTRY}"
echo "EVAL_ENTRY: ${EVAL_ENTRY}"
echo "STAGE3_FINAL_ADAPTER_PATH: ${STAGE3_FINAL_ADAPTER_PATH}"
echo "BUCKET_PLAN_PATH: ${BUCKET_PLAN_PATH}"
echo "NUM_SHARDS: ${NUM_SHARDS}"
echo "GPU_IDS: ${GPU_IDS}"
echo "EVAL_MAX_NEW_TOKENS: ${EVAL_MAX_NEW_TOKENS}"
echo "ENABLE_SECOND_PASS: ${ENABLE_SECOND_PASS}"
echo "SECOND_PASS_MAX_NEW_TOKENS: ${SECOND_PASS_MAX_NEW_TOKENS}"
echo "EVAL_BUCKET_BATCH_SIZE: ${EVAL_BUCKET_BATCH_SIZE}"
echo "SHARD_RESULT_DIR: ${SHARD_RESULT_DIR}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${PREPROCESS_ENTRY}" "预处理分桶入口"
align_vq_require_file "${EVAL_ENTRY}" "并行分桶评测入口"
align_vq_require_dir "${STAGE3_FINAL_ADAPTER_PATH}" "Stage3 最终 adapter 目录"
align_vq_require_file "${STAGE3_FINAL_ADAPTER_PATH}/adapter_config.json" "Stage3 adapter_config.json"
if [ "${REMOVE_VQ_CODEBOOK}" != "1" ]; then
    align_vq_require_file "${STAGE3_FINAL_ADAPTER_PATH}/vq_codebook.pt" "Stage3 vq_codebook"
fi
align_vq_assert_scienceqa_path

mkdir -p "${PREPROCESS_DIR}" "${SHARD_RESULT_DIR}" "$(dirname "${RESULT_PATH}")"

if [ ! -f "${BUCKET_PLAN_PATH}" ]; then
    echo "[Info] 未找到 test split 分桶文件，开始生成: ${BUCKET_PLAN_PATH}"
    "${PYTHON_BIN}" "${PREPROCESS_ENTRY}" \
        --dataset-path="${SCIENCEQA_PATH}" \
        --model-path="${MODEL_PATH}" \
        --split="${EVAL_SPLIT}" \
        --train-num="${EVAL_MAX_SAMPLES}" \
        --seed="${SCIENCEQA_SEED}" \
        --bucket-by="patches" \
        --bucket-batch-size="${EVAL_BUCKET_BATCH_SIZE}" \
        --bucket-drop-last="0" \
        --shuffle="${PREPROCESS_SHUFFLE}" \
        --save-json="${BUCKET_PLAN_PATH}"
fi

align_vq_require_file "${BUCKET_PLAN_PATH}" "评测分桶文件"

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
    shard_log_path="${LOG_DIR}/test_vq_lord_stage3_parallel_shard$(printf '%02d' "${shard_id}")_$(date -u +%Y%m%d_%H%M%S).log"
    SHARD_LOGS+=("${shard_log_path}")

    echo "[Launch] shard=${shard_id}/${NUM_SHARDS} gpu=${gpu_id} result=${shard_result_path}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        "${PYTHON_BIN}" "${EVAL_ENTRY}" \
            --model_path="${MODEL_PATH}" \
    --student_model_type="${STUDENT_MODEL_TYPE}" \
            --adapter_path="${STAGE3_FINAL_ADAPTER_PATH}" \
            --scienceqa_path="${SCIENCEQA_PATH}" \
            --split="${EVAL_SPLIT}" \
            --max_samples="${EVAL_MAX_SAMPLES}" \
            --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
            --enable_second_pass="${ENABLE_SECOND_PASS}" \
            --second_pass_max_new_tokens="${SECOND_PASS_MAX_NEW_TOKENS}" \
            --use_4bit="${USE_4BIT}" \
            --use_vq="${USE_VQ}" \
            --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
            --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
            --vq_codebook_path="${STAGE3_FINAL_ADAPTER_PATH}/vq_codebook.pt" \
            --answer_mode="${EVAL_ANSWER_MODE}" \
            --bucket_plan_path="${BUCKET_PLAN_PATH}" \
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
    --student_model_type="${STUDENT_MODEL_TYPE}" \
    --adapter_path="${STAGE3_FINAL_ADAPTER_PATH}" \
    --scienceqa_path="${SCIENCEQA_PATH}" \
    --split="${EVAL_SPLIT}" \
    --max_samples="${EVAL_MAX_SAMPLES}" \
    --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    --enable_second_pass="${ENABLE_SECOND_PASS}" \
    --second_pass_max_new_tokens="${SECOND_PASS_MAX_NEW_TOKENS}" \
    --use_4bit="${USE_4BIT}" \
    --use_vq="${USE_VQ}" \
    --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
    --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    --vq_codebook_path="${STAGE3_FINAL_ADAPTER_PATH}/vq_codebook.pt" \
    --answer_mode="${EVAL_ANSWER_MODE}" \
    --bucket_plan_path="${BUCKET_PLAN_PATH}" \
    --num_shards="${NUM_SHARDS}" \
    --shard_result_dir="${SHARD_RESULT_DIR}" \
    --merge_only=1 \
    --save_path="${RESULT_PATH}"

align_vq_require_file "${RESULT_PATH}" "Stage3 并行分桶 test 结果"
eval "$(align_vq_extract_eval_metrics "${RESULT_PATH}")"

align_vq_print_header "Stage3 产物并行分桶评测完成"
echo "ACCURACY=${ACCURACY}"
echo "FORMAT_RATE=${FORMAT_RATE}"
echo "N=${N}"
echo "Bucket 文件: ${BUCKET_PLAN_PATH}"
echo "Shard 目录: ${SHARD_RESULT_DIR}"
echo "结果文件: ${RESULT_PATH}"
