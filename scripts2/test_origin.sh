#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align}"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "test_origin"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Paths
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/sciqa_process.py"
RESULT_PATH="${RESULT_PATH:-${TEST_RESULT_DIR}/origin_test_logits.json}"

# Evaluation
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-64}"
EVAL_ANSWER_MODE="${EVAL_ANSWER_MODE:-logits}"
USE_4BIT="${USE_4BIT:-0}"
VQ_CODEBOOK_SIZE="${VQ_CODEBOOK_SIZE:-1024}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-0}"

align_vq_print_header "原始学生模型评测"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SCIENCEQA_PATH: ${SCIENCEQA_PATH}"
echo "EVAL_SPLIT: ${EVAL_SPLIT}"
echo "EVAL_ANSWER_MODE: ${EVAL_ANSWER_MODE}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${EVAL_ENTRY}" "评测入口"
align_vq_assert_scienceqa_path
align_vq_require_path "${MODEL_PATH}" "模型路径"
mkdir -p "$(dirname "${RESULT_PATH}")"

"${PYTHON_BIN}" "${EVAL_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --adapter_path="" \
    --scienceqa_path="${SCIENCEQA_PATH}" \
    --split="${EVAL_SPLIT}" \
    --max_samples="${EVAL_MAX_SAMPLES}" \
    --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    --use_4bit="${USE_4BIT}" \
    --use_vq=0 \
    --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
    --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    --vq_codebook_path="" \
    --answer_mode="${EVAL_ANSWER_MODE}" \
    --save_path="${RESULT_PATH}"

align_vq_require_file "${RESULT_PATH}" "原始模型评测结果"
align_vq_print_header "原始学生模型评测完成"
echo "结果文件: ${RESULT_PATH}"
