#!/bin/bash
set -euo pipefail

# ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align}"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "test_teacher_reasoned"

# Paths
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/sciqa_process2_teacher_reasoned.py"
RESULT_PATH="${RESULT_PATH:-${TEST_RESULT_DIR}/teacher_test_reasoned.json}"

# Teacher
TEACHER_ENABLE_THINKING=0
MAX_RETRIES=3
SLEEP_SEC=0

# Evaluation
EVAL_SPLIT="test"
EVAL_MAX_SAMPLES=0
EVAL_MAX_NEW_TOKENS=768
TEMPERATURE=0.0

align_vq_print_header "教师模型 Reasoned ScienceQA 评测"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "EVAL_ENTRY: ${EVAL_ENTRY}"
echo "SCIENCEQA_PATH: ${SCIENCEQA_PATH}"
echo "EVAL_SPLIT: ${EVAL_SPLIT}"
echo "VICTIM_MODEL: ${VICTIM_MODEL}"
echo "TEACHER_API_BASE: ${TEACHER_API_BASE}"
echo "TEACHER_ENABLE_THINKING: ${TEACHER_ENABLE_THINKING}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${EVAL_ENTRY}" "教师 reasoned 评测入口"
align_vq_assert_scienceqa_path

if [ -z "${TEACHER_API_KEY}" ]; then
    echo "错误: 未提供 TEACHER_API_KEY / OPENAI_API_KEY"
    exit 1
fi

mkdir -p "$(dirname "${RESULT_PATH}")"

"${PYTHON_BIN}" "${EVAL_ENTRY}" \
    --scienceqa_path="${SCIENCEQA_PATH}" \
    --split="${EVAL_SPLIT}" \
    --max_samples="${EVAL_MAX_SAMPLES}" \
    --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    --temperature="${TEMPERATURE}" \
    --victim_model="${VICTIM_MODEL}" \
    --teacher_api_base="${TEACHER_API_BASE}" \
    --teacher_api_key="${TEACHER_API_KEY}" \
    --teacher_enable_thinking="${TEACHER_ENABLE_THINKING}" \
    --max_retries="${MAX_RETRIES}" \
    --sleep_sec="${SLEEP_SEC}" \
    --save_path="${RESULT_PATH}"

align_vq_require_file "${RESULT_PATH}" "教师 reasoned 评测结果"
eval "$(align_vq_extract_eval_metrics "${RESULT_PATH}")"

align_vq_print_header "教师模型 Reasoned ScienceQA 评测完成"
echo "ACCURACY=${ACCURACY}"
echo "FORMAT_RATE=${FORMAT_RATE}"
echo "N=${N}"
echo "结果文件: ${RESULT_PATH}"
