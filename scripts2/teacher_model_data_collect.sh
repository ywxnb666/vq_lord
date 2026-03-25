#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align}"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "teacher_model_data_collect"

# Paths
COLLECT_ENTRY="${ROOT_DIR}/vq_lord3/data_collector2.py"

# Data
SCIENCEQA_SPLIT="${SCIENCEQA_SPLIT:-train}"
TRAIN_NUM="${TRAIN_NUM:-0}"
SCIENCEQA_SEED="${SCIENCEQA_SEED:-20240306}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# Teacher
VICTIM_MODEL="${VICTIM_MODEL:-qwen3.5-flash-2026-02-23}"
TEACHER_LANG="${TEACHER_LANG:-en}"
TEACHER_API_BASE="${TEACHER_API_BASE:-${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
TEACHER_API_KEY="${TEACHER_API_KEY:-${OPENAI_API_KEY:-}}"
TEACHER_ENABLE_THINKING="${TEACHER_ENABLE_THINKING:-0}"
COLLECT_TEACHER_DATA="${COLLECT_TEACHER_DATA:-1}"
STRICT_TEACHER_DISTILL="${STRICT_TEACHER_DISTILL:-1}"
TEACHER_OBSERVED_MAX_TOKENS="${TEACHER_OBSERVED_MAX_TOKENS:-256}"
TEACHER_CONTEXT_MAX_TOKENS="${TEACHER_CONTEXT_MAX_TOKENS:-192}"
TEACHER_REASONING_MAX_TOKENS="${TEACHER_REASONING_MAX_TOKENS:-256}"
TEACHER_ANSWER_MAX_TOKENS="${TEACHER_ANSWER_MAX_TOKENS:-64}"
TEACHER_MAX_NEW_TOKENS_TOTAL="${TEACHER_MAX_NEW_TOKENS_TOTAL:-768}"

# Runtime
MAX_RETRIES="${MAX_RETRIES:-3}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SLEEP_SEC="${SLEEP_SEC:-0}"
VICTIM_TAG="$(echo "${VICTIM_MODEL}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
TEACHER_CACHE_PATH="${TEACHER_CACHE_PATH:-${DATA_DIR}/scienceqa_teacher_${VICTIM_TAG}_${SCIENCEQA_SPLIT}_n${TRAIN_NUM}_seed${SCIENCEQA_SEED}_new.json}"

align_vq_print_header "教师模型标注采集"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "COLLECT_ENTRY: ${COLLECT_ENTRY}"
echo "SCIENCEQA_PATH: ${SCIENCEQA_PATH}"
echo "SCIENCEQA_SPLIT: ${SCIENCEQA_SPLIT}"
echo "TRAIN_NUM: ${TRAIN_NUM}"
echo "VICTIM_MODEL: ${VICTIM_MODEL}"
echo "TEACHER_API_BASE: ${TEACHER_API_BASE}"
echo "TEACHER_CACHE_PATH: ${TEACHER_CACHE_PATH}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${COLLECT_ENTRY}" "教师标注入口"
align_vq_assert_scienceqa_path

if [ "${COLLECT_TEACHER_DATA}" = "1" ] && [ -z "${TEACHER_API_KEY}" ]; then
    echo "错误: COLLECT_TEACHER_DATA=1，但未提供 TEACHER_API_KEY / OPENAI_API_KEY"
    exit 1
fi

"${PYTHON_BIN}" "${COLLECT_ENTRY}" \
    --task="collect_scienceqa_struct" \
    --scienceqa_path="${SCIENCEQA_PATH}" \
    --scienceqa_split="${SCIENCEQA_SPLIT}" \
    --train_num="${TRAIN_NUM}" \
    --scienceqa_seed="${SCIENCEQA_SEED}" \
    --max_samples="${MAX_SAMPLES}" \
    --data_dir="${DATA_DIR}" \
    --teacher_cache_path="${TEACHER_CACHE_PATH}" \
    --victim_model="${VICTIM_MODEL}" \
    --teacher_lang="${TEACHER_LANG}" \
    --teacher_api_base="${TEACHER_API_BASE}" \
    --teacher_api_key="${TEACHER_API_KEY}" \
    --teacher_enable_thinking="${TEACHER_ENABLE_THINKING}" \
    --collect_teacher_data="${COLLECT_TEACHER_DATA}" \
    --strict_teacher_distill="${STRICT_TEACHER_DISTILL}" \
    --teacher_observed_max_tokens="${TEACHER_OBSERVED_MAX_TOKENS}" \
    --teacher_context_max_tokens="${TEACHER_CONTEXT_MAX_TOKENS}" \
    --teacher_reasoning_max_tokens="${TEACHER_REASONING_MAX_TOKENS}" \
    --teacher_answer_max_tokens="${TEACHER_ANSWER_MAX_TOKENS}" \
    --teacher_max_new_tokens_total="${TEACHER_MAX_NEW_TOKENS_TOTAL}" \
    --max_retries="${MAX_RETRIES}" \
    --num_workers="${NUM_WORKERS}" \
    --save_every="${SAVE_EVERY}" \
    --sleep_sec="${SLEEP_SEC}"

align_vq_require_file "${TEACHER_CACHE_PATH}" "教师标注缓存"
align_vq_print_header "教师模型标注采集完成"
echo "缓存文件: ${TEACHER_CACHE_PATH}"
