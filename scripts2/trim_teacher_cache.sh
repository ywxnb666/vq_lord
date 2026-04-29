#!/bin/bash
set -euo pipefail

SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "trim_teacher_cache"

# Paths
TRIM_ENTRY="${ROOT_DIR}/vq_lord3/trim_teacher_cache.py"

# Data
SCIENCEQA_SPLIT="${SCIENCEQA_SPLIT:-train}"
SCIENCEQA_SEED="${SCIENCEQA_SEED:-20240306}"
SOURCE_TRAIN_NUM="${SOURCE_TRAIN_NUM:-6200}"
TRIM_SAMPLE_COUNT="${TRIM_SAMPLE_COUNT:-320}"
TARGET_TRAIN_NUM="${TARGET_TRAIN_NUM:-${TRIM_SAMPLE_COUNT}}"
VICTIM_TAG="$(echo "${VICTIM_MODEL}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
INPUT_TEACHER_CACHE_PATH="${INPUT_TEACHER_CACHE_PATH:-${DATA_DIR}/aokvqa_teacher_${VICTIM_TAG}_${SCIENCEQA_SPLIT}_n${SOURCE_TRAIN_NUM}_new.json}"
OUTPUT_TEACHER_CACHE_PATH="${OUTPUT_TEACHER_CACHE_PATH:-${DATA_DIR}/aokvqa_teacher_${VICTIM_TAG}_${SCIENCEQA_SPLIT}_n${TARGET_TRAIN_NUM}_new.json}"

align_vq_print_header "裁剪教师缓存"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "TRIM_ENTRY: ${TRIM_ENTRY}"
echo "VICTIM_MODEL: ${VICTIM_MODEL}"
echo "SCIENCEQA_SPLIT: ${SCIENCEQA_SPLIT}"
echo "SCIENCEQA_SEED: ${SCIENCEQA_SEED}"
echo "SOURCE_TRAIN_NUM: ${SOURCE_TRAIN_NUM}"
echo "TRIM_SAMPLE_COUNT: ${TRIM_SAMPLE_COUNT}"
echo "TARGET_TRAIN_NUM: ${TARGET_TRAIN_NUM}"
echo "INPUT_TEACHER_CACHE_PATH: ${INPUT_TEACHER_CACHE_PATH}"
echo "OUTPUT_TEACHER_CACHE_PATH: ${OUTPUT_TEACHER_CACHE_PATH}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${TRIM_ENTRY}" "裁剪脚本入口"
align_vq_require_file "${INPUT_TEACHER_CACHE_PATH}" "输入教师缓存"

mkdir -p "$(dirname "${OUTPUT_TEACHER_CACHE_PATH}")"

"${PYTHON_BIN}" "${TRIM_ENTRY}" \
    --input-path="${INPUT_TEACHER_CACHE_PATH}" \
    --output-path="${OUTPUT_TEACHER_CACHE_PATH}" \
    --sample-count="${TRIM_SAMPLE_COUNT}" \
    --train-num="${TARGET_TRAIN_NUM}"

align_vq_require_file "${OUTPUT_TEACHER_CACHE_PATH}" "输出教师缓存"

align_vq_print_header "裁剪教师缓存完成"
echo "输出文件: ${OUTPUT_TEACHER_CACHE_PATH}"
