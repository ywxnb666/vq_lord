#!/bin/bash
set -euo pipefail

# ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align}"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "data_preprocess"

# Paths
PREPROCESS_ENTRY="${ROOT_DIR}/data_preprocess/sciqa_preprocess.py"

# Data
TRAIN_NUM="${TRAIN_NUM:-0}"
SCIENCEQA_SPLIT="${SCIENCEQA_SPLIT:-train}"
SCIENCEQA_SEED="${SCIENCEQA_SEED:-20240306}"
BUCKET_BY="${BUCKET_BY:-patches}"
BUCKET_BATCH_SIZE="${BUCKET_BATCH_SIZE:-8}"
BUCKET_DROP_LAST="${BUCKET_DROP_LAST:-0}"
SHUFFLE="${SHUFFLE:-1}"
PREVIEW_BUCKETS="${PREVIEW_BUCKETS:-10}"
PREVIEW_BATCHES="${PREVIEW_BATCHES:-10}"
SCIENCEQA_PREPROCESSED_PATH="${SCIENCEQA_PREPROCESSED_PATH:-${PREPROCESS_DIR}/scienceqa_${SCIENCEQA_SPLIT}_n${TRAIN_NUM}_seed${SCIENCEQA_SEED}_${BUCKET_BY}_bs${BUCKET_BATCH_SIZE}.json}"

align_vq_print_header "ScienceQA 预处理"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SCIENCEQA_PATH: ${SCIENCEQA_PATH}"
echo "SCIENCEQA_SPLIT: ${SCIENCEQA_SPLIT}"
echo "TRAIN_NUM: ${TRAIN_NUM}"
echo "SCIENCEQA_SEED: ${SCIENCEQA_SEED}"
echo "BUCKET_BY: ${BUCKET_BY}"
echo "BUCKET_BATCH_SIZE: ${BUCKET_BATCH_SIZE}"
echo "SCIENCEQA_PREPROCESSED_PATH: ${SCIENCEQA_PREPROCESSED_PATH}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${PREPROCESS_ENTRY}" "预处理入口"
align_vq_prepare_scienceqa_preprocess \
    "${SCIENCEQA_PREPROCESSED_PATH}" \
    "${SCIENCEQA_SPLIT}" \
    "${TRAIN_NUM}" \
    "${SCIENCEQA_SEED}" \
    "${BUCKET_BY}" \
    "${BUCKET_BATCH_SIZE}" \
    "${BUCKET_DROP_LAST}" \
    "${SHUFFLE}" \
    "${PREVIEW_BUCKETS}" \
    "${PREVIEW_BATCHES}"

align_vq_print_header "ScienceQA 预处理完成"
echo "输出文件: ${SCIENCEQA_PREPROCESSED_PATH}"
