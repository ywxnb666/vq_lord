#!/bin/bash
set -euo pipefail

# ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align}"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "test_vq_lord_stage2"

# Evaluation
EVAL_SPLIT="test"
EVAL_MAX_SAMPLES=0
EVAL_MAX_NEW_TOKENS=128
EVAL_ANSWER_MODE="logits"
USE_4BIT=0
USE_VQ=0
VQ_CODEBOOK_SIZE=1024
FREEZE_VISION_TOWER=0

# Paths
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/sciqa_process.py"
STAGE2_CKPT_PATH="${STAGE2_CKPT_PATH:-${CKPT_DIR}/stage2/stage2_vision_epoch7}"
# STAGE2_CKPT_PATH="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts_stage2_tune/round1_e3/stage2_vision_epoch15"
RESULT_PATH="${RESULT_PATH:-${TEST_RESULT_DIR}/stage2_${EVAL_SPLIT}_${EVAL_ANSWER_MODE}_vq${USE_VQ}.json}"

align_vq_print_header "Stage2 产物评测"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SCIENCEQA_PATH: ${SCIENCEQA_PATH}"
echo "STAGE2_CKPT_PATH: ${STAGE2_CKPT_PATH}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${EVAL_ENTRY}" "评测入口"
align_vq_require_stage2_artifacts "${STAGE2_CKPT_PATH}"
align_vq_assert_scienceqa_path
mkdir -p "$(dirname "${RESULT_PATH}")"

"${PYTHON_BIN}" "${EVAL_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --student_model_type="${STUDENT_MODEL_TYPE}" \
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
    --save_path="${RESULT_PATH}"

align_vq_require_file "${RESULT_PATH}" "Stage2 test 结果"
eval "$(align_vq_extract_eval_metrics "${RESULT_PATH}")"

align_vq_print_header "Stage2 产物评测完成"
echo "ACCURACY=${ACCURACY}"
echo "FORMAT_RATE=${FORMAT_RATE}"
echo "N=${N}"
echo "结果文件: ${RESULT_PATH}"
