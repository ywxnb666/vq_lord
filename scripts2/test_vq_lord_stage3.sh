#!/bin/bash
set -euo pipefail

# shellcheck source=./common.sh
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
# align_vq_ensure_runtime_dirs
align_vq_setup_logging "test_vq_lord_stage3"

# Evaluation
EVAL_SPLIT="test"
EVAL_MAX_SAMPLES=0      # 0 表示全量
EVAL_MAX_NEW_TOKENS=128
EVAL_ANSWER_MODE=logits     # logits, generate, hybrid
USE_4BIT=0
USE_VQ=1
VQ_CODEBOOK_SIZE=1024
FREEZE_VISION_TOWER=0

# Paths
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/sciqa_process2.py"
STAGE3_FINAL_ADAPTER_PATH="${STAGE3_FINAL_ADAPTER_PATH:-${CKPT_DIR}/stage3_sub1_period7}"
# STAGE3_FINAL_ADAPTER_PATH="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts_stage3_tune/run_20260323_140152/stage3_sub1_period7"
RESULT_PATH="${RESULT_PATH:-${TEST_RESULT_DIR}/stage3_${EVAL_SPLIT}_${EVAL_ANSWER_MODE}_vq${USE_VQ}.json}"

align_vq_print_header "Stage3 产物评测"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SCIENCEQA_PATH: ${SCIENCEQA_PATH}"
echo "STAGE3_FINAL_ADAPTER_PATH: ${STAGE3_FINAL_ADAPTER_PATH}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${EVAL_ENTRY}" "评测入口"
align_vq_require_dir "${STAGE3_FINAL_ADAPTER_PATH}" "Stage3 最终 adapter 目录"
align_vq_require_file "${STAGE3_FINAL_ADAPTER_PATH}/adapter_config.json" "Stage3 adapter_config.json"
align_vq_require_file "${STAGE3_FINAL_ADAPTER_PATH}/vq_codebook.pt" "Stage3 vq_codebook"
align_vq_assert_scienceqa_path
mkdir -p "$(dirname "${RESULT_PATH}")"

"${PYTHON_BIN}" "${EVAL_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --adapter_path="${STAGE3_FINAL_ADAPTER_PATH}" \
    --scienceqa_path="${SCIENCEQA_PATH}" \
    --split="${EVAL_SPLIT}" \
    --max_samples="${EVAL_MAX_SAMPLES}" \
    --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    --use_4bit="${USE_4BIT}" \
    --use_vq="${USE_VQ}" \
    --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
    --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    --vq_codebook_path="${STAGE3_FINAL_ADAPTER_PATH}/vq_codebook.pt" \
    --answer_mode="${EVAL_ANSWER_MODE}" \
    --save_path="${RESULT_PATH}"

align_vq_require_file "${RESULT_PATH}" "Stage3 test 结果"
eval "$(align_vq_extract_eval_metrics "${RESULT_PATH}")"

align_vq_print_header "Stage3 产物评测完成"
echo "ACCURACY=${ACCURACY}"
echo "FORMAT_RATE=${FORMAT_RATE}"
echo "N=${N}"
echo "结果文件: ${RESULT_PATH}"
