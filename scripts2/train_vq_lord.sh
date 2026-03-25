#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align}"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "train_vq_lord"

align_vq_print_header "VQ-LoRD 全流程"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SCIENCEQA_PATH: ${SCIENCEQA_PATH}"
echo "LOG_FILE: ${LOG_FILE}"

PIPELINE_STEPS=(
    "teacher_model_data_collect.sh"
    "data_preprocess.sh"
    "run_stage1.sh"
    "run_stage2.sh"
    "run_stage3.sh"
    "test_origin.sh"
    "test_vq_lord_stage2.sh"
    "test_vq_lord_stage3.sh"
)

for step_script in "${PIPELINE_STEPS[@]}"; do
    align_vq_print_header "执行 ${step_script}"
    bash "${SCRIPT_SOURCE_DIR}/${step_script}"
done

align_vq_print_header "VQ-LoRD 全流程完成"
