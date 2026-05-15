#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/songxinhao/workspace/vq_lord"
DATA_DIR="${ROOT_DIR}/vq_lord_data"
PREPROCESS_DIR="${DATA_DIR}/preprocess"
CKPT_ROOT="${ROOT_DIR}/vq_lord_ckpts"

QWEN_MODEL="/mnt/shared-storage-gpfs2/evoagi-share-gpfs2/xhsong/models/Qwen2-VL-7B-Instruct"
LLAVA_MODEL="/mnt/shared-storage-gpfs2/evoagi-share-gpfs2/xhsong/models/llava-v1.6-mistral-7b-hf"

SCIENCEQA_PATH="/mnt/shared-storage-gpfs2/evoagi-share-gpfs2/xhsong/datasets/ScienceQA"
AOKVQA_PATH="/mnt/shared-storage-gpfs2/evoagi-share-gpfs2/xhsong/datasets/A-OKVQA"
ICONQA_PATH="/mnt/shared-storage-gpfs2/evoagi-share-gpfs2/xhsong/datasets/IconQA"

SCIENCEQA_GEMINI_CACHE="${DATA_DIR}/scienceqa_teacher_gemini2.5-pro_train_n6190_seed20240306_aligned.json"
AOKVQA_GEMINI_CACHE="${DATA_DIR}/aokvqa_teacher_gemini-2.5-pro_train_n4283_mapped_valid.json"
ICONQA_GEMINI_CACHE="${DATA_DIR}/iconqa_choose_txt_gemini2.5-pro_n6199_aligned.json"

if [ ! -f "${SCIENCEQA_GEMINI_CACHE}" ]; then
  echo "缺少 ScienceQA gemini 教师文件: ${SCIENCEQA_GEMINI_CACHE}"
  exit 1
fi
if [ ! -f "${AOKVQA_GEMINI_CACHE}" ]; then
  echo "缺少 A-OKVQA gemini 教师文件: ${AOKVQA_GEMINI_CACHE}"
  exit 1
fi
if [ ! -f "${ICONQA_GEMINI_CACHE}" ]; then
  echo "缺少 IconQA gemini 教师文件: ${ICONQA_GEMINI_CACHE}"
  exit 1
fi

mkdir -p "${PREPROCESS_DIR}"

# echo "===== qwen / scienceqa / gemini-2.5-pro: stage2 ====="
# DATASET_NAME="scienceqa" \
# DATASET_TAG="scienceqa" \
# TRAIN_DATASET_NAME="scienceqa" \
# DATASET_PATH="${SCIENCEQA_PATH}" \
# DATASET_SPLIT="train" \
# TRAIN_NUM="6190" \
# VICTIM_MODEL="gemini2.5-pro" \
# TEACHER_CACHE_PATH="${SCIENCEQA_GEMINI_CACHE}" \
# SAMPLE_ONLY_CACHED_TEACHER="1" \
# STUDENT_MODEL_TYPE="qwen2_vl" \
# MODEL_PATH="${QWEN_MODEL}" \
# EPOCHS="4" \
# REUSE_STAGE2="0" \
# STAGE2_RESUME_PATH="" \
# SAVE_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage2" \
# STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage2/stage2_vision" \
# STAGE2_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage2/stage2_resume_latest" \
# RESULT_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage2/stage2_validation.json" \
# LOG_FILE="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage2/run_stage2.log" \
# bash "${ROOT_DIR}/scripts2/run_stage2.sh"

# echo "===== qwen / scienceqa / gemini-2.5-pro: stage3 ====="
# DATASET_NAME="scienceqa" \
# DATASET_TAG="scienceqa" \
# TRAIN_DATASET_NAME="scienceqa" \
# DATASET_PATH="${SCIENCEQA_PATH}" \
# DATASET_SPLIT="train" \
# TRAIN_NUM="6190" \
# VICTIM_MODEL="gemini2.5-pro" \
# TEACHER_CACHE_PATH="${SCIENCEQA_GEMINI_CACHE}" \
# SAMPLE_ONLY_CACHED_TEACHER="1" \
# STUDENT_MODEL_TYPE="qwen2_vl" \
# MODEL_PATH="${QWEN_MODEL}" \
# EPOCHS="15" \
# PERIOD_NUM="15" \
# REUSE_STAGE2="1" \
# STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage2/stage2_vision" \
# SAVE_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage3" \
# STAGE3_FINAL_ADAPTER_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage3/stage3_lord_final" \
# STAGE3_RESUME_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage3/stage3_resume_latest" \
# STAGE3_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage3/stage3_resume_latest" \
# STAGE3_VALIDATION_RESULT_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage3/stage3_validation.json" \
# STAGE3_TEST_RESULT_PATH="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage3/stage3_test.json" \
# LOG_FILE="${CKPT_ROOT}/stu-qwen/scienceqa/gemini-2.5-pro/stage3/run_stage3.log" \
# bash "${ROOT_DIR}/scripts2/run_stage3.sh"
# rm -f "${PREPROCESS_DIR}"/*.json

# echo "===== qwen / aokvqa / gemini-2.5-pro: stage2 ====="
# DATASET_NAME="aokvqa" \
# DATASET_TAG="aokvqa" \
# TRAIN_DATASET_NAME="aokvqa" \
# DATASET_PATH="${AOKVQA_PATH}" \
# DATASET_SPLIT="train" \
# TRAIN_NUM="4283" \
# VICTIM_MODEL="gemini-2.5-pro" \
# TEACHER_CACHE_PATH="${AOKVQA_GEMINI_CACHE}" \
# SAMPLE_ONLY_CACHED_TEACHER="1" \
# STUDENT_MODEL_TYPE="qwen2_vl" \
# MODEL_PATH="${QWEN_MODEL}" \
# EPOCHS="4" \
# REUSE_STAGE2="0" \
# STAGE2_RESUME_PATH="" \
# SAVE_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage2" \
# STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage2/stage2_vision" \
# STAGE2_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage2/stage2_resume_latest" \
# RESULT_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage2/stage2_validation.json" \
# LOG_FILE="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage2/run_stage2.log" \
# bash "${ROOT_DIR}/scripts2/run_stage2.sh"

echo "===== qwen / aokvqa / gemini-2.5-pro: stage3 ====="
DATASET_NAME="aokvqa" \
DATASET_TAG="aokvqa" \
TRAIN_DATASET_NAME="aokvqa" \
DATASET_PATH="${AOKVQA_PATH}" \
DATASET_SPLIT="train" \
TRAIN_NUM="4283" \
VICTIM_MODEL="gemini-2.5-pro" \
TEACHER_CACHE_PATH="${AOKVQA_GEMINI_CACHE}" \
SAMPLE_ONLY_CACHED_TEACHER="1" \
STUDENT_MODEL_TYPE="qwen2_vl" \
MODEL_PATH="${QWEN_MODEL}" \
EPOCHS="15" \
PERIOD_NUM="15" \
REUSE_STAGE2="1" \
STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage2/stage2_vision" \
SAVE_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage3" \
STAGE3_FINAL_ADAPTER_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage3/stage3_lord_final" \
STAGE3_RESUME_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage3/no_resume" \
STAGE3_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage3/stage3_resume_latest" \
STAGE3_VALIDATION_RESULT_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage3/stage3_validation.json" \
STAGE3_TEST_RESULT_PATH="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage3/stage3_test.json" \
LOG_FILE="${CKPT_ROOT}/stu-qwen/a-okvqa/gemini-2.5-pro/stage3/run_stage3.log" \
bash "${ROOT_DIR}/scripts2/run_stage3.sh"
rm -f "${PREPROCESS_DIR}"/*.json

# echo "===== llava / scienceqa / gemini-2.5-pro: stage2 ====="
# DATASET_NAME="scienceqa" \
# DATASET_TAG="scienceqa" \
# TRAIN_DATASET_NAME="scienceqa" \
# DATASET_PATH="${SCIENCEQA_PATH}" \
# DATASET_SPLIT="train" \
# TRAIN_NUM="6190" \
# VICTIM_MODEL="gemini2.5-pro" \
# TEACHER_CACHE_PATH="${SCIENCEQA_GEMINI_CACHE}" \
# SAMPLE_ONLY_CACHED_TEACHER="1" \
# STUDENT_MODEL_TYPE="llava_next" \
# MODEL_PATH="${LLAVA_MODEL}" \
# EPOCHS="4" \
# REUSE_STAGE2="0" \
# STAGE2_RESUME_PATH="" \
# SAVE_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage2" \
# STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage2/stage2_vision" \
# STAGE2_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage2/stage2_resume_latest" \
# RESULT_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage2/stage2_validation.json" \
# LOG_FILE="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage2/run_stage2.log" \
# bash "${ROOT_DIR}/scripts2/run_stage2.sh"

echo "===== llava / scienceqa / gemini-2.5-pro: stage3 ====="
DATASET_NAME="scienceqa" \
DATASET_TAG="scienceqa" \
TRAIN_DATASET_NAME="scienceqa" \
DATASET_PATH="${SCIENCEQA_PATH}" \
DATASET_SPLIT="train" \
TRAIN_NUM="6190" \
VICTIM_MODEL="gemini2.5-pro" \
TEACHER_CACHE_PATH="${SCIENCEQA_GEMINI_CACHE}" \
SAMPLE_ONLY_CACHED_TEACHER="1" \
STUDENT_MODEL_TYPE="llava_next" \
MODEL_PATH="${LLAVA_MODEL}" \
EPOCHS="15" \
PERIOD_NUM="15" \
REUSE_STAGE2="1" \
STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage2/stage2_vision" \
SAVE_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage3" \
STAGE3_FINAL_ADAPTER_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage3/stage3_lord_final" \
STAGE3_RESUME_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage3/no_resume" \
STAGE3_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage3/stage3_resume_latest" \
STAGE3_VALIDATION_RESULT_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage3/stage3_validation.json" \
STAGE3_TEST_RESULT_PATH="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage3/stage3_test.json" \
LOG_FILE="${CKPT_ROOT}/stu-llava/scienceqa/gemini-2.5-pro/stage3/run_stage3.log" \
bash "${ROOT_DIR}/scripts2/run_stage3.sh"
rm -f "${PREPROCESS_DIR}"/*.json

echo "===== llava / aokvqa / gemini-2.5-pro: stage2 ====="
DATASET_NAME="aokvqa" \
DATASET_TAG="aokvqa" \
TRAIN_DATASET_NAME="aokvqa" \
DATASET_PATH="${AOKVQA_PATH}" \
DATASET_SPLIT="train" \
TRAIN_NUM="4284" \
VICTIM_MODEL="gemini-2.5-pro" \
TEACHER_CACHE_PATH="${AOKVQA_GEMINI_CACHE}" \
SAMPLE_ONLY_CACHED_TEACHER="1" \
STUDENT_MODEL_TYPE="llava_next" \
MODEL_PATH="${LLAVA_MODEL}" \
EPOCHS="4" \
REUSE_STAGE2="0" \
STAGE2_RESUME_PATH="" \
SAVE_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage2" \
STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage2/stage2_vision" \
STAGE2_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage2/stage2_resume_latest" \
RESULT_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage2/stage2_validation.json" \
LOG_FILE="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage2/run_stage2.log" \
bash "${ROOT_DIR}/scripts2/run_stage2.sh"

echo "===== llava / aokvqa / gemini-2.5-pro: stage3 ====="
DATASET_NAME="aokvqa" \
DATASET_TAG="aokvqa" \
TRAIN_DATASET_NAME="aokvqa" \
DATASET_PATH="${AOKVQA_PATH}" \
DATASET_SPLIT="train" \
TRAIN_NUM="4284" \
VICTIM_MODEL="gemini-2.5-pro" \
TEACHER_CACHE_PATH="${AOKVQA_GEMINI_CACHE}" \
SAMPLE_ONLY_CACHED_TEACHER="1" \
STUDENT_MODEL_TYPE="llava_next" \
MODEL_PATH="${LLAVA_MODEL}" \
EPOCHS="15" \
PERIOD_NUM="15" \
REUSE_STAGE2="1" \
STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage2/stage2_vision" \
SAVE_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage3" \
STAGE3_FINAL_ADAPTER_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage3/stage3_lord_final" \
STAGE3_RESUME_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage3/no_resume" \
STAGE3_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage3/stage3_resume_latest" \
STAGE3_VALIDATION_RESULT_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage3/stage3_validation.json" \
STAGE3_TEST_RESULT_PATH="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage3/stage3_test.json" \
LOG_FILE="${CKPT_ROOT}/stu-llava/a-okvqa/gemini-2.5-pro/stage3/run_stage3.log" \
bash "${ROOT_DIR}/scripts2/run_stage3.sh"
rm -f "${PREPROCESS_DIR}"/*.json

echo "===== qwen / iconqa / gemini-2.5-pro: stage2 ====="
DATASET_NAME="iconqa" \
DATASET_TAG="iconqa" \
TRAIN_DATASET_NAME="iconqa" \
DATASET_PATH="${ICONQA_PATH}" \
DATASET_SPLIT="val" \
TRAIN_NUM="6199" \
VICTIM_MODEL="gemini2.5-pro" \
TEACHER_CACHE_PATH="${ICONQA_GEMINI_CACHE}" \
SAMPLE_ONLY_CACHED_TEACHER="1" \
STUDENT_MODEL_TYPE="qwen2_vl" \
MODEL_PATH="${QWEN_MODEL}" \
EPOCHS="4" \
REUSE_STAGE2="0" \
STAGE2_RESUME_PATH="" \
SAVE_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage2" \
STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage2/stage2_vision" \
STAGE2_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage2/stage2_resume_latest" \
RESULT_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage2/stage2_validation.json" \
LOG_FILE="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage2/run_stage2.log" \
bash "${ROOT_DIR}/scripts2/run_stage2.sh"

echo "===== qwen / iconqa / gemini-2.5-pro: stage3 ====="
DATASET_NAME="iconqa" \
DATASET_TAG="iconqa" \
TRAIN_DATASET_NAME="iconqa" \
DATASET_PATH="${ICONQA_PATH}" \
DATASET_SPLIT="val" \
TRAIN_NUM="6199" \
VICTIM_MODEL="gemini2.5-pro" \
TEACHER_CACHE_PATH="${ICONQA_GEMINI_CACHE}" \
SAMPLE_ONLY_CACHED_TEACHER="1" \
STUDENT_MODEL_TYPE="qwen2_vl" \
MODEL_PATH="${QWEN_MODEL}" \
EPOCHS="15" \
PERIOD_NUM="15" \
REUSE_STAGE2="1" \
STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage2/stage2_vision" \
SAVE_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage3" \
STAGE3_FINAL_ADAPTER_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage3/stage3_lord_final" \
STAGE3_RESUME_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage3/no_resume" \
STAGE3_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage3/stage3_resume_latest" \
STAGE3_VALIDATION_RESULT_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage3/stage3_validation.json" \
STAGE3_TEST_RESULT_PATH="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage3/stage3_test.json" \
LOG_FILE="${CKPT_ROOT}/stu-qwen/iconqa/gemini-2.5-pro/stage3/run_stage3.log" \
bash "${ROOT_DIR}/scripts2/run_stage3.sh"
rm -f "${PREPROCESS_DIR}"/*.json

echo "===== llava / iconqa / gemini-2.5-pro: stage2 ====="
DATASET_NAME="iconqa" \
DATASET_TAG="iconqa" \
TRAIN_DATASET_NAME="iconqa" \
DATASET_PATH="${ICONQA_PATH}" \
DATASET_SPLIT="val" \
TRAIN_NUM="6199" \
VICTIM_MODEL="gemini2.5-pro" \
TEACHER_CACHE_PATH="${ICONQA_GEMINI_CACHE}" \
SAMPLE_ONLY_CACHED_TEACHER="1" \
STUDENT_MODEL_TYPE="llava_next" \
MODEL_PATH="${LLAVA_MODEL}" \
EPOCHS="4" \
REUSE_STAGE2="0" \
STAGE2_RESUME_PATH="" \
SAVE_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage2" \
STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage2/stage2_vision" \
STAGE2_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage2/stage2_resume_latest" \
RESULT_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage2/stage2_validation.json" \
LOG_FILE="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage2/run_stage2.log" \
bash "${ROOT_DIR}/scripts2/run_stage2.sh"

echo "===== llava / iconqa / gemini-2.5-pro: stage3 ====="
DATASET_NAME="iconqa" \
DATASET_TAG="iconqa" \
TRAIN_DATASET_NAME="iconqa" \
DATASET_PATH="${ICONQA_PATH}" \
DATASET_SPLIT="val" \
TRAIN_NUM="6199" \
VICTIM_MODEL="gemini2.5-pro" \
TEACHER_CACHE_PATH="${ICONQA_GEMINI_CACHE}" \
SAMPLE_ONLY_CACHED_TEACHER="1" \
STUDENT_MODEL_TYPE="llava_next" \
MODEL_PATH="${LLAVA_MODEL}" \
EPOCHS="15" \
PERIOD_NUM="15" \
REUSE_STAGE2="1" \
STAGE2_CKPT_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage2/stage2_vision" \
SAVE_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage3" \
STAGE3_FINAL_ADAPTER_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage3/stage3_lord_final" \
STAGE3_RESUME_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage3/no_resume" \
STAGE3_RESUME_SAVE_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage3/stage3_resume_latest" \
STAGE3_VALIDATION_RESULT_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage3/stage3_validation.json" \
STAGE3_TEST_RESULT_PATH="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage3/stage3_test.json" \
LOG_FILE="${CKPT_ROOT}/stu-llava/iconqa/gemini-2.5-pro/stage3/run_stage3.log" \
bash "${ROOT_DIR}/scripts2/run_stage3.sh"
rm -f "${PREPROCESS_DIR}"/*.json

rm -f "${PREPROCESS_DIR}"/*.json
bash "${ROOT_DIR}/scripts2/occupy_stage2.sh"
