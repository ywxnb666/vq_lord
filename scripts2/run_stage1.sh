#!/bin/bash
set -euo pipefail

# ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align}"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DDP_NPROC="${DDP_NPROC:-8}"

# CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# DDP_NPROC="${DDP_NPROC:-4}"

# CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# DDP_NPROC="${DDP_NPROC:-2}"

align_vq_init_paths
align_vq_setup_env
align_vq_setup_distributed_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "run_stage1"

# Paths
TRAIN_ENTRY="${ROOT_DIR}/vq_lord3/training/train_vq_lord.py"
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/evaluation/sciqa_process2.py"
SAVE_PATH="${SAVE_PATH:-${CKPT_DIR}/stage1}"
STAGE0_CODEBOOK_PATH="${STAGE0_CODEBOOK_PATH:-${CKPT_DIR}/stage2_sub1_period7_1/vq_codebook.pt}"
STAGE1_CKPT_PATH="${STAGE1_CKPT_PATH:-${CKPT_DIR}/stage1_vision}"
STAGE1_RESUME_PATH="${STAGE1_RESUME_PATH:-}"
# STAGE1_RESUME_PATH="${STAGE1_RESUME_PATH:-${CKPT_DIR}/stage1_resume_latest}"
STAGE1_RESUME_SAVE_PATH="${STAGE1_RESUME_SAVE_PATH:-${CKPT_DIR}/stage1/stage1_resume_latest}"
STAGE1_RESUME_SAVE_OPTIMIZER="${STAGE1_RESUME_SAVE_OPTIMIZER:-1}"
STAGE1_RESUME_SAVE_INTERVAL="${STAGE1_RESUME_SAVE_INTERVAL:-1}"

# Data
STAGE2_BUCKET_BATCH_SIZE="4"

# Stage1 4x H200 config
# 保持与原 1 卡配置近似的有效 batch：8 * 4(accum) -> 8 * 4卡 * 1(accum)
STAGE="1"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PREPROCESS_BUCKET_BATCH_SIZE="${BATCH_SIZE}"
BUCKET_BATCH_SIZE="${BATCH_SIZE}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
STAGE1_GRAD_ACCUM="${STAGE1_GRAD_ACCUM:-2}"
STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-0}"
LR="${LR:-3e-5}"
STAGE0_LR="${STAGE0_LR:-5e-5}"
STAGE0_RECON_WEIGHT="${STAGE0_RECON_WEIGHT:-1.0}"
STAGE0_COSINE_WEIGHT="${STAGE0_COSINE_WEIGHT:-0.25}"
STAGE0_VQ_WEIGHT="${STAGE0_VQ_WEIGHT:-1.0}"
STAGE0_GRAD_CLIP="${STAGE0_GRAD_CLIP:-5.0}"
STAGE1_FIELD_WEIGHT_OBSERVED="${STAGE1_FIELD_WEIGHT_OBSERVED:-1.0}"
STAGE1_FIELD_WEIGHT_CONTEXT="${STAGE1_FIELD_WEIGHT_CONTEXT:-1.0}"
STAGE1_FIELD_WEIGHT_REASONING="${STAGE1_FIELD_WEIGHT_REASONING:-2.0}"
STAGE1_FIELD_WEIGHT_ANSWER="${STAGE1_FIELD_WEIGHT_ANSWER:-12.0}"
STAGE1_ANSWER_PREFIX_WEIGHT="${STAGE1_ANSWER_PREFIX_WEIGHT:-12.0}"
STAGE1_PREPOST_LR_SCALE="${STAGE1_PREPOST_LR_SCALE:-0.2}"
STAGE1_VISION_LR_SCALE="${STAGE1_VISION_LR_SCALE:-0.2}"
STAGE1_GRAD_CLIP="${STAGE1_GRAD_CLIP:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEACHER_OBSERVED_MAX_TOKENS="${TEACHER_OBSERVED_MAX_TOKENS:-192}"
TEACHER_CONTEXT_MAX_TOKENS="${TEACHER_CONTEXT_MAX_TOKENS:-160}"
TEACHER_REASONING_MAX_TOKENS="${TEACHER_REASONING_MAX_TOKENS:-96}"
TEACHER_ANSWER_MAX_TOKENS="${TEACHER_ANSWER_MAX_TOKENS:-64}"
align_vq_set_dataset_preprocessed_path "${PREPROCESS_BUCKET_BATCH_SIZE}"

# VQ / model
VQ_CODEBOOK_SIZE="${VQ_CODEBOOK_SIZE:-1024}"
VQ_COMMITMENT_COST="${VQ_COMMITMENT_COST:-0.25}"
VQ_DEAD_CODE_THRESHOLD="${VQ_DEAD_CODE_THRESHOLD:-1.0}"
VQ_USAGE_DECAY="${VQ_USAGE_DECAY:-0.995}"
VQ_DEAD_CODE_RESET_INTERVAL="${VQ_DEAD_CODE_RESET_INTERVAL:-10}"
VQ_LEGACY_LOSS="${VQ_LEGACY_LOSS:-0}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-0}"
BETA="${BETA:-0.05}"
TEMPERATURE="${TEMPERATURE:-1.5}"
TAU1="${TAU1:-0.01}"
USE_LORA="${USE_LORA:-1}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
USE_4BIT="${USE_4BIT:-0}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"

# Distillation / reuse
REUSE_STAGE1="0"

# Evaluation
if [ "${DATASET_NAME}" = "iconqa" ]; then
    EVAL_SPLIT="${EVAL_SPLIT:-test}"
else
    EVAL_SPLIT="${EVAL_SPLIT:-validation}"
fi
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-500}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_ANSWER_MODE="${EVAL_ANSWER_MODE:-generate_readable}"
RUN_STAGE1_EVAL="${RUN_STAGE1_EVAL:-0}"
RESULT_PATH="${RESULT_PATH:-${TEST_RESULT_DIR}/${DATASET_TAG}_stage1_validation_generate_readable.json}"

# Logging / save
LOG_STEP="${LOG_STEP:-1000000}"
SAVE_STEP="${SAVE_STEP:-100}"
SAVE_EACH_EPOCH="${SAVE_EACH_EPOCH:-1}"

align_vq_print_header "Stage1 训练"
# echo "ROOT_DIR: ${ROOT_DIR}"
# echo "TRAIN_ENTRY: ${TRAIN_ENTRY}"
# echo "EVAL_ENTRY: ${EVAL_ENTRY}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "DATASET_NAME: ${DATASET_NAME}"
# echo "DATASET_TAG: ${DATASET_TAG}"
# echo "TRAIN_DATASET_NAME: ${TRAIN_DATASET_NAME}"
echo "DATASET_PATH: ${DATASET_PATH}"
# echo "DATASET_PREPROCESSED_PATH: ${DATASET_PREPROCESSED_PATH}"
# echo "STAGE0_CODEBOOK_PATH: ${STAGE0_CODEBOOK_PATH}"
echo "STAGE1_CKPT_PATH: ${STAGE1_CKPT_PATH}"
echo "STAGE1_RESUME_PATH: ${STAGE1_RESUME_PATH:-<empty>}"
echo "STAGE1_RESUME_SAVE_PATH: ${STAGE1_RESUME_SAVE_PATH}"
# echo "REMOVE_VQ_CODEBOOK: ${REMOVE_VQ_CODEBOOK}"
echo "TEACHER_CACHE_PATH: ${TEACHER_CACHE_PATH:-<empty>}"
echo "SAMPLE_ONLY_CACHED_TEACHER: ${SAMPLE_ONLY_CACHED_TEACHER}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "RUN_STAGE1_EVAL: ${RUN_STAGE1_EVAL}"
echo "EPOCHS: ${EPOCHS}"
echo "LR: ${LR}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "GRAD_ACCUM/STAGE1_GRAD_ACCUM: ${GRAD_ACCUM}/${STAGE1_GRAD_ACCUM}"
echo "MAX_LENGTH: ${MAX_LENGTH}"
echo "STAGE1_FIELD_WEIGHTS: observed=${STAGE1_FIELD_WEIGHT_OBSERVED}, context=${STAGE1_FIELD_WEIGHT_CONTEXT}, reasoning=${STAGE1_FIELD_WEIGHT_REASONING}, answer=${STAGE1_FIELD_WEIGHT_ANSWER}, answer_prefix=${STAGE1_ANSWER_PREFIX_WEIGHT}"
echo "TEACHER_FIELD_MAX_TOKENS: observed=${TEACHER_OBSERVED_MAX_TOKENS}, context=${TEACHER_CONTEXT_MAX_TOKENS}, reasoning=${TEACHER_REASONING_MAX_TOKENS}, answer=${TEACHER_ANSWER_MAX_TOKENS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "DDP_NPROC: ${DDP_NPROC}"
# echo "ALIGN_VQ_NCCL_DEBUG: ${ALIGN_VQ_NCCL_DEBUG:-0}"
# echo "TORCH_DDP_TIMEOUT_SEC: ${TORCH_DDP_TIMEOUT_SEC:-<unset>}"
# echo "TORCH_DISTRIBUTED_DEBUG: ${TORCH_DISTRIBUTED_DEBUG:-<unset>}"
# echo "NCCL_DEBUG: ${NCCL_DEBUG:-<unset>}"
# echo "NCCL_DEBUG_SUBSYS: ${NCCL_DEBUG_SUBSYS:-<unset>}"
# echo "NCCL_BLOCKING_WAIT/TORCH_NCCL_BLOCKING_WAIT: ${NCCL_BLOCKING_WAIT:-<unset>}/${TORCH_NCCL_BLOCKING_WAIT:-<unset>}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${TRAIN_ENTRY}" "Stage1 训练入口"
align_vq_require_file "${EVAL_ENTRY}" "Stage1 评测入口"
if [ "${REMOVE_VQ_CODEBOOK}" != "1" ]; then
    align_vq_require_stage0_artifacts "${STAGE0_CODEBOOK_PATH}"
fi
if [ -n "${STAGE1_RESUME_PATH}" ]; then
    align_vq_require_file "${STAGE1_RESUME_PATH}/stage1_resume_state.pt" "Stage1 resume state"
fi
align_vq_prepare_dataset_preprocess \
    "${DATASET_PREPROCESSED_PATH}" \
    "${DATASET_SPLIT}" \
    "${TRAIN_NUM}" \
    "${DATASET_SEED}" \
    "${BUCKET_BY}" \
    "${PREPROCESS_BUCKET_BATCH_SIZE}" \
    "${BUCKET_DROP_LAST}" \
    "1" \
    "10" \
    "10"

mkdir -p "${STAGE1_RESUME_SAVE_PATH}"

TRAIN_LAUNCHER=()
align_vq_make_train_launcher TRAIN_LAUNCHER

"${TRAIN_LAUNCHER[@]}" "${TRAIN_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --student_model_type="${STUDENT_MODEL_TYPE}" \
    --victim_model="${VICTIM_MODEL}" \
    --teacher_api_base="${TEACHER_API_BASE}" \
    --teacher_api_key="${TEACHER_API_KEY}" \
    --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
    --vq_commitment_cost="${VQ_COMMITMENT_COST}" \
    --vq_dead_code_threshold="${VQ_DEAD_CODE_THRESHOLD}" \
    --vq_usage_decay="${VQ_USAGE_DECAY}" \
    --vq_dead_code_reset_interval="${VQ_DEAD_CODE_RESET_INTERVAL}" \
    --vq_legacy_loss="${VQ_LEGACY_LOSS}" \
    --remove_vq_codebook="${REMOVE_VQ_CODEBOOK}" \
    --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    --beta="${BETA}" \
    --temperature="${TEMPERATURE}" \
    --tau1="${TAU1}" \
    --stage="${STAGE}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --lr="${LR}" \
    --stage0_lr="${STAGE0_LR}" \
    --stage0_recon_weight="${STAGE0_RECON_WEIGHT}" \
    --stage0_cosine_weight="${STAGE0_COSINE_WEIGHT}" \
    --stage0_vq_weight="${STAGE0_VQ_WEIGHT}" \
    --stage0_grad_clip="${STAGE0_GRAD_CLIP}" \
    --max_length="${MAX_LENGTH}" \
    --use_lora="${USE_LORA}" \
    --lora_rank="${LORA_RANK}" \
    --lora_alpha="${LORA_ALPHA}" \
    --use_4bit="${USE_4BIT}" \
    --model_dtype="${MODEL_DTYPE}" \
    --grad_accum="${GRAD_ACCUM}" \
    --stage1_grad_accum="${STAGE1_GRAD_ACCUM}" \
    --stage1_field_weight_observed="${STAGE1_FIELD_WEIGHT_OBSERVED}" \
    --stage1_field_weight_context="${STAGE1_FIELD_WEIGHT_CONTEXT}" \
    --stage1_field_weight_reasoning="${STAGE1_FIELD_WEIGHT_REASONING}" \
    --stage1_field_weight_answer="${STAGE1_FIELD_WEIGHT_ANSWER}" \
    --stage1_answer_prefix_weight="${STAGE1_ANSWER_PREFIX_WEIGHT}" \
    --stage1_prepost_lr_scale="${STAGE1_PREPOST_LR_SCALE}" \
    --stage1_vision_lr_scale="${STAGE1_VISION_LR_SCALE}" \
    --stage1_grad_clip="${STAGE1_GRAD_CLIP}" \
    --stage2_grad_accum="${STAGE2_GRAD_ACCUM}" \
    --stage2_lr_scale="0.2" \
    --stage2_train_projector="0" \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --data_dir="${DATA_DIR}" \
    --train_num="${TRAIN_NUM}" \
    --dataset_name="${TRAIN_DATASET_NAME}" \
    --scienceqa_path="${DATASET_PATH}" \
    --scienceqa_split="${DATASET_SPLIT}" \
    --scienceqa_seed="${DATASET_SEED}" \
    --scienceqa_preprocessed_path="${DATASET_PREPROCESSED_PATH}" \
    --bucket_batch_size="${BUCKET_BATCH_SIZE}" \
    --stage2_bucket_batch_size="${STAGE2_BUCKET_BATCH_SIZE}" \
    --disable_bucket_for_stage2="${DISABLE_BUCKET_FOR_STAGE2}" \
    --collect_teacher_data="${COLLECT_TEACHER_DATA}" \
    --strict_teacher_distill="${STRICT_TEACHER_DISTILL}" \
    --teacher_lang="${TEACHER_LANG}" \
    --teacher_cache_path="${TEACHER_CACHE_PATH}" \
    --teacher_observed_max_tokens="${TEACHER_OBSERVED_MAX_TOKENS}" \
    --teacher_context_max_tokens="${TEACHER_CONTEXT_MAX_TOKENS}" \
    --teacher_reasoning_max_tokens="${TEACHER_REASONING_MAX_TOKENS}" \
    --teacher_answer_max_tokens="${TEACHER_ANSWER_MAX_TOKENS}" \
    --sample_only_cached_teacher="${SAMPLE_ONLY_CACHED_TEACHER}" \
    --reuse_vq_codebook="${REUSE_VQ_CODEBOOK}" \
    --reuse_stage1="${REUSE_STAGE1}" \
    --vq_codebook_path="${STAGE0_CODEBOOK_PATH}" \
    --stage1_ckpt_path="${STAGE1_CKPT_PATH}" \
    --stage1_resume_path="${STAGE1_RESUME_PATH}" \
    --stage1_resume_save_path="${STAGE1_RESUME_SAVE_PATH}" \
    --stage1_resume_save_optimizer="${STAGE1_RESUME_SAVE_OPTIMIZER}" \
    --stage1_resume_save_interval="${STAGE1_RESUME_SAVE_INTERVAL}" \
    --save_path="${SAVE_PATH}" \
    --log_step="${LOG_STEP}" \
    --save_step="${SAVE_STEP}" \
    --save_each_epoch="${SAVE_EACH_EPOCH}" \
    --device="cuda"

align_vq_require_stage1_artifacts "${STAGE1_CKPT_PATH}"

if [ "${RUN_STAGE1_EVAL}" != "1" ]; then
    align_vq_print_header "Stage1 训练完成"
    echo "Stage1 checkpoint: ${STAGE1_CKPT_PATH}"
    echo "Validation skipped: RUN_STAGE1_EVAL=${RUN_STAGE1_EVAL}"
    exit 0
fi

mkdir -p "$(dirname "${RESULT_PATH}")"

EVAL_USE_VQ=1
if [ "${REMOVE_VQ_CODEBOOK}" = "1" ]; then
    EVAL_USE_VQ=0
fi

"${PYTHON_BIN}" "${EVAL_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --student_model_type="${STUDENT_MODEL_TYPE}" \
    --adapter_path="${STAGE1_CKPT_PATH}" \
    --dataset_name="${DATASET_NAME}" \
    --scienceqa_path="${DATASET_PATH}" \
    --split="${EVAL_SPLIT}" \
    --max_samples="${EVAL_MAX_SAMPLES}" \
    --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    --use_4bit="${USE_4BIT}" \
    --use_vq="${EVAL_USE_VQ}" \
    --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
    --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    --vq_codebook_path="${STAGE1_CKPT_PATH}/vq_codebook.pt" \
    --answer_mode="${EVAL_ANSWER_MODE}" \
    --save_path="${RESULT_PATH}"

align_vq_require_file "${RESULT_PATH}" "Stage1 validation 结果"
eval "$(align_vq_extract_eval_metrics "${RESULT_PATH}")"

align_vq_print_header "Stage1 训练完成"
echo "ACCURACY=${ACCURACY}"
echo "FORMAT_RATE=${FORMAT_RATE}"
echo "N=${N}"
echo "Stage1 checkpoint: ${STAGE1_CKPT_PATH}"
echo "Validation 结果: ${RESULT_PATH}"
