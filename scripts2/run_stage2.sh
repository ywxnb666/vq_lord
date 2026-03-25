#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align}"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

align_vq_init_paths
align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "run_stage2"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Paths
TRAIN_ENTRY="${ROOT_DIR}/vq_lord3/train_vq_lord3.py"
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/sciqa_process.py"
SAVE_PATH="${CKPT_DIR}"
STAGE1_CODEBOOK_PATH="${STAGE1_CODEBOOK_PATH:-${CKPT_DIR}/stage1_vq/vq_codebook.pt}"
STAGE2_CKPT_PATH="${STAGE2_CKPT_PATH:-${CKPT_DIR}/stage2_vision}"

# Data
DATASET_NAME="scienceqa"
SCIENCEQA_SPLIT="${SCIENCEQA_SPLIT:-train}"
TRAIN_NUM="${TRAIN_NUM:-0}"
SCIENCEQA_SEED="${SCIENCEQA_SEED:-20240306}"
BUCKET_BY="${BUCKET_BY:-patches}"
PREPROCESS_BUCKET_BATCH_SIZE="${PREPROCESS_BUCKET_BATCH_SIZE:-8}"
BUCKET_BATCH_SIZE="${BUCKET_BATCH_SIZE:-8}"
STAGE3_BUCKET_BATCH_SIZE="${STAGE3_BUCKET_BATCH_SIZE:-8}"
BUCKET_DROP_LAST="${BUCKET_DROP_LAST:-0}"
DISABLE_BUCKET_FOR_STAGE3="${DISABLE_BUCKET_FOR_STAGE3:-0}"
SCIENCEQA_PREPROCESSED_PATH="${SCIENCEQA_PREPROCESSED_PATH:-${PREPROCESS_DIR}/scienceqa_${SCIENCEQA_SPLIT}_n${TRAIN_NUM}_seed${SCIENCEQA_SEED}_${BUCKET_BY}_bs${PREPROCESS_BUCKET_BATCH_SIZE}.json}"

# Stage2 latest H200 config
STAGE="2"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-4}"
STAGE3_GRAD_ACCUM="${STAGE3_GRAD_ACCUM:-0}"
LR="${LR:-3e-5}"
STAGE1_LR="${STAGE1_LR:-5e-5}"
STAGE1_RECON_WEIGHT="${STAGE1_RECON_WEIGHT:-1.0}"
STAGE1_COSINE_WEIGHT="${STAGE1_COSINE_WEIGHT:-0.25}"
STAGE1_VQ_WEIGHT="${STAGE1_VQ_WEIGHT:-1.0}"
STAGE1_GRAD_CLIP="${STAGE1_GRAD_CLIP:-5.0}"
STAGE2_ANSWER_WEIGHT="${STAGE2_ANSWER_WEIGHT:-1.0}"
STAGE2_RATIONALE_WEIGHT="${STAGE2_RATIONALE_WEIGHT:-0.2}"
STAGE2_PREPOST_LR_SCALE="${STAGE2_PREPOST_LR_SCALE:-0.2}"
STAGE2_VISION_LR_SCALE="${STAGE2_VISION_LR_SCALE:-0.2}"
STAGE2_GRAD_CLIP="${STAGE2_GRAD_CLIP:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

# VQ / model
VICTIM_MODEL="${VICTIM_MODEL:-qwen3.5-flash-2026-02-23}"
TEACHER_API_BASE="${TEACHER_API_BASE:-${OPENAI_BASE_URL:-}}"
TEACHER_API_KEY="${TEACHER_API_KEY:-${OPENAI_API_KEY:-}}"
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
COLLECT_TEACHER_DATA="${COLLECT_TEACHER_DATA:-0}"
STRICT_TEACHER_DISTILL="${STRICT_TEACHER_DISTILL:-0}"
TEACHER_LANG="${TEACHER_LANG:-en}"
REUSE_VQ_CODEBOOK="${REUSE_VQ_CODEBOOK:-1}"
REUSE_STAGE2="${REUSE_STAGE2:-0}"

# Evaluation
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-500}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-64}"
EVAL_ANSWER_MODE="${EVAL_ANSWER_MODE:-logits}"
RESULT_PATH="${RESULT_PATH:-${TEST_RESULT_DIR}/stage2_validation_logits.json}"

# Logging / save
LOG_STEP="${LOG_STEP:-50}"
SAVE_STEP="${SAVE_STEP:-100}"
SAVE_EACH_EPOCH="${SAVE_EACH_EPOCH:-1}"

align_vq_print_header "Stage2 训练"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "TRAIN_ENTRY: ${TRAIN_ENTRY}"
echo "EVAL_ENTRY: ${EVAL_ENTRY}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SCIENCEQA_PATH: ${SCIENCEQA_PATH}"
echo "SCIENCEQA_PREPROCESSED_PATH: ${SCIENCEQA_PREPROCESSED_PATH}"
echo "STAGE1_CODEBOOK_PATH: ${STAGE1_CODEBOOK_PATH}"
echo "STAGE2_CKPT_PATH: ${STAGE2_CKPT_PATH}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "EPOCHS: ${EPOCHS}"
echo "LR: ${LR}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "GRAD_ACCUM/STAGE2_GRAD_ACCUM: ${GRAD_ACCUM}/${STAGE2_GRAD_ACCUM}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${TRAIN_ENTRY}" "Stage2 训练入口"
align_vq_require_file "${EVAL_ENTRY}" "Stage2 评测入口"
align_vq_require_stage1_artifacts "${STAGE1_CODEBOOK_PATH}"
align_vq_prepare_scienceqa_preprocess \
    "${SCIENCEQA_PREPROCESSED_PATH}" \
    "${SCIENCEQA_SPLIT}" \
    "${TRAIN_NUM}" \
    "${SCIENCEQA_SEED}" \
    "${BUCKET_BY}" \
    "${PREPROCESS_BUCKET_BATCH_SIZE}" \
    "${BUCKET_DROP_LAST}" \
    "1" \
    "10" \
    "10"

"${PYTHON_BIN}" "${TRAIN_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --victim_model="${VICTIM_MODEL}" \
    --teacher_api_base="${TEACHER_API_BASE}" \
    --teacher_api_key="${TEACHER_API_KEY}" \
    --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
    --vq_commitment_cost="${VQ_COMMITMENT_COST}" \
    --vq_dead_code_threshold="${VQ_DEAD_CODE_THRESHOLD}" \
    --vq_usage_decay="${VQ_USAGE_DECAY}" \
    --vq_dead_code_reset_interval="${VQ_DEAD_CODE_RESET_INTERVAL}" \
    --vq_legacy_loss="${VQ_LEGACY_LOSS}" \
    --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    --beta="${BETA}" \
    --temperature="${TEMPERATURE}" \
    --tau1="${TAU1}" \
    --stage="${STAGE}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --lr="${LR}" \
    --stage1_lr="${STAGE1_LR}" \
    --stage1_recon_weight="${STAGE1_RECON_WEIGHT}" \
    --stage1_cosine_weight="${STAGE1_COSINE_WEIGHT}" \
    --stage1_vq_weight="${STAGE1_VQ_WEIGHT}" \
    --stage1_grad_clip="${STAGE1_GRAD_CLIP}" \
    --max_length="${MAX_LENGTH}" \
    --use_lora="${USE_LORA}" \
    --lora_rank="${LORA_RANK}" \
    --lora_alpha="${LORA_ALPHA}" \
    --use_4bit="${USE_4BIT}" \
    --model_dtype="${MODEL_DTYPE}" \
    --grad_accum="${GRAD_ACCUM}" \
    --stage2_grad_accum="${STAGE2_GRAD_ACCUM}" \
    --stage2_answer_weight="${STAGE2_ANSWER_WEIGHT}" \
    --stage2_rationale_weight="${STAGE2_RATIONALE_WEIGHT}" \
    --stage2_prepost_lr_scale="${STAGE2_PREPOST_LR_SCALE}" \
    --stage2_vision_lr_scale="${STAGE2_VISION_LR_SCALE}" \
    --stage2_grad_clip="${STAGE2_GRAD_CLIP}" \
    --stage3_grad_accum="${STAGE3_GRAD_ACCUM}" \
    --stage3_lr_scale="0.2" \
    --stage3_train_projector="0" \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --data_dir="${DATA_DIR}" \
    --train_num="${TRAIN_NUM}" \
    --dataset_name="${DATASET_NAME}" \
    --scienceqa_path="${SCIENCEQA_PATH}" \
    --scienceqa_split="${SCIENCEQA_SPLIT}" \
    --scienceqa_seed="${SCIENCEQA_SEED}" \
    --scienceqa_preprocessed_path="${SCIENCEQA_PREPROCESSED_PATH}" \
    --bucket_batch_size="${BUCKET_BATCH_SIZE}" \
    --stage3_bucket_batch_size="${STAGE3_BUCKET_BATCH_SIZE}" \
    --disable_bucket_for_stage3="${DISABLE_BUCKET_FOR_STAGE3}" \
    --collect_teacher_data="${COLLECT_TEACHER_DATA}" \
    --strict_teacher_distill="${STRICT_TEACHER_DISTILL}" \
    --teacher_lang="${TEACHER_LANG}" \
    --reuse_vq_codebook="${REUSE_VQ_CODEBOOK}" \
    --reuse_stage2="${REUSE_STAGE2}" \
    --vq_codebook_path="${STAGE1_CODEBOOK_PATH}" \
    --stage2_ckpt_path="${STAGE2_CKPT_PATH}" \
    --save_path="${SAVE_PATH}" \
    --log_step="${LOG_STEP}" \
    --save_step="${SAVE_STEP}" \
    --save_each_epoch="${SAVE_EACH_EPOCH}" \
    --device="cuda"

align_vq_require_stage2_artifacts "${STAGE2_CKPT_PATH}"
mkdir -p "$(dirname "${RESULT_PATH}")"

"${PYTHON_BIN}" "${EVAL_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --adapter_path="${STAGE2_CKPT_PATH}" \
    --scienceqa_path="${SCIENCEQA_PATH}" \
    --split="${EVAL_SPLIT}" \
    --max_samples="${EVAL_MAX_SAMPLES}" \
    --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    --use_4bit="${USE_4BIT}" \
    --use_vq=1 \
    --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
    --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    --vq_codebook_path="${STAGE2_CKPT_PATH}/vq_codebook.pt" \
    --answer_mode="${EVAL_ANSWER_MODE}" \
    --save_path="${RESULT_PATH}"

align_vq_require_file "${RESULT_PATH}" "Stage2 validation 结果"
eval "$(align_vq_extract_eval_metrics "${RESULT_PATH}")"

align_vq_print_header "Stage2 训练完成"
echo "ACCURACY=${ACCURACY}"
echo "FORMAT_RATE=${FORMAT_RATE}"
echo "N=${N}"
echo "Stage2 checkpoint: ${STAGE2_CKPT_PATH}"
echo "Validation 结果: ${RESULT_PATH}"
