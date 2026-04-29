#!/bin/bash
set -euo pipefail

# ROOT_DIR="${ROOT_DIR:-/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align}"
SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DDP_NPROC="${DDP_NPROC:-4}"

# CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# DDP_NPROC="${DDP_NPROC:-2}"

# CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# DDP_NPROC="${DDP_NPROC:-1}"

align_vq_init_paths
align_vq_setup_env
align_vq_setup_distributed_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "run_stage3"

# Paths
TRAIN_ENTRY="${ROOT_DIR}/vq_lord3/train_vq_lord3.py"
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/sciqa_process2.py"
SAVE_PATH="${CKPT_DIR}/stage3"
STAGE1_CODEBOOK_PATH="${STAGE1_CODEBOOK_PATH:-${CKPT_DIR}/stage2/stage2_vision_epoch3/vq_codebook.pt}"
STAGE2_CKPT_PATH="${STAGE2_CKPT_PATH:-${CKPT_DIR}/stage2/stage2_vision_epoch3}"
STAGE3_FINAL_ADAPTER_PATH="${STAGE3_FINAL_ADAPTER_PATH:-${SAVE_PATH}/stage3_lord_final}"
STAGE3_RESUME_SAVE_PATH="${STAGE3_RESUME_SAVE_PATH:-${SAVE_PATH}/stage3_resume_latest}"
STAGE3_RESUME_PATH="${STAGE3_RESUME_PATH:-${STAGE3_RESUME_SAVE_PATH}}"

# Stage3 4x H200 config
# 保持与原 1 卡配置近似的有效 batch：2 * 4(accum) -> 2 * 4卡 * 1(accum)
STAGE="3"
EPOCHS="${EPOCHS:-50}"
SUB_STAGE_NUM="${SUB_STAGE_NUM:-1}"
PERIOD_NUM="${PERIOD_NUM:-50}"
SUB_SET_NUM="${SUB_SET_NUM:-0}"
PHASE_A_BATCH_SIZE="${PHASE_A_BATCH_SIZE:-8}"
PHASE_B_BATCH_SIZE="${PHASE_B_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-1}"
STAGE3_GRAD_ACCUM="${STAGE3_GRAD_ACCUM:-1}"
LR="${LR:-3e-5}"
STAGE3_LR_SCALE="${STAGE3_LR_SCALE:-0.2}"
STAGE3_GRAD_CLIP="${STAGE3_GRAD_CLIP:-1.0}"
STAGE3_TRAIN_PROJECTOR="${STAGE3_TRAIN_PROJECTOR:-0}"
TAU1="${TAU1:-0.001}"
TAU_DELTA="${TAU_DELTA:-0.005}"
TEMPERATURE="${TEMPERATURE:-1.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-${PHASE_B_BATCH_SIZE}}"
BUCKET_BATCH_SIZE="${BUCKET_BATCH_SIZE:-${PHASE_A_BATCH_SIZE}}"
STAGE3_BUCKET_BATCH_SIZE="${STAGE3_BUCKET_BATCH_SIZE:-${PHASE_A_BATCH_SIZE}}"
PREPROCESS_BUCKET_BATCH_SIZE="${PREPROCESS_BUCKET_BATCH_SIZE:-${BUCKET_BATCH_SIZE}}"
align_vq_set_dataset_preprocessed_path "${PREPROCESS_BUCKET_BATCH_SIZE}"

# Stage3 objective
STAGE3_EVAL_MAX_SAMPLES="${STAGE3_EVAL_MAX_SAMPLES:-2097}"
STAGE3_EVAL_EVERY_PERIOD="${STAGE3_EVAL_EVERY_PERIOD:-1}"
STAGE3_EVAL_SPLIT="${STAGE3_EVAL_SPLIT:-validation}"
STAGE3_EVAL_DATASET_PATH="${STAGE3_EVAL_DATASET_PATH:-}"
STAGE3_EVAL_TRAIN_NUM="${STAGE3_EVAL_TRAIN_NUM:-0}"
# STAGE3_EVAL_TRAIN_NUM="${STAGE3_EVAL_TRAIN_NUM:-320}"
STAGE3_EVAL_ANSWER_MODE="${STAGE3_EVAL_ANSWER_MODE:-logits}"
STAGE3_FIELD_WEIGHT_OBSERVED="${STAGE3_FIELD_WEIGHT_OBSERVED:-1.15}"
STAGE3_FIELD_WEIGHT_CONTEXT="${STAGE3_FIELD_WEIGHT_CONTEXT:-0.30}"
STAGE3_FIELD_WEIGHT_REASONING="${STAGE3_FIELD_WEIGHT_REASONING:-1.35}"
STAGE3_FIELD_WEIGHT_ANSWER="${STAGE3_FIELD_WEIGHT_ANSWER:-1.50}"
STAGE3_MC_WEIGHT="${STAGE3_MC_WEIGHT:-1.00}"
STAGE3_OBJ_WEIGHT="${STAGE3_OBJ_WEIGHT:-0.05}"
STAGE3_REG_WEIGHT="${STAGE3_REG_WEIGHT:-0.30}"
STAGE3_ANSWER_ANCHOR_WEIGHT="${STAGE3_ANSWER_ANCHOR_WEIGHT:-1.00}"
STAGE3_PAIR_USE_ANSWER_CORRECTNESS="${STAGE3_PAIR_USE_ANSWER_CORRECTNESS:-1}"
STAGE3_WRONG_IMAGE_ENABLE="${STAGE3_WRONG_IMAGE_ENABLE:-1}"
STAGE3_WRONG_IMAGE_WEIGHT="${STAGE3_WRONG_IMAGE_WEIGHT:-0.25}"
STAGE3_WRONG_IMAGE_MARGIN="${STAGE3_WRONG_IMAGE_MARGIN:-0.1}"
STAGE3_FORCE_COLD_START_PERIOD0="${STAGE3_FORCE_COLD_START_PERIOD0:-0}"

# VQ / model
VQ_CODEBOOK_SIZE="${VQ_CODEBOOK_SIZE:-1024}"
VQ_COMMITMENT_COST="${VQ_COMMITMENT_COST:-0.25}"
VQ_DEAD_CODE_THRESHOLD="${VQ_DEAD_CODE_THRESHOLD:-1.0}"
VQ_USAGE_DECAY="${VQ_USAGE_DECAY:-0.995}"
VQ_DEAD_CODE_RESET_INTERVAL="${VQ_DEAD_CODE_RESET_INTERVAL:-10}"
VQ_LEGACY_LOSS="${VQ_LEGACY_LOSS:-0}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-0}"
BETA="${BETA:-0.05}"
USE_LORA="${USE_LORA:-1}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
USE_4BIT="${USE_4BIT:-0}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"

# Distillation / cache
TEACHER_OBSERVED_MAX_TOKENS="${TEACHER_OBSERVED_MAX_TOKENS:-256}"
TEACHER_CONTEXT_MAX_TOKENS="${TEACHER_CONTEXT_MAX_TOKENS:-192}"
TEACHER_REASONING_MAX_TOKENS="${TEACHER_REASONING_MAX_TOKENS:-256}"
TEACHER_ANSWER_MAX_TOKENS="${TEACHER_ANSWER_MAX_TOKENS:-64}"
TEACHER_MAX_NEW_TOKENS_TOTAL="${TEACHER_MAX_NEW_TOKENS_TOTAL:-768}"
STAGE3_VIC_INCLUDE_CONTEXT="${STAGE3_VIC_INCLUDE_CONTEXT:-0}"
REUSE_STAGE2="${REUSE_STAGE2:-1}"
STAGE3_SAMPLE_CACHE_DATASET_PREFIX="${STAGE3_SAMPLE_CACHE_DATASET_PREFIX:-${DATASET_TAG}_}"
STAGE3_SAMPLE_CACHE_PATH="${STAGE3_SAMPLE_CACHE_PATH:-${DATA_DIR}/stage3/stage3_sample_cache_${STAGE3_SAMPLE_CACHE_DATASET_PREFIX}${VICTIM_MODEL}_${DATASET_SPLIT}_n${TRAIN_NUM}_seed${DATASET_SEED}_ml${MAX_LENGTH}_ctx${STAGE3_VIC_INCLUDE_CONTEXT}_obs${TEACHER_OBSERVED_MAX_TOKENS}_txt${TEACHER_CONTEXT_MAX_TOKENS}_rsn${TEACHER_REASONING_MAX_TOKENS}_ans${TEACHER_ANSWER_MAX_TOKENS}.pt}"

# Stage1/2 filler args required by parser
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

# Evaluation after training
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-2017}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-64}"
EVAL_ANSWER_MODE="${EVAL_ANSWER_MODE:-logits}"
STAGE3_VALIDATION_RESULT_PATH="${STAGE3_VALIDATION_RESULT_PATH:-${TEST_RESULT_DIR}/stage3_validation_logits.json}"
STAGE3_TEST_RESULT_PATH="${STAGE3_TEST_RESULT_PATH:-${TEST_RESULT_DIR}/stage3_test_logits.json}"

# Logging / save
LOG_STEP="${LOG_STEP:-1000000}"
SAVE_STEP="${SAVE_STEP:-0}"
SAVE_EACH_EPOCH="${SAVE_EACH_EPOCH:-1}"
STAGE3_RESUME_SAVE_OPTIMIZER="${STAGE3_RESUME_SAVE_OPTIMIZER:-1}"
STAGE3_RESUME_SAVE_INTERVAL="${STAGE3_RESUME_SAVE_INTERVAL:-1}"

align_vq_print_header "Stage3 训练"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "TRAIN_ENTRY: ${TRAIN_ENTRY}"
echo "EVAL_ENTRY: ${EVAL_ENTRY}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "DATASET_NAME: ${DATASET_NAME}"
echo "DATASET_TAG: ${DATASET_TAG}"
echo "TRAIN_DATASET_NAME: ${TRAIN_DATASET_NAME}"
echo "DATASET_PATH: ${DATASET_PATH}"
echo "DATASET_PREPROCESSED_PATH: ${DATASET_PREPROCESSED_PATH}"
echo "STAGE1_CODEBOOK_PATH: ${STAGE1_CODEBOOK_PATH}"
echo "STAGE2_CKPT_PATH: ${STAGE2_CKPT_PATH}"
echo "STAGE3_FINAL_ADAPTER_PATH: ${STAGE3_FINAL_ADAPTER_PATH}"
echo "STAGE3_SAMPLE_CACHE_PATH: ${STAGE3_SAMPLE_CACHE_PATH}"
echo "STAGE3_RESUME_PATH: ${STAGE3_RESUME_PATH:-<empty>}"
echo "STAGE3_RESUME_SAVE_PATH: ${STAGE3_RESUME_SAVE_PATH}"
echo "REMOVE_VQ_CODEBOOK: ${REMOVE_VQ_CODEBOOK}"
echo "TEACHER_CACHE_PATH: ${TEACHER_CACHE_PATH:-<empty>}"
echo "SAMPLE_ONLY_CACHED_TEACHER: ${SAMPLE_ONLY_CACHED_TEACHER}"
echo "EPOCHS/PERIOD_NUM: ${EPOCHS}/${PERIOD_NUM}"
echo "PHASE_A_BATCH_SIZE/PHASE_B_BATCH_SIZE: ${PHASE_A_BATCH_SIZE}/${BATCH_SIZE}"
echo "PREPROCESS_BUCKET_BATCH_SIZE: ${PREPROCESS_BUCKET_BATCH_SIZE}"
echo "LR: ${LR}"
echo "TAU1/TAU_DELTA: ${TAU1}/${TAU_DELTA}"
echo "STAGE3_MC_WEIGHT/STAGE3_OBJ_WEIGHT/STAGE3_REG_WEIGHT: ${STAGE3_MC_WEIGHT}/${STAGE3_OBJ_WEIGHT}/${STAGE3_REG_WEIGHT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "DDP_NPROC: ${DDP_NPROC}"
echo "LOG_FILE: ${LOG_FILE}"

align_vq_require_file "${TRAIN_ENTRY}" "Stage3 训练入口"
align_vq_require_file "${EVAL_ENTRY}" "Stage3 评测入口"
if [ "${REMOVE_VQ_CODEBOOK}" != "1" ]; then
    align_vq_require_stage1_artifacts "${STAGE1_CODEBOOK_PATH}"
fi
align_vq_require_stage2_artifacts "${STAGE2_CKPT_PATH}"
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

mkdir -p "$(dirname "${STAGE3_SAMPLE_CACHE_PATH}")" "${STAGE3_RESUME_SAVE_PATH}" "$(dirname "${STAGE3_VALIDATION_RESULT_PATH}")"

TRAIN_LAUNCHER=()
align_vq_make_train_launcher TRAIN_LAUNCHER

"${TRAIN_LAUNCHER[@]}" "${TRAIN_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --student_model_type="${STUDENT_MODEL_TYPE}" \
    --victim_model="${VICTIM_MODEL}" \
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
    --tau_delta="${TAU_DELTA}" \
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
    --stage3_lr_scale="${STAGE3_LR_SCALE}" \
    --stage3_grad_clip="${STAGE3_GRAD_CLIP}" \
    --stage3_train_projector="${STAGE3_TRAIN_PROJECTOR}" \
    --stage3_eval_max_samples="${STAGE3_EVAL_MAX_SAMPLES}" \
    --stage3_eval_every_period="${STAGE3_EVAL_EVERY_PERIOD}" \
    --stage3_eval_scienceqa_split="${STAGE3_EVAL_SPLIT}" \
    --stage3_eval_scienceqa_path="${STAGE3_EVAL_DATASET_PATH}" \
    --stage3_eval_train_num="${STAGE3_EVAL_TRAIN_NUM}" \
    --stage3_eval_answer_mode="${STAGE3_EVAL_ANSWER_MODE}" \
    --stage3_field_weight_observed="${STAGE3_FIELD_WEIGHT_OBSERVED}" \
    --stage3_field_weight_context="${STAGE3_FIELD_WEIGHT_CONTEXT}" \
    --stage3_field_weight_reasoning="${STAGE3_FIELD_WEIGHT_REASONING}" \
    --stage3_field_weight_answer="${STAGE3_FIELD_WEIGHT_ANSWER}" \
    --stage3_mc_weight="${STAGE3_MC_WEIGHT}" \
    --stage3_obj_weight="${STAGE3_OBJ_WEIGHT}" \
    --stage3_reg_weight="${STAGE3_REG_WEIGHT}" \
    --stage3_answer_anchor_weight="${STAGE3_ANSWER_ANCHOR_WEIGHT}" \
    --stage3_pair_use_answer_correctness="${STAGE3_PAIR_USE_ANSWER_CORRECTNESS}" \
    --stage3_wrong_image_enable="${STAGE3_WRONG_IMAGE_ENABLE}" \
    --stage3_wrong_image_weight="${STAGE3_WRONG_IMAGE_WEIGHT}" \
    --stage3_wrong_image_margin="${STAGE3_WRONG_IMAGE_MARGIN}" \
    --stage3_force_cold_start_period0="${STAGE3_FORCE_COLD_START_PERIOD0}" \
    --sub_stage_num="${SUB_STAGE_NUM}" \
    --period_num="${PERIOD_NUM}" \
    --sub_set_num="${SUB_SET_NUM}" \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --data_dir="${DATA_DIR}" \
    --train_num="${TRAIN_NUM}" \
    --dataset_name="${TRAIN_DATASET_NAME}" \
    --scienceqa_path="${DATASET_PATH}" \
    --scienceqa_split="${DATASET_SPLIT}" \
    --scienceqa_seed="${DATASET_SEED}" \
    --scienceqa_preprocessed_path="${DATASET_PREPROCESSED_PATH}" \
    --bucket_batch_size="${BUCKET_BATCH_SIZE}" \
    --stage3_bucket_batch_size="${STAGE3_BUCKET_BATCH_SIZE}" \
    --disable_bucket_for_stage3="${DISABLE_BUCKET_FOR_STAGE3}" \
    --collect_teacher_data="${COLLECT_TEACHER_DATA}" \
    --strict_teacher_distill="${STRICT_TEACHER_DISTILL}" \
    --teacher_lang="${TEACHER_LANG}" \
    --teacher_cache_path="${TEACHER_CACHE_PATH}" \
    --sample_only_cached_teacher="${SAMPLE_ONLY_CACHED_TEACHER}" \
    --teacher_observed_max_tokens="${TEACHER_OBSERVED_MAX_TOKENS}" \
    --teacher_context_max_tokens="${TEACHER_CONTEXT_MAX_TOKENS}" \
    --teacher_reasoning_max_tokens="${TEACHER_REASONING_MAX_TOKENS}" \
    --teacher_answer_max_tokens="${TEACHER_ANSWER_MAX_TOKENS}" \
    --teacher_max_new_tokens_total="${TEACHER_MAX_NEW_TOKENS_TOTAL}" \
    --stage3_vic_include_context="${STAGE3_VIC_INCLUDE_CONTEXT}" \
    --reuse_vq_codebook="${REUSE_VQ_CODEBOOK}" \
    --reuse_stage2="${REUSE_STAGE2}" \
    --vq_codebook_path="${STAGE1_CODEBOOK_PATH}" \
    --stage2_ckpt_path="${STAGE2_CKPT_PATH}" \
    --save_path="${SAVE_PATH}" \
    --log_step="${LOG_STEP}" \
    --save_step="${SAVE_STEP}" \
    --save_each_epoch="${SAVE_EACH_EPOCH}" \
    --stage3_resume_path="${STAGE3_RESUME_PATH}" \
    --stage3_resume_save_path="${STAGE3_RESUME_SAVE_PATH}" \
    --stage3_resume_save_optimizer="${STAGE3_RESUME_SAVE_OPTIMIZER}" \
    --stage3_resume_save_interval="${STAGE3_RESUME_SAVE_INTERVAL}" \
    --stage3_sample_cache_path="${STAGE3_SAMPLE_CACHE_PATH}" \
    --device="cuda"

align_vq_require_dir "${STAGE3_FINAL_ADAPTER_PATH}" "Stage3 最终 adapter 目录"
align_vq_require_file "${STAGE3_FINAL_ADAPTER_PATH}/adapter_config.json" "Stage3 adapter_config.json"
if [ "${REMOVE_VQ_CODEBOOK}" != "1" ]; then
    align_vq_require_file "${STAGE3_FINAL_ADAPTER_PATH}/vq_codebook.pt" "Stage3 vq_codebook"
fi

EVAL_USE_VQ=1
if [ "${REMOVE_VQ_CODEBOOK}" = "1" ]; then
    EVAL_USE_VQ=0
fi

"${PYTHON_BIN}" "${EVAL_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --student_model_type="${STUDENT_MODEL_TYPE}" \
    --adapter_path="${STAGE3_FINAL_ADAPTER_PATH}" \
    --dataset_name="${DATASET_NAME}" \
    --scienceqa_path="${DATASET_PATH}" \
    --split="validation" \
    --max_samples="${EVAL_MAX_SAMPLES}" \
    --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    --use_4bit="${USE_4BIT}" \
    --use_vq="${EVAL_USE_VQ}" \
    --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
    --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    --vq_codebook_path="${STAGE3_FINAL_ADAPTER_PATH}/vq_codebook.pt" \
    --answer_mode="${EVAL_ANSWER_MODE}" \
    --save_path="${STAGE3_VALIDATION_RESULT_PATH}"

"${PYTHON_BIN}" "${EVAL_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --student_model_type="${STUDENT_MODEL_TYPE}" \
    --adapter_path="${STAGE3_FINAL_ADAPTER_PATH}" \
    --dataset_name="${DATASET_NAME}" \
    --scienceqa_path="${DATASET_PATH}" \
    --split="test" \
    --max_samples="${EVAL_MAX_SAMPLES}" \
    --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    --use_4bit="${USE_4BIT}" \
    --use_vq="${EVAL_USE_VQ}" \
    --vq_codebook_size="${VQ_CODEBOOK_SIZE}" \
    --freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    --vq_codebook_path="${STAGE3_FINAL_ADAPTER_PATH}/vq_codebook.pt" \
    --answer_mode="${EVAL_ANSWER_MODE}" \
    --save_path="${STAGE3_TEST_RESULT_PATH}"

align_vq_require_file "${STAGE3_VALIDATION_RESULT_PATH}" "Stage3 validation 结果"
align_vq_require_file "${STAGE3_TEST_RESULT_PATH}" "Stage3 test 结果"
eval "$(align_vq_extract_eval_metrics "${STAGE3_VALIDATION_RESULT_PATH}")"
VAL_ACCURACY="${ACCURACY}"
VAL_FORMAT_RATE="${FORMAT_RATE}"
VAL_N="${N}"
eval "$(align_vq_extract_eval_metrics "${STAGE3_TEST_RESULT_PATH}")"
TEST_ACCURACY="${ACCURACY}"
TEST_FORMAT_RATE="${FORMAT_RATE}"
TEST_N="${N}"

align_vq_print_header "Stage3 训练完成"
echo "VAL_ACCURACY=${VAL_ACCURACY}"
echo "VAL_FORMAT_RATE=${VAL_FORMAT_RATE}"
echo "VAL_N=${VAL_N}"
echo "TEST_ACCURACY=${TEST_ACCURACY}"
echo "TEST_FORMAT_RATE=${TEST_FORMAT_RATE}"
echo "TEST_N=${TEST_N}"
echo "Stage3 最终产物: ${STAGE3_FINAL_ADAPTER_PATH}"
