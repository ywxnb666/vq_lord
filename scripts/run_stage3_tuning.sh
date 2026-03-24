#!/bin/bash
set -euo pipefail

######################################################################
# RUN_STAGE3_TUNING.SH
#
# Stage3 专用调参脚本（默认 H200）
# 前提：Stage1/Stage2 已完成，可复用 Stage2 checkpoint
# 流程：Stage3 训练 -> Validation 评估 -> Test 评估
######################################################################

# CUDA 环境配置
export CUDA_HOME=/usr/local/cuda-12
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export BNB_CUDA_VERSION=121
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=8
fi

# 路径模式：默认 H200
export USE_H200_PATHS=1
if [ "${USE_H200_PATHS}" = "1" ]; then
    export SERVER_NAME="H200"
    default_python="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/.align_vq/bin/python"
    default_root_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/"
    default_model_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf"
    default_scienceqa_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/datasets/ScienceQA"
    default_base_ckpt_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts"
    default_stage2_base="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts_stage2_tune/round1_e3"
    default_stage3_root="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts_stage3_tune"
else
    export SERVER_NAME="A800"
    default_python="/root/autodl-tmp/conda/envs/align_vq/bin/python"
    default_root_dir="/root/workspace/align_vq/"
    default_model_path="/root/autodl-tmp/models/llama3-llava-next-8b-hf"
    default_scienceqa_path="/root/autodl-tmp/datasets/ScienceQA"
    default_base_ckpt_dir="/root/autodl-tmp/vq_lord_ckpts"
    default_stage2_base="/root/autodl-tmp/vq_lord_ckpts_stage2_tune/round1_e3"
    default_stage3_root="/root/autodl-tmp/vq_lord_ckpts_stage3_tune"
fi

export python="${PYTHON_BIN:-$default_python}"
export root_dir="${ROOT_DIR:-$default_root_dir}"
export model_path="${MODEL_PATH:-$default_model_path}"
export scienceqa_path="${SCIENCEQA_PATH:-$default_scienceqa_path}"
export base_ckpt_dir="${BASE_CKPT_DIR:-$default_base_ckpt_dir}"
export preprocess_dir="${PREPROCESS_DIR:-${base_ckpt_dir}/preprocess}"
export victim_model="${VICTIM_MODEL:-qwen3.5-flash-2026-02-23}"

export stage2_base_dir="${STAGE2_BASE_DIR:-$default_stage2_base}"
export stage2_ckpt_path="${STAGE2_CKPT_PATH:-${stage2_base_dir}/stage2_vision_epoch15}"
export stage1_codebook_path="${STAGE1_CODEBOOK_PATH:-${stage2_ckpt_path}/vq_codebook.pt}"

legacy_run_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts_stage3_tune/run_20260323_140152"
ts="$(date +%Y%m%d_%H%M%S)"
export save_dir="${SAVE_DIR:-${legacy_run_dir}}"
export stage3_resume_dir="${STAGE3_RESUME_DIR:-${save_dir}/stage3_resume_latest}"

export CUDA_VISIBLE_DEVICES=3
export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------- Stage3 参数 ----------------------
export stage=3
export stage3_epochs=50  # period_num<=0 时回退
export sub_stage_num=1
export period_num=50
export sub_set_num=0

export train_num=0        # 0=全量 train
export batch_size=2     # phaseB
export grad_accum=4
export stage3_grad_accum=4

export lr="${LR:-3e-5}"
export stage3_lr_scale="${STAGE3_LR_SCALE:-0.2}"
export stage3_grad_clip="${STAGE3_GRAD_CLIP:-1.0}"
export stage3_train_projector="${STAGE3_TRAIN_PROJECTOR:-0}"

export tau1="${TAU1:-0.001}"
export tau_delta="${TAU_DELTA:-0.005}"
export temperature="${TEMPERATURE:-1.2}"
export max_new_tokens="${MAX_NEW_TOKENS:-128}"
export max_length="${MAX_LENGTH:-1024}"

export stage3_eval_max_samples=0
export stage3_eval_every_period=1
export stage3_eval_scienceqa_split="${STAGE3_EVAL_SCIENCEQA_SPLIT:-validation}"
export stage3_eval_scienceqa_path="${STAGE3_EVAL_SCIENCEQA_PATH:-}"
export stage3_eval_train_num="${STAGE3_EVAL_TRAIN_NUM:-0}"
export stage3_eval_answer_mode="${STAGE3_EVAL_ANSWER_MODE:-logits}"
export stage3_field_weight_observed="${STAGE3_FIELD_WEIGHT_OBSERVED:-1.15}"
export stage3_field_weight_context="${STAGE3_FIELD_WEIGHT_CONTEXT:-0.30}"
export stage3_field_weight_reasoning="${STAGE3_FIELD_WEIGHT_REASONING:-1.35}"
export stage3_field_weight_answer="${STAGE3_FIELD_WEIGHT_ANSWER:-1.50}"
export stage3_mc_weight="${STAGE3_MC_WEIGHT:-1.00}"
export stage3_obj_weight="${STAGE3_OBJ_WEIGHT:-0.05}"
export stage3_reg_weight="${STAGE3_REG_WEIGHT:-0.30}"
export stage3_answer_anchor_weight="${STAGE3_ANSWER_ANCHOR_WEIGHT:-1.00}"
export stage3_pair_use_answer_correctness="${STAGE3_PAIR_USE_ANSWER_CORRECTNESS:-1}"
export stage3_wrong_image_enable="${STAGE3_WRONG_IMAGE_ENABLE:-0}"
export stage3_wrong_image_weight="${STAGE3_WRONG_IMAGE_WEIGHT:-0.2}"
export stage3_wrong_image_margin="${STAGE3_WRONG_IMAGE_MARGIN:-0.0}"
export stage3_force_cold_start_period0=0
export stage3_resume_save_optimizer="${STAGE3_RESUME_SAVE_OPTIMIZER:-1}"
export stage3_resume_save_interval="${STAGE3_RESUME_SAVE_INTERVAL:-1}"
export stage3_resume_path="${STAGE3_RESUME_PATH:-${stage3_resume_dir}}"

# 与 Stage2 保持一致
export vq_codebook_size="${VQ_CODEBOOK_SIZE:-1024}"
export vq_commitment_cost="${VQ_COMMITMENT_COST:-0.25}"
export vq_dead_code_threshold="${VQ_DEAD_CODE_THRESHOLD:-1.0}"
export vq_usage_decay="${VQ_USAGE_DECAY:-0.995}"
export vq_dead_code_reset_interval="${VQ_DEAD_CODE_RESET_INTERVAL:-10}"
export vq_legacy_loss="${VQ_LEGACY_LOSS:-0}"
export freeze_vision_tower="${FREEZE_VISION_TOWER:-0}"
export beta="${BETA:-0.05}"

export use_lora=1
export lora_rank=64
export lora_alpha_val="${LORA_ALPHA:-128}"
export use_4bit=0
export model_dtype="${MODEL_DTYPE:-bfloat16}"

export dataset_name="scienceqa"
export scienceqa_split="${SCIENCEQA_SPLIT:-train}"
export scienceqa_seed="${SCIENCEQA_SEED:-20240306}"
export teacher_lang="${TEACHER_LANG:-en}"
export collect_teacher_data="${COLLECT_TEACHER_DATA:-0}"
export strict_teacher_distill="${STRICT_TEACHER_DISTILL:-0}"
export teacher_cache_path="${TEACHER_CACHE_PATH:-}"
export teacher_observed_max_tokens="${TEACHER_OBSERVED_MAX_TOKENS:-256}"
export teacher_context_max_tokens="${TEACHER_CONTEXT_MAX_TOKENS:-192}"
export teacher_reasoning_max_tokens="${TEACHER_REASONING_MAX_TOKENS:-256}"
export teacher_answer_max_tokens="${TEACHER_ANSWER_MAX_TOKENS:-64}"
export teacher_max_new_tokens_total="${TEACHER_MAX_NEW_TOKENS_TOTAL:-768}"
export stage3_vic_include_context="${STAGE3_VIC_INCLUDE_CONTEXT:-0}"
export scienceqa_preprocessed_path="${SCIENCEQA_PREPROCESSED_PATH:-${preprocess_dir}/scienceqa_${scienceqa_split}_n${train_num}_seed${scienceqa_seed}_patches_bs8.json}"
export bucket_batch_size=8
export stage3_bucket_batch_size=16      # phaseA
export disable_bucket_for_stage3="${DISABLE_BUCKET_FOR_STAGE3:-0}"
export stage3_sample_cache_path="${STAGE3_SAMPLE_CACHE_PATH:-${stage2_base_dir}/stage3_sample_cache_${victim_model}_${scienceqa_split}_n${train_num}_seed${scienceqa_seed}_ml${max_length}_ctx${stage3_vic_include_context}_obs${teacher_observed_max_tokens}_txt${teacher_context_max_tokens}_rsn${teacher_reasoning_max_tokens}_ans${teacher_answer_max_tokens}.pt}"

export reuse_vq_codebook=1
export reuse_stage2=1

# 非 Stage3 训练项（仅为了参数完整）
export stage1_lr="${STAGE1_LR:-5e-5}"
export stage1_recon_weight="${STAGE1_RECON_WEIGHT:-1.0}"
export stage1_cosine_weight="${STAGE1_COSINE_WEIGHT:-0.25}"
export stage1_vq_weight="${STAGE1_VQ_WEIGHT:-1.0}"
export stage1_grad_clip="${STAGE1_GRAD_CLIP:-5.0}"
export stage2_answer_weight="${STAGE2_ANSWER_WEIGHT:-1.0}"
export stage2_rationale_weight="${STAGE2_RATIONALE_WEIGHT:-0.2}"
export stage2_prepost_lr_scale="${STAGE2_PREPOST_LR_SCALE:-0.2}"
export stage2_vision_lr_scale="${STAGE2_VISION_LR_SCALE:-0.2}"
export stage2_grad_clip="${STAGE2_GRAD_CLIP:-1.0}"
export stage2_grad_accum="${STAGE2_GRAD_ACCUM:-4}"

export log_step="${LOG_STEP:-100}"
export save_step=0
export save_each_epoch=1

# 评估配置（Stage3 后）
export eval_max_samples=2097
export eval_max_new_tokens="${EVAL_MAX_NEW_TOKENS:-64}"
export eval_answer_mode="${EVAL_ANSWER_MODE:-logit}"
export eval_use_vq=1
export eval_val_save_path="${save_dir}/stage3_eval_validation_${eval_answer_mode}.json"
export eval_test_save_path="${save_dir}/stage3_eval_test_${eval_answer_mode}.json"

echo "======================================================"
echo "Stage3 Tuning 开始"
echo "======================================================"
echo "路径模式: ${SERVER_NAME} (USE_H200_PATHS=${USE_H200_PATHS})"
echo "python: ${python}"
echo "root_dir: ${root_dir}"
echo "model_path: ${model_path}"
echo "scienceqa_path: ${scienceqa_path}"
echo "stage2_ckpt_path: ${stage2_ckpt_path}"
echo "save_dir: ${save_dir}"
echo "stage3_resume_path: ${stage3_resume_path:-<empty>}"
echo "stage3_resume_dir: ${stage3_resume_dir}"
echo "stage3_sample_cache_path: ${stage3_sample_cache_path}"
echo "scienceqa_preprocessed_path: ${scienceqa_preprocessed_path}"
echo "stage3_bucket_batch_size/disable_bucket_for_stage3: ${stage3_bucket_batch_size}/${disable_bucket_for_stage3}"
echo "teacher_cache_path: ${teacher_cache_path:-<auto *_new.json>}"
echo "sub_stage/period/subset: ${sub_stage_num}/${period_num}/${sub_set_num}"
echo "batch_size/grad_accum/stage3_grad_accum: ${batch_size}/${grad_accum}/${stage3_grad_accum}"
echo "tau1/tau_delta/max_new_tokens: ${tau1}/${tau_delta}/${max_new_tokens}"
echo "stage3_eval_max_samples/every_period: ${stage3_eval_max_samples}/${stage3_eval_every_period}"
echo "stage3_eval_split/path/train_num: ${stage3_eval_scienceqa_split}/${stage3_eval_scienceqa_path:-<reuse scienceqa_path>}/${stage3_eval_train_num}"
echo "stage3_eval_answer_mode: ${stage3_eval_answer_mode}"
echo "stage3_field_weight(obs/ctx/reason/ans): ${stage3_field_weight_observed}/${stage3_field_weight_context}/${stage3_field_weight_reasoning}/${stage3_field_weight_answer}"
echo "stage3_mc/obj/reg/ans_anchor: ${stage3_mc_weight}/${stage3_obj_weight}/${stage3_reg_weight}/${stage3_answer_anchor_weight}"
echo "stage3_pair_by_answer: ${stage3_pair_use_answer_correctness}"
echo "stage3_wrong_image(enable/weight/margin): ${stage3_wrong_image_enable}/${stage3_wrong_image_weight}/${stage3_wrong_image_margin}"
echo "stage3_force_cold_start_period0: ${stage3_force_cold_start_period0}"
echo "======================================================"

mkdir -p "${save_dir}"

train_entry="${root_dir}vq_lord3/train_vq_lord3.py"
eval_entry="${root_dir}vq_lord3/sciqa_process.py"
if [ ! -f "${train_entry}" ]; then
    echo "错误: 未找到训练入口 ${train_entry}"
    exit 1
fi
if [ ! -f "${eval_entry}" ]; then
    echo "错误: 未找到评估入口 ${eval_entry}"
    exit 1
fi

if [ ! -d "${stage2_ckpt_path}" ]; then
    echo "错误: 未找到 Stage2 checkpoint 目录: ${stage2_ckpt_path}"
    exit 1
fi
if [ ! -f "${stage2_ckpt_path}/vq_codebook.pt" ]; then
    echo "错误: Stage2 缺少 vq_codebook.pt: ${stage2_ckpt_path}"
    exit 1
fi
if [ ! -f "${stage2_ckpt_path}/projector.pt" ]; then
    echo "错误: Stage2 缺少 projector.pt: ${stage2_ckpt_path}"
    exit 1
fi
if [ ! -f "${stage2_ckpt_path}/adapter_config.json" ]; then
    echo "错误: Stage2 缺少 adapter_config.json: ${stage2_ckpt_path}"
    exit 1
fi
if [ ! -f "${stage1_codebook_path}" ]; then
    echo "错误: 未找到 Stage1 codebook: ${stage1_codebook_path}"
    exit 1
fi

"${python}" "${train_entry}" \
    --model_path="${model_path}" \
    --victim_model="${victim_model}" \
    --vq_codebook_size="${vq_codebook_size}" \
    --vq_commitment_cost="${vq_commitment_cost}" \
    --vq_dead_code_threshold="${vq_dead_code_threshold}" \
    --vq_usage_decay="${vq_usage_decay}" \
    --vq_dead_code_reset_interval="${vq_dead_code_reset_interval}" \
    --vq_legacy_loss="${vq_legacy_loss}" \
    --freeze_vision_tower="${freeze_vision_tower}" \
    --beta="${beta}" \
    --temperature="${temperature}" \
    --tau1="${tau1}" \
    --tau_delta="${tau_delta}" \
    --stage="${stage}" \
    --epochs="${stage3_epochs}" \
    --batch_size="${batch_size}" \
    --lr="${lr}" \
    --stage1_lr="${stage1_lr}" \
    --stage1_recon_weight="${stage1_recon_weight}" \
    --stage1_cosine_weight="${stage1_cosine_weight}" \
    --stage1_vq_weight="${stage1_vq_weight}" \
    --stage1_grad_clip="${stage1_grad_clip}" \
    --max_length="${max_length}" \
    --use_lora="${use_lora}" \
    --lora_rank="${lora_rank}" \
    --lora_alpha="${lora_alpha_val}" \
    --use_4bit="${use_4bit}" \
    --model_dtype="${model_dtype}" \
    --grad_accum="${grad_accum}" \
    --stage2_grad_accum="${stage2_grad_accum}" \
    --stage2_answer_weight="${stage2_answer_weight}" \
    --stage2_rationale_weight="${stage2_rationale_weight}" \
    --stage2_prepost_lr_scale="${stage2_prepost_lr_scale}" \
    --stage2_vision_lr_scale="${stage2_vision_lr_scale}" \
    --stage2_grad_clip="${stage2_grad_clip}" \
    --stage3_grad_accum="${stage3_grad_accum}" \
    --stage3_lr_scale="${stage3_lr_scale}" \
    --stage3_grad_clip="${stage3_grad_clip}" \
    --stage3_train_projector="${stage3_train_projector}" \
    --stage3_eval_max_samples="${stage3_eval_max_samples}" \
    --stage3_eval_every_period="${stage3_eval_every_period}" \
    --stage3_eval_scienceqa_split="${stage3_eval_scienceqa_split}" \
    --stage3_eval_scienceqa_path="${stage3_eval_scienceqa_path}" \
    --stage3_eval_train_num="${stage3_eval_train_num}" \
    --stage3_eval_answer_mode="${stage3_eval_answer_mode}" \
    --stage3_field_weight_observed="${stage3_field_weight_observed}" \
    --stage3_field_weight_context="${stage3_field_weight_context}" \
    --stage3_field_weight_reasoning="${stage3_field_weight_reasoning}" \
    --stage3_field_weight_answer="${stage3_field_weight_answer}" \
    --stage3_mc_weight="${stage3_mc_weight}" \
    --stage3_obj_weight="${stage3_obj_weight}" \
    --stage3_reg_weight="${stage3_reg_weight}" \
    --stage3_answer_anchor_weight="${stage3_answer_anchor_weight}" \
    --stage3_pair_use_answer_correctness="${stage3_pair_use_answer_correctness}" \
    --stage3_wrong_image_enable="${stage3_wrong_image_enable}" \
    --stage3_wrong_image_weight="${stage3_wrong_image_weight}" \
    --stage3_wrong_image_margin="${stage3_wrong_image_margin}" \
    --stage3_force_cold_start_period0="${stage3_force_cold_start_period0}" \
    --sub_stage_num="${sub_stage_num}" \
    --period_num="${period_num}" \
    --sub_set_num="${sub_set_num}" \
    --max_new_tokens="${max_new_tokens}" \
    --data_dir="${root_dir}vq_lord_data/" \
    --train_num="${train_num}" \
    --dataset_name="${dataset_name}" \
    --scienceqa_path="${scienceqa_path}" \
    --scienceqa_split="${scienceqa_split}" \
    --scienceqa_seed="${scienceqa_seed}" \
    --scienceqa_preprocessed_path="${scienceqa_preprocessed_path}" \
    --bucket_batch_size="${bucket_batch_size}" \
    --stage3_bucket_batch_size="${stage3_bucket_batch_size}" \
    --disable_bucket_for_stage3="${disable_bucket_for_stage3}" \
    --collect_teacher_data="${collect_teacher_data}" \
    --strict_teacher_distill="${strict_teacher_distill}" \
    --teacher_lang="${teacher_lang}" \
    --teacher_cache_path="${teacher_cache_path}" \
    --teacher_observed_max_tokens="${teacher_observed_max_tokens}" \
    --teacher_context_max_tokens="${teacher_context_max_tokens}" \
    --teacher_reasoning_max_tokens="${teacher_reasoning_max_tokens}" \
    --teacher_answer_max_tokens="${teacher_answer_max_tokens}" \
    --teacher_max_new_tokens_total="${teacher_max_new_tokens_total}" \
    --stage3_vic_include_context="${stage3_vic_include_context}" \
    --reuse_vq_codebook="${reuse_vq_codebook}" \
    --reuse_stage2="${reuse_stage2}" \
    --vq_codebook_path="${stage1_codebook_path}" \
    --stage2_ckpt_path="${stage2_ckpt_path}" \
    --save_path="${save_dir}" \
    --log_step="${log_step}" \
    --save_step="${save_step}" \
    --save_each_epoch="${save_each_epoch}" \
    --stage3_resume_path="${stage3_resume_path}" \
    --stage3_resume_save_path="${stage3_resume_dir}" \
    --stage3_resume_save_optimizer="${stage3_resume_save_optimizer}" \
    --stage3_resume_save_interval="${stage3_resume_save_interval}" \
    --stage3_sample_cache_path="${stage3_sample_cache_path}" \
    --device="cuda"

stage3_final_adapter="${save_dir}/stage3_lord_final"
stage3_final_vq="${stage3_final_adapter}/vq_codebook.pt"
if [ ! -f "${stage3_final_adapter}/adapter_config.json" ]; then
    echo "错误: Stage3 最终 adapter 不存在: ${stage3_final_adapter}"
    exit 1
fi
if [ ! -f "${stage3_final_vq}" ]; then
    echo "错误: Stage3 最终 vq_codebook 不存在: ${stage3_final_vq}"
    exit 1
fi

run_eval() {
    local split="$1"
    local save_path="$2"
    "${python}" "${eval_entry}" \
        --model_path="${model_path}" \
        --adapter_path="${stage3_final_adapter}" \
        --scienceqa_path="${scienceqa_path}" \
        --split="${split}" \
        --max_samples="${eval_max_samples}" \
        --max_new_tokens="${eval_max_new_tokens}" \
        --use_4bit="${use_4bit}" \
        --use_vq="${eval_use_vq}" \
        --vq_codebook_size="${vq_codebook_size}" \
        --freeze_vision_tower="${freeze_vision_tower}" \
        --vq_codebook_path="${stage3_final_vq}" \
        --answer_mode="${eval_answer_mode}" \
        --save_path="${save_path}"
}

extract_eval_summary() {
    local json_path="$1"
    "${python}" - "${json_path}" <<'PY'
import json
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

metrics = payload.get("metrics", {})
acc = float(metrics.get("accuracy", 0.0))
results = payload.get("results", [])

pat = re.compile(r"(?:answer|答案)\s*[:：]\s*\(?\s*([A-D])\s*\)?", re.IGNORECASE)
head_pat = re.compile(r"^\s*\(?\s*([A-D])\s*\)?\b", re.IGNORECASE)
fmt_hits = 0
for item in results:
    text = str(item.get("output", "") or "")
    if pat.search(text) or head_pat.search(text):
        fmt_hits += 1
total = max(1, len(results))
fmt_rate = fmt_hits / total
print(f"ACC={acc:.6f}")
print(f"FORMAT_RATE={fmt_rate:.6f}")
print(f"N={len(results)}")
PY
}

echo "======================================================"
echo "Stage3 训练完成，开始评估"
echo "======================================================"

run_eval "validation" "${eval_val_save_path}"
if [ ! -f "${eval_val_save_path}" ]; then
    echo "错误: validation 评估结果未生成: ${eval_val_save_path}"
    exit 1
fi
eval "$(extract_eval_summary "${eval_val_save_path}")"
val_acc="$ACC"
val_fmt="$FORMAT_RATE"
val_n="$N"

run_eval "test" "${eval_test_save_path}"
if [ ! -f "${eval_test_save_path}" ]; then
    echo "错误: test 评估结果未生成: ${eval_test_save_path}"
    exit 1
fi
eval "$(extract_eval_summary "${eval_test_save_path}")"
test_acc="$ACC"
test_fmt="$FORMAT_RATE"
test_n="$N"

echo "======================================================"
echo "Stage3 Tuning 完成"
echo "Validation: acc=${val_acc}, format_rate=${val_fmt}, n=${val_n}"
echo "Test:       acc=${test_acc}, format_rate=${test_fmt}, n=${test_n}"
echo "保存目录: ${save_dir}"
echo "======================================================"
