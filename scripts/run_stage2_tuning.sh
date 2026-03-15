#!/bin/bash
######################################################################
# RUN_STAGE2_TUNING.SH
#
# Stage2 专用调参脚本（复用已有 Stage1 结果）
# - 对齐 run_stage1_tuning.sh 的路径模板与环境设置
# - 对齐上次“第二次全量 Stage2 测试”参数
######################################################################

# CUDA 环境配置
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=121
export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_OFFLINE=1

# 避免 libgomp 报 OMP_NUM_THREADS 非法值（必须为正整数）
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=8
fi

echo "HOME: ${HOME}"

# 路径模式切换：0=A800(/root), 1=H200(/inspire)
export USE_H200_PATHS=0
if [ "${USE_H200_PATHS}" = "1" ]; then
    export SERVER_NAME="H200"
    default_python="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/.align_vq/bin/python"
    default_script_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/scripts"
    default_root_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/"
    default_base_ckpt_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts"
    default_tune_save_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts_stage2_tune/round1_e3"
    default_model_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf"
    default_scienceqa_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/datasets/ScienceQA"
else
    export SERVER_NAME="A800"
    default_python="/root/autodl-tmp/conda/envs/align_vq/bin/python"
    default_script_dir="/root/workspace/align_vq/scripts"
    default_root_dir="/root/workspace/align_vq/"
    default_base_ckpt_dir="/root/autodl-tmp/vq_lord_ckpts"
    default_tune_save_dir="/root/autodl-tmp/vq_lord_ckpts_stage2_tune/round1_e3"
    default_model_path="/root/autodl-tmp/models/llama3-llava-next-8b-hf"
    default_scienceqa_path="/root/autodl-tmp/datasets/ScienceQA"
fi
export python="${default_python}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "${DEBUG_CUDA:-1}" = "1" ]; then
    export CUDA_LAUNCH_BLOCKING=1
fi

# 项目路径
export script_dir="${default_script_dir}"
export root_dir="${default_root_dir}"
export base_ckpt_dir="${default_base_ckpt_dir}"
export save_dir="${default_tune_save_dir}"
export data_dir="${root_dir}vq_lord_data/"
export preprocess_dir="${base_ckpt_dir}/preprocess"

# 模型路径
export model_path="${default_model_path}"
export victim_model="gpt-4o-mini"

export OPENAI_BASE_URL="https://sg.uiuiapi.com/v1"
export OPENAI_API_KEY="sk-7F7uBRbIJhbziKqHoyCqH0dDl3qT3r1WEN0ne9bebujDZLzr"

# ================== Stage2 调参参数（对齐上次全量 Stage2 计划） ==================

# VQ 参数
export vq_codebook_size=512
export vq_commitment_cost=0.25
export vq_dead_code_threshold=1.0
export vq_usage_decay=0.995
export vq_dead_code_reset_interval=10
export vq_legacy_loss=0
export freeze_vision_tower=0

# 损失权重
export beta=0.05
export temperature=1.5
export tau1=0.01

# 训练阶段：仅跑 Stage2（复用 Stage1）
export stage=2
export stage1_epochs=0
export stage2_epochs=3
export stage3_epochs=0

# 训练超参
export batch_size=2
export grad_accum=8
export stage2_grad_accum=8
export stage3_grad_accum=0
export lr=2e-5
export stage1_lr=5e-5
export stage1_recon_weight=1.0
export stage1_cosine_weight=0.25
export stage1_vq_weight=1.0
export stage1_grad_clip=5.0
export stage2_answer_weight=1.0
export stage2_rationale_weight=0.2
export stage2_prepost_lr_scale=0.2
export stage2_vision_lr_scale=0.2
export stage2_grad_clip=1.0
export stage3_lr_scale=0.2
export stage3_train_projector=0
export max_length=512
export max_new_tokens=128
export use_lora=1
export lora_rank=64
export lora_alpha_val=128
export use_4bit=0
export model_dtype="bfloat16"

# 数据
export train_num=0
export dataset_name="scienceqa"
export scienceqa_path="${default_scienceqa_path}"
export scienceqa_split="train"
export scienceqa_seed=20240306
export collect_teacher_data=0
export strict_teacher_distill=0
export teacher_lang="en"

# 分桶：训练使用 bs=2；预处理文件沿用已存在的 bs=4
export bucket_by="patches"
export bucket_batch_size=2
export stage3_bucket_batch_size=2
export bucket_drop_last=0
export disable_bucket_for_stage3=0
export preprocess_bucket_batch_size=4
export scienceqa_preprocessed_path="${preprocess_dir}/scienceqa_${scienceqa_split}_n${train_num}_seed${scienceqa_seed}_${bucket_by}_bs${preprocess_bucket_batch_size}.json"

# checkpoint 复用策略
export stage1_codebook_path="${base_ckpt_dir}/stage1_vq/vq_codebook.pt"
export stage2_ckpt_path="${save_dir}/stage2_vision"
export reuse_vq_codebook=1
export reuse_stage2=0

# 日志和保存
export log_step=100
export save_step=100

# 评估与决策参数（Step3）
export eval_split="validation"
export eval_max_samples=200
export eval_max_new_tokens=64
export eval_answer_mode="hybrid"
export eval_use_vq=1
export eval_save_path="${save_dir}/stage2_eval_${eval_split}_${eval_answer_mode}.json"
export go_threshold=0.50
export low_acc_threshold=0.45
export format_rate_threshold=0.80

# 可选：GO 后自动触发 Stage3 小步验证
export AUTO_STAGE3_SMOKE=0
export stage3_smoke_train_num=200
export stage3_smoke_epochs=1
export stage3_smoke_save_dir="${save_dir}/stage3_smoke"

echo "======================================================"
echo "Stage2 Tuning 开始"
echo "======================================================"
echo "路径模式: $SERVER_NAME (USE_H200_PATHS=$USE_H200_PATHS)"
echo "python: $python"
echo "root_dir: $root_dir"
echo "模型路径: $model_path"
echo "ScienceQA 路径: $scienceqa_path"
echo "Stage1 codebook: $stage1_codebook_path"
echo "预处理文件: $scienceqa_preprocessed_path"
echo "保存路径: $save_dir"
echo "stage/epochs: $stage / $stage2_epochs"
echo "batch_size/grad_accum/stage2_grad_accum: $batch_size / $grad_accum / $stage2_grad_accum"
echo "beta/answer_w/rationale_w: $beta / $stage2_answer_weight / $stage2_rationale_weight"
echo "prepost_lr_scale/vision_lr_scale: $stage2_prepost_lr_scale / $stage2_vision_lr_scale"
echo "eval: split=$eval_split, max_samples=$eval_max_samples, mode=$eval_answer_mode"
echo "======================================================"

mkdir -p "$save_dir"
mkdir -p "$data_dir"
mkdir -p "$preprocess_dir"

if [ ! -f "${root_dir}vq_lord2/train_vq_lord2.py" ]; then
    echo "错误: 未找到训练入口 ${root_dir}vq_lord2/train_vq_lord2.py"
    exit 1
fi

if [ ! -f "${root_dir}data_preprocess/sciqa_preprocess.py" ]; then
    echo "错误: 未找到预处理脚本 ${root_dir}data_preprocess/sciqa_preprocess.py"
    exit 1
fi

if [ ! -f "${root_dir}vq_lord2/sciqa_process.py" ]; then
    echo "错误: 未找到评估脚本 ${root_dir}vq_lord2/sciqa_process.py"
    exit 1
fi

if [ ! -f "$stage1_codebook_path" ]; then
    echo "错误: Stage2 需要已有 Stage1 codebook，但未找到: $stage1_codebook_path"
    exit 1
fi

if [ "$scienceqa_path" = "ScienceQA" ] && [ "${HF_DATASETS_OFFLINE:-0}" = "1" ]; then
    echo "错误: HF_DATASETS_OFFLINE=1 但未找到本地 ScienceQA。"
    echo "请设置 scienceqa_path 或把数据放到 $default_scienceqa_path"
    exit 1
fi

if [ -f "$scienceqa_preprocessed_path" ]; then
    echo "检测到预处理文件，直接复用: $scienceqa_preprocessed_path"
else
    echo "未找到预处理文件，先生成: $scienceqa_preprocessed_path"
    "$python" "${root_dir}data_preprocess/sciqa_preprocess.py" \
        --dataset-path="$scienceqa_path" \
        --model-path="$model_path" \
        --split="$scienceqa_split" \
        --train-num="$train_num" \
        --seed="$scienceqa_seed" \
        --bucket-by="$bucket_by" \
        --bucket-batch-size="$preprocess_bucket_batch_size" \
        --bucket-drop-last="$bucket_drop_last" \
        --shuffle=1 \
        --save-json="$scienceqa_preprocessed_path"
fi

if [ ! -f "$scienceqa_preprocessed_path" ]; then
    echo "错误: 预处理结果不存在 $scienceqa_preprocessed_path"
    exit 1
fi

"$python" "${root_dir}vq_lord2/train_vq_lord2.py" \
    --model_path="$model_path" \
    --victim_model="$victim_model" \
    --vq_codebook_size="$vq_codebook_size" \
    --vq_commitment_cost="$vq_commitment_cost" \
    --vq_dead_code_threshold="$vq_dead_code_threshold" \
    --vq_usage_decay="$vq_usage_decay" \
    --vq_dead_code_reset_interval="$vq_dead_code_reset_interval" \
    --vq_legacy_loss="$vq_legacy_loss" \
    --freeze_vision_tower="$freeze_vision_tower" \
    --beta="$beta" \
    --temperature="$temperature" \
    --tau1="$tau1" \
    --stage="$stage" \
    --epochs="$stage2_epochs" \
    --batch_size="$batch_size" \
    --lr="$lr" \
    --stage1_lr="$stage1_lr" \
    --stage1_recon_weight="$stage1_recon_weight" \
    --stage1_cosine_weight="$stage1_cosine_weight" \
    --stage1_vq_weight="$stage1_vq_weight" \
    --stage1_grad_clip="$stage1_grad_clip" \
    --max_length="$max_length" \
    --use_lora="$use_lora" \
    --lora_rank="$lora_rank" \
    --lora_alpha="$lora_alpha_val" \
    --use_4bit="$use_4bit" \
    --model_dtype="$model_dtype" \
    --grad_accum="$grad_accum" \
    --stage2_grad_accum="$stage2_grad_accum" \
    --stage2_answer_weight="$stage2_answer_weight" \
    --stage2_rationale_weight="$stage2_rationale_weight" \
    --stage2_prepost_lr_scale="$stage2_prepost_lr_scale" \
    --stage2_vision_lr_scale="$stage2_vision_lr_scale" \
    --stage2_grad_clip="$stage2_grad_clip" \
    --stage3_grad_accum="$stage3_grad_accum" \
    --stage3_lr_scale="$stage3_lr_scale" \
    --stage3_train_projector="$stage3_train_projector" \
    --max_new_tokens="$max_new_tokens" \
    --data_dir="$data_dir" \
    --train_num="$train_num" \
    --dataset_name="$dataset_name" \
    --scienceqa_path="$scienceqa_path" \
    --scienceqa_split="$scienceqa_split" \
    --scienceqa_seed="$scienceqa_seed" \
    --scienceqa_preprocessed_path="$scienceqa_preprocessed_path" \
    --bucket_batch_size="$bucket_batch_size" \
    --stage3_bucket_batch_size="$stage3_bucket_batch_size" \
    --disable_bucket_for_stage3="$disable_bucket_for_stage3" \
    --collect_teacher_data="$collect_teacher_data" \
    --strict_teacher_distill="$strict_teacher_distill" \
    --teacher_lang="$teacher_lang" \
    --reuse_vq_codebook="$reuse_vq_codebook" \
    --reuse_stage2="$reuse_stage2" \
    --vq_codebook_path="$stage1_codebook_path" \
    --stage2_ckpt_path="$stage2_ckpt_path" \
    --save_path="$save_dir" \
    --log_step="$log_step" \
    --save_step="$save_step" \
    --device="cuda"

echo "======================================================"
echo "Stage2 训练完成，开始 Validation 评估"
echo "======================================================"

stage2_vq_codebook_path="${stage2_ckpt_path}/vq_codebook.pt"
if [ ! -f "$stage2_vq_codebook_path" ]; then
    echo "错误: Stage2 产物缺少 vq_codebook.pt: $stage2_vq_codebook_path"
    exit 1
fi

"$python" "${root_dir}vq_lord2/sciqa_process.py" \
    --model_path="$model_path" \
    --adapter_path="$stage2_ckpt_path" \
    --split="$eval_split" \
    --max_samples="$eval_max_samples" \
    --max_new_tokens="$eval_max_new_tokens" \
    --use_4bit="$use_4bit" \
    --use_vq="$eval_use_vq" \
    --vq_codebook_size="$vq_codebook_size" \
    --freeze_vision_tower="$freeze_vision_tower" \
    --vq_codebook_path="$stage2_vq_codebook_path" \
    --answer_mode="$eval_answer_mode" \
    --save_path="$eval_save_path"

if [ ! -f "$eval_save_path" ]; then
    echo "错误: 未生成评估结果文件: $eval_save_path"
    exit 1
fi

eval_summary="$("$python" - "$eval_save_path" <<'PY'
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

print(f"VAL_ACC={acc:.6f}")
print(f"FORMAT_RATE={fmt_rate:.6f}")
PY
)"

eval "$eval_summary"

echo "======================================================"
echo "Stage2 Validation 指标"
echo "VAL_ACC=$VAL_ACC"
echo "FORMAT_RATE=$FORMAT_RATE"
echo "======================================================"

decision="$("$python" - "$VAL_ACC" "$FORMAT_RATE" "$go_threshold" "$low_acc_threshold" "$format_rate_threshold" <<'PY'
import sys
acc = float(sys.argv[1])
fmt = float(sys.argv[2])
go_th = float(sys.argv[3])
low_th = float(sys.argv[4])
fmt_th = float(sys.argv[5])

if acc >= go_th:
    print("GO_STAGE3")
elif acc < low_th and fmt >= fmt_th:
    print("CHECK_STAGE1_UTIL")
else:
    print("RETRY_WITH_SCHEDULER")
PY
)"

echo "Step3 Decision: $decision"

if [ "$decision" = "GO_STAGE3" ]; then
    echo "[Decision] val_acc 达标，可进入 Stage3 小步验证。"
    if [ "${AUTO_STAGE3_SMOKE}" = "1" ]; then
        echo "[Stage3 Smoke] AUTO_STAGE3_SMOKE=1，开始小步验证..."
        mkdir -p "$stage3_smoke_save_dir"
        "$python" "${root_dir}vq_lord2/train_vq_lord2.py" \
            --model_path="$model_path" \
            --victim_model="$victim_model" \
            --vq_codebook_size="$vq_codebook_size" \
            --vq_commitment_cost="$vq_commitment_cost" \
            --vq_dead_code_threshold="$vq_dead_code_threshold" \
            --vq_usage_decay="$vq_usage_decay" \
            --vq_dead_code_reset_interval="$vq_dead_code_reset_interval" \
            --vq_legacy_loss="$vq_legacy_loss" \
            --freeze_vision_tower="$freeze_vision_tower" \
            --beta="$beta" \
            --temperature="$temperature" \
            --tau1="$tau1" \
            --stage=3 \
            --epochs="$stage3_smoke_epochs" \
            --batch_size="$batch_size" \
            --lr="$lr" \
            --stage1_lr="$stage1_lr" \
            --stage1_recon_weight="$stage1_recon_weight" \
            --stage1_cosine_weight="$stage1_cosine_weight" \
            --stage1_vq_weight="$stage1_vq_weight" \
            --stage1_grad_clip="$stage1_grad_clip" \
            --max_length="$max_length" \
            --use_lora="$use_lora" \
            --lora_rank="$lora_rank" \
            --lora_alpha="$lora_alpha_val" \
            --use_4bit="$use_4bit" \
            --model_dtype="$model_dtype" \
            --grad_accum="$grad_accum" \
            --stage2_grad_accum="$stage2_grad_accum" \
            --stage2_answer_weight="$stage2_answer_weight" \
            --stage2_rationale_weight="$stage2_rationale_weight" \
            --stage2_prepost_lr_scale="$stage2_prepost_lr_scale" \
            --stage2_vision_lr_scale="$stage2_vision_lr_scale" \
            --stage2_grad_clip="$stage2_grad_clip" \
            --stage3_grad_accum="$stage3_grad_accum" \
            --stage3_lr_scale="$stage3_lr_scale" \
            --stage3_train_projector="$stage3_train_projector" \
            --max_new_tokens="$max_new_tokens" \
            --data_dir="$data_dir" \
            --train_num="$stage3_smoke_train_num" \
            --dataset_name="$dataset_name" \
            --scienceqa_path="$scienceqa_path" \
            --scienceqa_split="$scienceqa_split" \
            --scienceqa_seed="$scienceqa_seed" \
            --scienceqa_preprocessed_path="" \
            --bucket_batch_size="$bucket_batch_size" \
            --stage3_bucket_batch_size="$stage3_bucket_batch_size" \
            --disable_bucket_for_stage3=1 \
            --collect_teacher_data="$collect_teacher_data" \
            --strict_teacher_distill="$strict_teacher_distill" \
            --teacher_lang="$teacher_lang" \
            --reuse_vq_codebook=1 \
            --reuse_stage2=1 \
            --vq_codebook_path="$stage1_codebook_path" \
            --stage2_ckpt_path="$stage2_ckpt_path" \
            --save_path="$stage3_smoke_save_dir" \
            --log_step="$log_step" \
            --save_step="$save_step" \
            --device="cuda"
    else
        echo "[Stage3 Smoke] 已跳过（AUTO_STAGE3_SMOKE=0）。"
    fi
elif [ "$decision" = "RETRY_WITH_SCHEDULER" ]; then
    echo "[Decision] val_acc 未达标且非低位坍塌，建议第二轮加入 warmup+cosine 后复跑。"
else
    echo "[Decision] val_acc 偏低但格式率正常，优先检查 Stage1 表征利用率（vq_perplexity/dead_code_count）。"
fi

echo "======================================================"
echo "Stage2 Tuning 完成"
echo "产物目录: $save_dir"
echo "======================================================"
