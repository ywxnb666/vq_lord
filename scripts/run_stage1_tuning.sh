#!/bin/bash
######################################################################
# 7.0.VQ_LORD_TRAIN.SH ---
#
# VQ-LoRD 训练脚本
# 用于窃取多模态模型的图像识别能力
#
# 训练流程:
# 1. 阶段1: VQ Codebook 预训练
# 2. 阶段2: 视觉能力蒸馏 
# 3. 阶段3: LoRD 联合训练
#
# Author: VQ-LoRD Project
# Created: January 2026
######################################################################

# CUDA 环境配置
# export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=121
export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_OFFLINE=1

# 避免 libgomp 报 OMP_NUM_THREADS 非法值
if ! [[ "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]]; then
    export OMP_NUM_THREADS=8
fi

echo "HOME: ${HOME}"
export python=${HOME}/autodl-tmp/conda/envs/align_vq/bin/python3
# export python=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/.align_vq/bin/python


export CUDA_VISIBLE_DEVICES=0

export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 需要定位 CUDA 异步报错时可开启：export DEBUG_CUDA=1
if [ "${DEBUG_CUDA:-1}" = "1" ]; then
    export CUDA_LAUNCH_BLOCKING=1
fi

# 项目路径
# 自动定位到当前脚本所在仓库根目录，避免误指向旧工程
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export root_dir="$(cd "${script_dir}/.." && pwd)/"
export save_dir="/root/autodl-tmp/vq_lord_ckpts"
export data_dir="${root_dir}vq_lord_data/"
export preprocess_dir="${save_dir}/preprocess"

# 模型路径
export model_path="/root/autodl-tmp/models/llama3-llava-next-8b-hf"
# export model_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf"
export victim_model="gpt-4o-mini"

export OPENAI_BASE_URL="https://sg.uiuiapi.com/v1"
export OPENAI_API_KEY="sk-7F7uBRbIJhbziKqHoyCqH0dDl3qT3r1WEN0ne9bebujDZLzr"

# ================== VQ-LoRD 参数配置 ==================

# VQ 参数
export vq_codebook_size=512    # Codebook 大小 (降低至512测试坍缩情况)
export vq_commitment_cost=0.05   # Commitment loss 权重 (极大地降低，释放特征聚拢牵引)
export vq_dead_code_threshold=50  # Dead code restart 阈值 (大幅降低以增加重启频率)
export freeze_vision_tower=0    # 是否冻结原始视觉编码器 (0=不冻结, 1=冻结)

# 损失权重
export beta=0.25                # VQ 损失权重
export temperature=1.5          # Stage3 采样温度

# ================== 训练参数选择区 (请二选一取消注释) ==================

# --- 选项 A: 针对 A800 80GB PCIe 优化 (默认) ---
export stage=1                  # 只测 Stage1
export stage1_epochs=1          # Stage1 训练轮数 (测试设为1)
export stage2_epochs=0          # Stage2 训练轮数
export stage3_epochs=0          # Stage3 训练轮数
export batch_size=4             # 常规/回退 DataLoader 批次大小
export grad_accum=8             # 梯度累积步数 (等效batch_size=32)
export lr=2e-5                  # 学习率
export stage1_lr=1e-4           # Stage1 专用学习率 (调大冲出局部最优点)
export stage1_recon_weight=1.0  # Stage1 特征重建损失权重
export stage1_cosine_weight=0.5 # Stage1 余弦一致性损失权重 (调大保留语义)
export stage1_vq_weight=1.0     # Stage1 VQ 损失权重
export max_length=1024          # 最大序列长度 (80GB充分利用长上下文)
export max_new_tokens=256       # LoRD生成最大token数 (更完整的回复用于对比)
# LoRA 参数
export use_lora=1               # 使用 LoRA
export lora_rank=64             # LoRA rank (80GB可支持更大rank,提升拟合能力)
export lora_alpha_val=128       # LoRA alpha (通常为rank的2倍)
# 量化
export use_4bit=1               # 使用 4-bit 量化

# --- 选项 B: 针对 H200 141GB SXM 优化 (高性能) ---
# export stage=3                  # 训练阶段
# export stage1_epochs=1          # Stage1 训练轮数
# export stage2_epochs=1          # Stage2 训练轮数
# export stage3_epochs=1          # Stage3 训练轮数
# export batch_size=8             # 常规/回退 DataLoader 批次大小；启用分桶时由 bucket_batch_size 控制
# export grad_accum=8             # 全局回退梯度累积步数
export stage2_grad_accum=0      # Stage2 单独梯度累积；0 表示回退到 grad_accum
export stage3_grad_accum=0      # Stage3 单独梯度累积；0 表示回退到 grad_accum
# export lr=4e-5                  # 学习率 (BS增大，LR适当增加)
# export max_length=1024          # 最大序列长度 (141GB显存可支持极长上下文，提升指令跟随)
# export max_new_tokens=256       # LoRD生成最长token数 (生成更详细的推理过程)
# # LoRA 参数
# export use_lora=1               # 使用 LoRA
# export lora_rank=256            # LoRA rank (H200可支持极大秩，接近全量微调效果)
# export lora_alpha_val=512       # LoRA alpha 
# # 量化
# export use_4bit=0               # 使用 4-bit 量化（若需全精度训练请设为0）
export model_dtype="bfloat16"   # 非 4bit 模式下使用 bfloat16 / float16 / float32

# # ---------------------

# 数据
export train_num=0           # 训练样本数（0表示使用全部数据）
export dataset_name="scienceqa" # 使用 ScienceQA 数据集
# ScienceQA 数据路径（优先环境变量，其次本地目录，最后才回退到 HF 数据集名）
# SCIENCEQA_PATH="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/datasets/ScienceQA"
SCIENCEQA_PATH="/root/autodl-tmp/datasets/ScienceQA"

if [ -n "${SCIENCEQA_PATH:-}" ]; then
    export scienceqa_path="$SCIENCEQA_PATH"
elif [ -d "${root_dir}downloads/dataset/ScienceQA" ]; then
    export scienceqa_path="${root_dir}downloads/dataset/ScienceQA"
elif [ -d "${root_dir}downloads/datasets/ScienceQA" ]; then
    export scienceqa_path="${root_dir}downloads/datasets/ScienceQA"
elif [ -d "/root/autodl-tmp/datasets/ScienceQA" ]; then
    export scienceqa_path="/root/autodl-tmp/datasets/ScienceQA"
else
    export scienceqa_path="ScienceQA"
fi
export scienceqa_split="train"  # ScienceQA split
export scienceqa_eval_split="validation"  # ScienceQA 验证/测试 split
export scienceqa_seed=20240306  # ScienceQA 划分随机种子
export collect_teacher_data=1    # 自动补齐 GPT-4V 教师回答
export strict_teacher_distill=1  # 严格模式：缺少教师回答则报错
export teacher_lang="en"        # 教师回答统一语言: zh / en
export bucket_by="patches"      # 预处理分桶方式: patches / size / none
export bucket_batch_size=4       # Stage1/2 分桶 batch size
export stage3_bucket_batch_size=4  # Stage3 分桶 batch size，先保守设为 2，可稳定后再提到 4
export bucket_drop_last=0        # 是否丢弃尾包
export disable_bucket_for_stage3=0  # Stage3 启用分桶
export scienceqa_preprocessed_path="${preprocess_dir}/scienceqa_${scienceqa_split}_n${train_num}_seed${scienceqa_seed}_${bucket_by}_bs${bucket_batch_size}.json"
export stage1_codebook_path="${save_dir}/stage1_vq/vq_codebook.pt"
export stage2_ckpt_path="${save_dir}/stage2_vision"
export reuse_vq_codebook=1
export reuse_stage2=1

# 日志和保存
export log_step=100
export save_step=100

# ================== 训练执行 ==================

echo "======================================================"
echo "VQ-LoRD 训练开始"
echo "======================================================"
echo "模型路径: $model_path"
echo "教师模型: $victim_model"
echo "ScienceQA 路径: $scienceqa_path"
echo "VQ Codebook 大小: $vq_codebook_size"
echo "VQ dead code threshold: $vq_dead_code_threshold"
echo "回退 batch_size: $batch_size"
echo "全局回退 grad_accum: $grad_accum"
echo "Stage2 grad_accum: $stage2_grad_accum"
echo "Stage3 grad_accum: $stage3_grad_accum"
echo "use_4bit: $use_4bit"
echo "model_dtype: $model_dtype"
echo "训练阶段: $stage"
echo "Stage1 epochs: $stage1_epochs"
echo "Stage1 lr: $stage1_lr"
echo "Stage1 recon/cos/vq: $stage1_recon_weight / $stage1_cosine_weight / $stage1_vq_weight"
echo "Stage2 epochs: $stage2_epochs"
echo "Stage3 epochs: $stage3_epochs"
echo "保存路径: $save_dir"
echo "Stage1/2 分桶文件: $scienceqa_preprocessed_path"
echo "Stage1/2 分桶 batch size: $bucket_batch_size"
echo "Stage3 分桶 batch size: $stage3_bucket_batch_size"
echo "Stage3 禁用分桶: $disable_bucket_for_stage3"
echo "======================================================"

# 创建必要目录
mkdir -p $save_dir
mkdir -p $data_dir
mkdir -p $preprocess_dir

# 校验训练入口是否存在
if [ ! -f "${root_dir}vq_lord2/train_vq_lord2.py" ]; then
    echo "错误: 未找到训练入口 ${root_dir}vq_lord2/train_vq_lord2.py"
    echo "当前 root_dir=${root_dir}"
    exit 1
fi

if [ ! -f "${root_dir}data_preprocess/sciqa_preprocess.py" ]; then
    echo "错误: 未找到预处理脚本 ${root_dir}data_preprocess/sciqa_preprocess.py"
    exit 1
fi

if [ "$scienceqa_path" = "ScienceQA" ] && [ "${HF_DATASETS_OFFLINE:-0}" = "1" ]; then
    echo "错误: 当前启用了 HF_DATASETS_OFFLINE=1，但未找到本地 ScienceQA 数据集目录。"
    echo "请任选其一："
    echo "1) 导出 SCIENCEQA_PATH=/你的本地/ScienceQA 路径"
    echo "2) 把数据放到 ${root_dir}downloads/dataset/ScienceQA"
    echo "3) 把数据放到 ${root_dir}downloads/datasets/ScienceQA"
    echo "4) 把数据放到 /root/autodl-tmp/datasets/ScienceQA"
    exit 1
fi

echo "======================================================"
echo "先执行 ScienceQA 预处理"
echo "预处理输出: $scienceqa_preprocessed_path"
echo "分桶方式: $bucket_by"
echo "分桶 batch size: $bucket_batch_size"
echo "======================================================"

$python ${root_dir}data_preprocess/sciqa_preprocess.py \
    --dataset-path=$scienceqa_path \
    --model-path=$model_path \
    --split=$scienceqa_split \
    --train-num=$train_num \
    --seed=$scienceqa_seed \
    --bucket-by=$bucket_by \
    --bucket-batch-size=$bucket_batch_size \
    --bucket-drop-last=$bucket_drop_last \
    --shuffle=1 \
    --save-json=$scienceqa_preprocessed_path

if [ ! -f "$scienceqa_preprocessed_path" ]; then
    echo "错误: 预处理结果不存在 $scienceqa_preprocessed_path"
    exit 1
fi

run_train_stage() {
    local stage_id="$1"
    local stage_epochs="$2"

    echo "======================================================"
    echo "开始 Stage${stage_id} 训练，epochs=${stage_epochs}"
    echo "======================================================"

    $python ${root_dir}vq_lord2/train_vq_lord2.py \
        --model_path=$model_path \
        --victim_model=$victim_model \
        --vq_codebook_size=$vq_codebook_size \
        --vq_commitment_cost=$vq_commitment_cost \
        --vq_dead_code_threshold=$vq_dead_code_threshold \
        --freeze_vision_tower=$freeze_vision_tower \
        --beta=$beta \
        --temperature=$temperature \
        --stage=$stage_id \
        --epochs=$stage_epochs \
        --batch_size=$batch_size \
        --lr=$lr \
        --stage1_lr=$stage1_lr \
        --stage1_recon_weight=$stage1_recon_weight \
        --stage1_cosine_weight=$stage1_cosine_weight \
        --stage1_vq_weight=$stage1_vq_weight \
        --max_length=$max_length \
        --use_lora=$use_lora \
        --lora_rank=$lora_rank \
        --lora_alpha=$lora_alpha_val \
        --use_4bit=$use_4bit \
        --model_dtype=$model_dtype \
        --grad_accum=$grad_accum \
        --stage2_grad_accum=$stage2_grad_accum \
        --stage3_grad_accum=$stage3_grad_accum \
        --max_new_tokens=$max_new_tokens \
        --data_dir=$data_dir \
        --train_num=$train_num \
        --dataset_name=$dataset_name \
        --scienceqa_path=$scienceqa_path \
        --scienceqa_split=$scienceqa_split \
        --scienceqa_seed=$scienceqa_seed \
        --scienceqa_preprocessed_path=$scienceqa_preprocessed_path \
        --bucket_batch_size=$bucket_batch_size \
        --stage3_bucket_batch_size=$stage3_bucket_batch_size \
        --disable_bucket_for_stage3=$disable_bucket_for_stage3 \
        --collect_teacher_data=$collect_teacher_data \
        --strict_teacher_distill=$strict_teacher_distill \
        --teacher_lang=$teacher_lang \
        --reuse_vq_codebook=$reuse_vq_codebook \
        --reuse_stage2=$reuse_stage2 \
        --vq_codebook_path=$stage1_codebook_path \
        --stage2_ckpt_path=$stage2_ckpt_path \
        --save_path=$save_dir \
        --log_step=$log_step \
        --save_step=$save_step \
        --device="cuda"
}

if [ "$stage" -ge 2 ] && [ "$stage1_epochs" -le 0 ] && [ ! -f "$stage1_codebook_path" ]; then
    echo "错误: Stage2/3 需要已有 Stage1 codebook，但未找到 $stage1_codebook_path"
    echo "请将 stage1_epochs 设为正数，或提供已有 Stage1 checkpoint"
    exit 1
fi

if [ "$stage" -ge 3 ] && [ "$stage2_epochs" -le 0 ] && [ ! -f "$stage2_ckpt_path/vq_codebook.pt" ]; then
    echo "错误: Stage3 需要已有 Stage2 checkpoint，但未找到 $stage2_ckpt_path"
    echo "请将 stage2_epochs 设为正数，或提供已有 Stage2 checkpoint"
    exit 1
fi

if [ "$stage" -ge 1 ] && [ "$stage1_epochs" -gt 0 ]; then
    run_train_stage 1 "$stage1_epochs"
fi

if [ "$stage" -ge 2 ] && [ "$stage2_epochs" -gt 0 ]; then
    run_train_stage 2 "$stage2_epochs"
fi

if [ "$stage" -ge 3 ] && [ "$stage3_epochs" -gt 0 ]; then
    run_train_stage 3 "$stage3_epochs"
fi

echo "======================================================"
echo "VQ-LoRD 训练完成"
echo "======================================================"
