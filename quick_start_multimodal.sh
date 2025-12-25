#!/bin/bash
######################################################################
# 多模态模型窃取 - 快速入门示例
# 
# 这个脚本演示如何使用最小配置快速测试多模态模型窃取功能
######################################################################

echo "=========================================="
echo "多模态模型窃取 - 快速入门示例"
echo "=========================================="
echo ""

# 检查环境
echo "1. 检查环境..."
if [ ! -d "$HOME/anaconda3/envs/align" ]; then
    echo "❌ 错误: align 环境不存在"
    echo "   请先创建环境: conda create -n align python=3.10"
    exit 1
fi

# 检查模型路径
echo "2. 检查模型路径..."
MODEL_PATH="/root/workspace/models/llama3-llava-next-8b-hf"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 找不到 LLaVA 模型: $MODEL_PATH"
    echo "   请确保模型已下载到正确路径"
    exit 1
else
    echo "✓ 找到模型: $MODEL_PATH"
fi

# 检查 GPU
echo "3. 检查 GPU 可用性..."
nvidia-smi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ 错误: 无法访问 GPU"
    exit 1
else
    echo "✓ GPU 可用"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

echo ""
echo "=========================================="
echo "快速测试配置"
echo "=========================================="
echo "训练样本数: 16"
echo "训练周期: 1"
echo "训练阶段: 8"
echo "预计训练时间: ~10-15分钟"
echo ""

read -p "是否开始测试训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "取消训练"
    exit 0
fi

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

# 设置环境变量
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="1"
export CUDA_VISIBLE_DEVICES=0

# 激活环境
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate align

# 设置路径
ROOT_DIR="/root/workspace/align"
SAVE_DIR="${ROOT_DIR}/sciqa_ckpts_test"

# 快速测试参数
TRAIN_NUM=16
EPOCH=1
PERIOD=1
SUB_STAGE=8
MAX_NEW_TOKENS=64
BATCH_SIZE=1

# 运行训练
python ${ROOT_DIR}/lord_train_mul.py \
    --dataset_task="scienceqa" \
    --use_lora=1 \
    --rank=32 \
    --lora_alpha=64 \
    --from_path="$MODEL_PATH" \
    --victim_path="gpt-4-vision-preview" \
    --is_black_box=1 \
    --sub_set_num=2 \
    --sub_stage_num=$SUB_STAGE \
    --infer_batch_size=1 \
    --tau1=0.80 \
    --tau2=0.85 \
    --task="LoRD-VI" \
    --device="cuda" \
    --epoch=$EPOCH \
    --period_num=$PERIOD \
    --acc_step=1 \
    --log_step=2 \
    --save_step=8 \
    --train_num=$TRAIN_NUM \
    --max_new_tokens=$MAX_NEW_TOKENS \
    --LR="3e-5" \
    --beta=1.0 \
    --temperature=1.5 \
    --batch_size=$BATCH_SIZE \
    --use_old_logits=1 \
    --use_vic_logits=1 \
    --use_kld=0 \
    --max_length=256 \
    --with_early_shut=0 \
    --save_path="$SAVE_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 训练完成！"
    echo "=========================================="
    echo ""
    echo "模型保存路径: $SAVE_DIR"
    echo ""
    echo "下一步:"
    echo "1. 合并 LoRA 权重:"
    echo "   python merge_lora_mul.py \\"
    echo "       --base_model $MODEL_PATH \\"
    echo "       --lora_path $SAVE_DIR \\"
    echo "       --save_path ${SAVE_DIR}/merged"
    echo ""
    echo "2. 使用合并后的模型进行推理"
    echo ""
    echo "3. 如需完整训练，运行:"
    echo "   bash scripts/6.0.sciqa_lord6_lora.sh"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败"
    echo "=========================================="
    echo ""
    echo "请检查错误信息并参考 README_MULTIMODAL.md"
    echo ""
fi
