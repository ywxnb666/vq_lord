#!/bin/bash
######################################################################
# 7.0.VQ_LORD_TRAIN.SH ---
#
# VQ-LoRD 训练脚本
# 用于窃取多模态模型的图像识别能力
#
# 测试
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

echo "HOME: ${HOME}"

# 路径模式切换：0=A800(/root), 1=H200(/inspire)
export USE_H200_PATHS=0
if [ "${USE_H200_PATHS}" = "1" ]; then
    export SERVER_NAME="H200"
    default_python="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/.align_vq/bin/python"
    default_script_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/scripts"
    default_root_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/"
    default_save_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts/"
    default_model_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf"
else
    export SERVER_NAME="A800"
    default_python="/root/autodl-tmp/conda/envs/align_vq/bin/python"
    default_script_dir="/root/workspace/align_vq/scripts"
    default_root_dir="/root/workspace/align_vq/"
    default_save_dir="/root/autodl-tmp/vq_lord_ckpts/"
    default_model_path="/root/autodl-tmp/models/llama3-llava-next-8b-hf"
fi
export python="${default_python}"

# 自动选择空闲 GPU
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
#                                         --format=csv,noheader,nounits | \
#                               awk -F ', ' '$2 < 100 && $3 == 0 {print $1}' | \
#                               paste -sd ",")

# if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
#     export CUDA_VISIBLE_DEVICES=0
#     echo "未找到空闲 GPU，使用 GPU 0"
# else
#     echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
# fi

export CUDA_VISIBLE_DEVICES=0

export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 项目路径
# 自动定位到当前脚本所在仓库根目录，避免误指向旧工程（旧工程的 sciqa_process.py 不支持 --scienceqa_path）
export script_dir="${default_script_dir}"
export root_dir="${default_root_dir}"
# export root_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/"
export save_dir="${default_save_dir}"
export data_dir="${root_dir}vq_lord_data/"
export vq_codebook_path="${save_dir}stage2_vision/vq_codebook.pt"  # VQ Codebook 保存路径

# 模型路径
export model_path="${default_model_path}"
export scienceqa_split="train"  # ScienceQA split
export scienceqa_eval_split="validation"  # ScienceQA 验证/测试 split
export adapter_path="${save_dir}/stage3_lord_final"
export eval_max_samples=200
export eval_max_new_tokens=128
export use_4bit=1
export use_vq=1
export vq_codebook_size=8192
export freeze_vision_tower=0
export save_path="${save_dir}/sciqa_eval.json"

# 日志和保存
echo "======================================================"
echo "VQ-LoRD 测试开始"
echo "======================================================"
echo "路径模式: $SERVER_NAME (USE_H200_PATHS=$USE_H200_PATHS)"
echo "python: $python"
echo "模型路径: $model_path"
echo "Adapter 路径: $adapter_path"
echo "VQ codebook 路径: $vq_codebook_path"
echo "======================================================"

# 创建必要目录
mkdir -p $save_dir
mkdir -p $data_dir

# 校验测试入口是否存在
if [ ! -f "${root_dir}vq_lord/sciqa_process.py" ]; then
    echo "错误: 未找到测试入口 ${root_dir}vq_lord/sciqa_process.py"
    echo "当前 root_dir=${root_dir}"
    exit 1
fi

# 校验 VQ codebook 是否存在（不存在时 sciqa_process 会记录 vq_codebook_loaded=false）
if [ ! -f "${vq_codebook_path}" ]; then
    echo "警告: 未找到 VQ codebook: ${vq_codebook_path}"
    echo "将继续运行，但结果中 vq_codebook_loaded 很可能为 false"
fi

echo "======================================================"
echo "开始 ScienceQA 验证"
echo "======================================================"

$python ${root_dir}vq_lord/sciqa_process.py \
    --model_path=$model_path \
    --adapter_path=$adapter_path \
    --split=$scienceqa_eval_split \
    --max_samples=$eval_max_samples \
    --max_new_tokens=$eval_max_new_tokens \
    --use_4bit=$use_4bit \
    --use_vq=$use_vq \
    --vq_codebook_size=$vq_codebook_size \
    --freeze_vision_tower=$freeze_vision_tower \
    --vq_codebook_path=$vq_codebook_path \
    --save_path=$save_path

echo "======================================================"
echo "ScienceQA 验证完成"
echo "结果保存在: $save_path"
echo "======================================================"
