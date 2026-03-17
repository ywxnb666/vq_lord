#!/bin/bash
######################################################################
# TEST_ORIGIN.SH
#
# 原生学生模型测试脚本
# - 与 test_vq_lord.sh 基本一致
# - 唯一区别：不加载 VQ、不加载任何微调 adapter
######################################################################

# CUDA 环境配置
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=121
export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_OFFLINE=1

echo "HOME: ${HOME}"

# 路径模式切换：0=A800(/root), 1=H200(/inspire)
export USE_H200_PATHS=1
if [ "${USE_H200_PATHS}" = "1" ]; then
    export SERVER_NAME="H200"
    default_python="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/.align_vq/bin/python"
    default_script_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/scripts"
    default_root_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/"
    default_save_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts/"
    default_model_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf"
    default_scienceqa_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/datasets/ScienceQA"
else
    export SERVER_NAME="A800"
    default_python="/root/autodl-tmp/conda/envs/align_vq/bin/python"
    default_script_dir="/root/workspace/align_vq/scripts"
    default_root_dir="/root/workspace/align_vq/"
    default_save_dir="/root/autodl-tmp/vq_lord_ckpts/"
    default_model_path="/root/autodl-tmp/models/llama3-llava-next-8b-hf"
    default_scienceqa_path="/root/autodl-tmp/datasets/ScienceQA"
fi
export python="${default_python}"

export CUDA_VISIBLE_DEVICES=1
export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 项目路径
export script_dir="${default_script_dir}"
export root_dir="${default_root_dir}"
export save_dir="${default_save_dir}"
export data_dir="${root_dir}vq_lord_data/"

# 模型/数据
export model_path="${default_model_path}"
export scienceqa_path="${default_scienceqa_path}"
export scienceqa_eval_split="validation"

# 评测参数
export eval_max_samples=500
export eval_max_new_tokens=128
export use_4bit=0
export answer_mode="hybrid"
export save_path="${save_dir}/sciqa_eval_origin.json"

echo "======================================================"
echo "原生学生模型测试开始"
echo "======================================================"
echo "路径模式: $SERVER_NAME (USE_H200_PATHS=$USE_H200_PATHS)"
echo "python: $python"
echo "模型路径: $model_path"
echo "ScienceQA 路径: $scienceqa_path"
echo "split: $scienceqa_eval_split"
echo "max_samples: $eval_max_samples"
echo "answer_mode: $answer_mode"
echo "保存路径: $save_path"
echo "======================================================"

mkdir -p "$save_dir"
mkdir -p "$data_dir"

if [ ! -f "${root_dir}vq_lord2/sciqa_process.py" ]; then
    echo "错误: 未找到测试入口 ${root_dir}vq_lord2/sciqa_process.py"
    echo "当前 root_dir=${root_dir}"
    exit 1
fi

"$python" "${root_dir}vq_lord2/sciqa_process.py" \
    --model_path="$model_path" \
    --adapter_path="" \
    --scienceqa_path="$scienceqa_path" \
    --split="$scienceqa_eval_split" \
    --max_samples="$eval_max_samples" \
    --max_new_tokens="$eval_max_new_tokens" \
    --use_4bit="$use_4bit" \
    --use_vq=0 \
    --vq_codebook_size=1024 \
    --freeze_vision_tower=0 \
    --vq_codebook_path="" \
    --answer_mode="$answer_mode" \
    --save_path="$save_path"

echo "======================================================"
echo "原生学生模型测试完成"
echo "结果保存在: $save_path"
echo "======================================================"
