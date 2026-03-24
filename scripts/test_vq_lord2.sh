#!/bin/bash
######################################################################
# TEST_VQ_LORD2.SH ---
#
# VQ-LoRD 测评脚本（旧 Stage3 prompt 口径）
# 直接调用 vq_lord3/sciqa_process2.py 做 ScienceQA 评估
# 默认顺序评测 stage3 的 period1 和 period2 产物
######################################################################

set -euo pipefail

# CUDA 环境配置
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export BNB_CUDA_VERSION=121
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

echo "HOME: ${HOME}"

# 路径模式切换：0=A800(/root), 1=H200(/inspire)
export USE_H200_PATHS=1
if [ "${USE_H200_PATHS}" = "1" ]; then
    export SERVER_NAME="H200"
    default_python="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/.align_vq/bin/python"
    default_root_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/"
    default_run_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts_stage2_tune/round1_e3"
    default_model_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf"
    default_scienceqa_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/datasets/ScienceQA"
else
    export SERVER_NAME="A800"
    default_python="/root/autodl-tmp/conda/envs/align_vq/bin/python"
    default_root_dir="/root/workspace/align_vq/"
    default_run_dir="/root/workspace/align_vq/stage3_tune"
    default_model_path="/root/autodl-tmp/models/llama3-llava-next-8b-hf"
    default_scienceqa_path="/root/autodl-tmp/datasets/ScienceQA"
fi

export python="${PYTHON_BIN:-$default_python}"
export CUDA_VISIBLE_DEVICES=3
export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 项目路径
export root_dir="${ROOT_DIR:-$default_root_dir}"
export run_dir="${RUN_DIR:-$default_run_dir}"
export scienceqa_path="${SCIENCEQA_PATH:-$default_scienceqa_path}"
export model_path="${MODEL_PATH:-$default_model_path}"
export scienceqa_split="${SCIENCEQA_SPLIT:-validation}"
export eval_max_samples=0
export eval_max_new_tokens="${EVAL_MAX_NEW_TOKENS:-64}"
export use_4bit="${USE_4BIT:-0}"
export use_vq="${USE_VQ:-1}"
export vq_codebook_size="${VQ_CODEBOOK_SIZE:-1024}"
export freeze_vision_tower="${FREEZE_VISION_TOWER:-0}"
export eval_answer_mode="${EVAL_ANSWER_MODE:-logits}"
export periods=5

echo "======================================================"
echo "VQ-LoRD2 测试开始"
echo "======================================================"
echo "路径模式: $SERVER_NAME (USE_H200_PATHS=$USE_H200_PATHS)"
echo "python: $python"
echo "root_dir: $root_dir"
echo "run_dir: $run_dir"
echo "模型路径: $model_path"
echo "ScienceQA 路径: $scienceqa_path"
echo "split: $scienceqa_split"
echo "periods: $periods"
echo "answer_mode: $eval_answer_mode"
echo "======================================================"

if [ ! -f "${root_dir}vq_lord3/sciqa_process2.py" ]; then
    echo "错误: 未找到测试入口 ${root_dir}vq_lord3/sciqa_process2.py"
    echo "当前 root_dir=${root_dir}"
    exit 1
fi

mkdir -p "$run_dir"

for period in $periods; do
    adapter_path="${ADAPTER_PATH_OVERRIDE:-${run_dir}/stage2_vision_epoch15}"
    vq_codebook_path="${VQ_CODEBOOK_PATH_OVERRIDE:-${adapter_path}/vq_codebook.pt}"
    save_path="${SAVE_PATH_OVERRIDE:-${run_dir}/test_eval_period${period}_${eval_answer_mode}_legacy.json}"

    echo "------------------------------------------------------"
    echo "开始评测 stage3_sub1_period${period}"
    echo "adapter_path: $adapter_path"
    echo "vq_codebook_path: $vq_codebook_path"
    echo "save_path: $save_path"
    echo "------------------------------------------------------"

    if [ ! -d "$adapter_path" ]; then
        echo "错误: 未找到 adapter 目录: $adapter_path"
        exit 1
    fi

    if [ ! -f "${vq_codebook_path}" ]; then
        echo "警告: 未找到 VQ codebook: ${vq_codebook_path}"
        echo "将继续运行，但结果中 vq_codebook_loaded 很可能为 false"
    fi

    "$python" "${root_dir}vq_lord3/sciqa_process2.py" \
        --model_path="$model_path" \
        --adapter_path="$adapter_path" \
        --scienceqa_path="$scienceqa_path" \
        --split="$scienceqa_split" \
        --max_samples="$eval_max_samples" \
        --max_new_tokens="$eval_max_new_tokens" \
        --use_4bit="$use_4bit" \
        --use_vq="$use_vq" \
        --vq_codebook_size="$vq_codebook_size" \
        --freeze_vision_tower="$freeze_vision_tower" \
        --vq_codebook_path="$vq_codebook_path" \
        --answer_mode="$eval_answer_mode" \
        --save_path="$save_path"
done

echo "======================================================"
echo "VQ-LoRD2 测试完成"
echo "run_dir: $run_dir"
echo "======================================================"
