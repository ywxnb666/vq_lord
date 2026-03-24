#!/bin/bash
######################################################################
# TEST_VQ_LORD.SH ---
#
# VQ-LoRD 测评脚本
# 直接调用 vq_lord3/sciqa_process.py 做 ScienceQA 评估
######################################################################

# CUDA 环境配置
# export CUDA_HOME=/usr/local/cuda-12.1
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
    default_script_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/scripts"
    default_root_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/"
    default_save_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts_stage2_tune"
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
export python="${PYTHON_BIN:-$default_python}"

export CUDA_VISIBLE_DEVICES=3

export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 项目路径
# 自动定位到当前脚本所在仓库根目录，避免误指向旧工程（旧工程的 sciqa_process.py 不支持 --scienceqa_path）
export script_dir="${default_script_dir}"
export root_dir="${ROOT_DIR:-$default_root_dir}"
export save_dir="${SAVE_DIR:-$default_save_dir}"
export data_dir="${root_dir}vq_lord_data/"
export scienceqa_path="${SCIENCEQA_PATH:-$default_scienceqa_path}"
export vq_codebook_path="${save_dir}/round1_e3/stage2_vision_epoch15/vq_codebook.pt"

# 模型路径
export model_path="${MODEL_PATH:-$default_model_path}"
export scienceqa_split="${SCIENCEQA_SPLIT:-validation}"
export adapter_path="${save_dir}/round1_e3/stage2_vision_epoch15"
export eval_max_samples=2097
export eval_max_new_tokens=64
export use_4bit=0
export use_vq=1
export vq_codebook_size=1024
export freeze_vision_tower="${FREEZE_VISION_TOWER:-0}"
export eval_answer_mode=logits
export save_path="${SAVE_PATH:-${save_dir}/sciqa_eval_${scienceqa_split}_${eval_answer_mode}.json}"

# 日志和保存
echo "======================================================"
echo "VQ-LoRD 测试开始"
echo "======================================================"
echo "路径模式: $SERVER_NAME (USE_H200_PATHS=$USE_H200_PATHS)"
echo "python: $python"
echo "root_dir: $root_dir"
echo "模型路径: $model_path"
echo "ScienceQA 路径: $scienceqa_path"
echo "split: $scienceqa_split"
echo "Adapter 路径: $adapter_path"
echo "VQ codebook 路径: $vq_codebook_path"
echo "save_path: $save_path"
echo "======================================================"

# 创建必要目录
mkdir -p $save_dir
mkdir -p $data_dir

# 校验测试入口是否存在
if [ ! -f "${root_dir}vq_lord3/sciqa_process.py" ]; then
    echo "错误: 未找到测试入口 ${root_dir}vq_lord3/sciqa_process.py"
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

$python "${root_dir}vq_lord3/sciqa_process.py" \
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

echo "======================================================"
echo "ScienceQA 验证完成"
echo "结果保存在: $save_path"
echo "======================================================"
