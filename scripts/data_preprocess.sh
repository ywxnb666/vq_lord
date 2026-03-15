#!/bin/bash
######################################################################
# DATA_PREPROCESS.SH
#
# ScienceQA 训练前预处理脚本
# - 对齐 train_vq_lord.sh 的关键参数
# - 调用 data_preprocess/sciqa_preprocess.py 生成分桶计划 JSON
######################################################################

set -e

# 环境变量
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Python 路径（可被外部覆盖）
# 路径模式切换：0=A800(/root), 1=H200(/inspire)
export USE_H200_PATHS=0
if [ "${USE_H200_PATHS}" = "1" ]; then
    export SERVER_NAME="H200"
    default_python="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/.align_vq/bin/python"
    default_script_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/scripts"
    default_root_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/"
    default_save_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/vq_lord_ckpts"
    default_model_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/models/llama3-llava-next-8b-hf"
    default_scienceqa_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/datasets/ScienceQA"
else
    export SERVER_NAME="A800"
    default_python="/root/autodl-tmp/conda/envs/align_vq/bin/python"
    default_script_dir="/root/workspace/align_vq/scripts"
    default_root_dir="/root/workspace/align_vq/"
    default_save_dir="/root/autodl-tmp/vq_lord_ckpts"
    default_model_path="/root/autodl-tmp/models/llama3-llava-next-8b-hf"
    default_scienceqa_path="/root/autodl-tmp/datasets/ScienceQA"
fi
export python="${default_python}"

# 项目路径
export script_dir="${default_script_dir}"
export root_dir="${default_root_dir}"
export save_dir="${default_save_dir}"
export preprocess_dir="${save_dir}/preprocess"

# 模型与数据参数（默认对齐 train_vq_lord.sh）
export model_path="${default_model_path}"
export train_num="${train_num:-0}"
export scienceqa_split="${scienceqa_split:-train}"
export scienceqa_seed="${scienceqa_seed:-20240306}"
export bucket_by="${bucket_by:-patches}"                # patches / size / none
export bucket_batch_size="${bucket_batch_size:-4}"
export bucket_drop_last="${bucket_drop_last:-0}"        # 0=保留尾包, 1=丢弃
export shuffle="${shuffle:-1}"                          # 0=不打乱, 1=打乱
export preview_buckets="${preview_buckets:-10}"
export preview_batches="${preview_batches:-10}"

# ScienceQA 数据路径（和 train_vq_lord.sh 一致：优先环境变量）
if [ -n "${SCIENCEQA_PATH:-}" ]; then
    export scienceqa_path="$SCIENCEQA_PATH"
elif [ -n "${scienceqa_path:-}" ]; then
    export scienceqa_path="$scienceqa_path"
elif [ -d "${root_dir}downloads/dataset/ScienceQA" ]; then
    export scienceqa_path="${root_dir}downloads/dataset/ScienceQA"
elif [ -d "${root_dir}downloads/datasets/ScienceQA" ]; then
    export scienceqa_path="${root_dir}downloads/datasets/ScienceQA"
elif [ -d "$default_scienceqa_path" ]; then
    export scienceqa_path="$default_scienceqa_path"
else
    export scienceqa_path="ScienceQA"
fi

export scienceqa_preprocessed_path="${scienceqa_preprocessed_path:-${preprocess_dir}/scienceqa_${scienceqa_split}_n${train_num}_seed${scienceqa_seed}_${bucket_by}_bs${bucket_batch_size}.json}"

echo "======================================================"
echo "ScienceQA 预处理开始"
echo "======================================================"
echo "路径模式: $SERVER_NAME (USE_H200_PATHS=$USE_H200_PATHS)"
echo "python: $python"
echo "root_dir: $root_dir"
echo "model_path: $model_path"
echo "scienceqa_path: $scienceqa_path"
echo "split: $scienceqa_split"
echo "train_num: $train_num"
echo "seed: $scienceqa_seed"
echo "bucket_by: $bucket_by"
echo "bucket_batch_size: $bucket_batch_size"
echo "bucket_drop_last: $bucket_drop_last"
echo "shuffle: $shuffle"
echo "save_json: $scienceqa_preprocessed_path"
echo "======================================================"

mkdir -p "$save_dir"
mkdir -p "$preprocess_dir"

if [ ! -f "${root_dir}data_preprocess/sciqa_preprocess.py" ]; then
    echo "错误: 未找到预处理脚本 ${root_dir}data_preprocess/sciqa_preprocess.py"
    exit 1
fi

if [ "$scienceqa_path" = "ScienceQA" ] && [ "${HF_DATASETS_OFFLINE:-0}" = "1" ]; then
    echo "错误: 当前启用了 HF_DATASETS_OFFLINE=1，但未找到本地 ScienceQA 数据集目录。"
    echo "请设置 SCIENCEQA_PATH=/你的本地/ScienceQA"
    exit 1
fi

"$python" "${root_dir}data_preprocess/sciqa_preprocess.py" \
    --dataset-path="$scienceqa_path" \
    --model-path="$model_path" \
    --split="$scienceqa_split" \
    --train-num="$train_num" \
    --seed="$scienceqa_seed" \
    --bucket-by="$bucket_by" \
    --bucket-batch-size="$bucket_batch_size" \
    --bucket-drop-last="$bucket_drop_last" \
    --shuffle="$shuffle" \
    --preview-buckets="$preview_buckets" \
    --preview-batches="$preview_batches" \
    --save-json="$scienceqa_preprocessed_path"

if [ ! -f "$scienceqa_preprocessed_path" ]; then
    echo "错误: 预处理结果不存在 $scienceqa_preprocessed_path"
    exit 1
fi

echo "======================================================"
echo "ScienceQA 预处理完成"
echo "输出文件: $scienceqa_preprocessed_path"
echo "训练时请传入: --scienceqa_preprocessed_path=$scienceqa_preprocessed_path"
echo "======================================================"
