#!/bin/bash
set -euo pipefail

######################################################################
# Collect full ScienceQA train teacher annotations (schema v2)
# Output cache filename uses suffix: *_new.json
######################################################################

# export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12}"
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

export USE_H200_PATHS=1
if [ "${USE_H200_PATHS}" = "1" ]; then
    default_python="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/.align_vq/bin/python"
    default_root_dir="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/align/"
    default_scienceqa_path="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/align_vq/downloads/datasets/ScienceQA"
else
    default_python="/root/autodl-tmp/conda/envs/align_vq/bin/python"
    default_root_dir="/root/workspace/align_vq/"
    default_scienceqa_path="/root/autodl-tmp/datasets/ScienceQA"
fi

export python_bin="${PYTHON_BIN:-$default_python}"
export root_dir="${ROOT_DIR:-$default_root_dir}"
export scienceqa_path="${SCIENCEQA_PATH:-$default_scienceqa_path}"
export data_dir="${DATA_DIR:-${root_dir}vq_lord_data/}"

export scienceqa_split="${SCIENCEQA_SPLIT:-train}"
export train_num="${TRAIN_NUM:-0}"                     # 0 = full split
export scienceqa_seed="${SCIENCEQA_SEED:-20240306}"

export victim_model="qwen3.5-plus"
export teacher_lang="en"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="sk-abc8c59df2d64b7ba22718eae4fe80c2"
export teacher_api_base="${TEACHER_API_BASE:-${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}}"
export teacher_api_key="${TEACHER_API_KEY:-${OPENAI_API_KEY:-}}"

export collect_teacher_data="${COLLECT_TEACHER_DATA:-1}"
export strict_teacher_distill="${STRICT_TEACHER_DISTILL:-1}"
export teacher_cache_path="${TEACHER_CACHE_PATH:-}"

export teacher_observed_max_tokens="${TEACHER_OBSERVED_MAX_TOKENS:-256}"
export teacher_context_max_tokens="${TEACHER_CONTEXT_MAX_TOKENS:-192}"
export teacher_reasoning_max_tokens="${TEACHER_REASONING_MAX_TOKENS:-256}"
export teacher_answer_max_tokens="${TEACHER_ANSWER_MAX_TOKENS:-64}"
export teacher_max_new_tokens_total="${TEACHER_MAX_NEW_TOKENS_TOTAL:-768}"
export num_workers="${NUM_WORKERS:-4}"

if [ -z "${teacher_cache_path}" ]; then
    victim_tag="$(echo "${victim_model}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
    teacher_cache_path="${data_dir}scienceqa_teacher_${victim_tag}_${scienceqa_split}_n${train_num}_seed${scienceqa_seed}_new.json"
fi

mkdir -p "${data_dir}"

entry="${root_dir}vq_lord3/data_collector2.py"
if [ ! -f "${entry}" ]; then
    echo "错误: 未找到入口脚本 ${entry}"
    exit 1
fi

if [ "${collect_teacher_data}" = "1" ] && [ -z "${teacher_api_key}" ]; then
    echo "错误: COLLECT_TEACHER_DATA=1 但未提供 TEACHER_API_KEY / OPENAI_API_KEY"
    exit 1
fi

echo "======================================================"
echo "Collect Teacher Annotations (ScienceQA train full)"
echo "======================================================"
echo "python: ${python_bin}"
echo "entry: ${entry}"
echo "scienceqa_path: ${scienceqa_path}"
echo "split/train_num/seed: ${scienceqa_split}/${train_num}/${scienceqa_seed}"
echo "victim_model/lang: ${victim_model}/${teacher_lang}"
echo "collect/strict: ${collect_teacher_data}/${strict_teacher_distill}"
echo "cache_path: ${teacher_cache_path}"
echo "num_workers: ${num_workers}"
echo "budget(obs/ctx/reason/ans/total): ${teacher_observed_max_tokens}/${teacher_context_max_tokens}/${teacher_reasoning_max_tokens}/${teacher_answer_max_tokens}/${teacher_max_new_tokens_total}"
echo "======================================================"

"${python_bin}" "${entry}" \
    --task="collect_scienceqa_struct" \
    --scienceqa_path="${scienceqa_path}" \
    --scienceqa_split="${scienceqa_split}" \
    --train_num="${train_num}" \
    --scienceqa_seed="${scienceqa_seed}" \
    --max_samples=0 \
    --data_dir="${data_dir}" \
    --teacher_cache_path="${teacher_cache_path}" \
    --victim_model="${victim_model}" \
    --teacher_lang="${teacher_lang}" \
    --teacher_api_base="${teacher_api_base}" \
    --teacher_api_key="${teacher_api_key}" \
    --collect_teacher_data="${collect_teacher_data}" \
    --strict_teacher_distill="${strict_teacher_distill}" \
    --teacher_observed_max_tokens="${teacher_observed_max_tokens}" \
    --teacher_context_max_tokens="${teacher_context_max_tokens}" \
    --teacher_reasoning_max_tokens="${teacher_reasoning_max_tokens}" \
    --teacher_answer_max_tokens="${teacher_answer_max_tokens}" \
    --teacher_max_new_tokens_total="${teacher_max_new_tokens_total}" \
    --max_retries=10 \
    --num_workers="${num_workers}" \
    --save_every=10 \
    --sleep_sec=0

echo "======================================================"
echo "教师标注采集完成"
echo "输出缓存: ${teacher_cache_path}"
echo "======================================================"
