#!/bin/bash
######################################################################
#6.0.SCIQA_LORD6_LORA ---

# MULTIMODAL TRAINING LORD6 on ScienceQA with LLaVA-Llama-3-8B

# Author: Adapted for multimodal learning
# Copyright © 2024, ZiLiang, all rights reserved.
# Created: December 2024
######################################################################

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=121
export HF_ENDPOINT="https://hf-mirror.com"

echo "HOME: ${HOME}"
export python=${HOME}/anaconda3/envs/align_v/bin/python3
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                                        --format=csv,noheader,nounits | \
                              awk -F ', ' '$2 < 100 && $3 == 0 {print $1}' | \
                              paste -sd ",")

# 如果没有找到空闲 GPU，默认使用 GPU 0
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "No idle GPUs found, using GPU 0"
else
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

export PYTHONIOENCODING=utf-8
export TORCH_USE_CUDA_DSA="1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少显存碎片
export root_dir="/root/workspace/align/"
export POD_save_dir="${root_dir}sciqa_ckpts/"

# 关键修改1：使用 LLaVA-Llama-3 模型路径
export from_path="/root/workspace/models/llama3-llava-next-8b-hf"

# 关键修改2：教师模型使用支持多模态的 API (如 GPT-4V)
export victim_model="gpt-4-vision-preview"

# 训练参数
export TRAIN_NUMS=(128)  # 可调整训练样本数量
export train_times=(1 2 3)

# 关键修改3：任务名称设置为 scienceqa
export task_ls=("scienceqa")
export train_taskls=("LoRD-VI")  # 使用 LoRD-VI 方法

export is_black_box=1  # 黑盒模式
export use_lora=1  # 使用 LoRA
export use_4bit=1  # 是否启用4bit量化 (1=启用, 0=禁用)

# 训练超参数
export epoch=2
export period=3  # 多个period可以让模型逐步学习
export sub_set_num=4  # 每次训练的子集数量
export sub_stage_num=64  # 训练阶段数
export max_new_tokens=64  # 生成的最大token数（进一步降低显存）
export infer_batch_size=1  # 推理批次大小（多模态通常为1）
export batch_size=1  # 训练批次大小

export beta=1.0
export temperature=1.5  # 生成温度

export use_old_logits=1
export use_vic_logits=1
export use_kld=0
export use_entropy=0

export tau1=0.80
export tau2=0.85

# 关键修改4：序列长度适应多模态输入
# LLaVA-Next 使用动态分辨率，每个图像可能有多个patches
# 显存极限：22GB GPU 只能支持较短序列，会截断大部分 image tokens
# 这会导致性能下降，但是唯一能运行的配置
export msl=1024  # 最大序列长度 (极限压缩)
export max_length=1024

# LoRA 参数（适配 LLaVA 架构，降低以节省显存）
export rank=32
export lora_alpha=64

# 学习率
export LR="3e-5"

for train_num in ${TRAIN_NUMS[*]}
do
    for train_time in ${train_times[*]}
    do
        for task in ${task_ls[*]}
        do
            for train_task in ${train_taskls[*]}
            do
                echo "====================================================="
                echo "+++++++train_num: ${train_num}+++++++"
                echo "+++++++train_time: ${train_time}+++++++"
                echo "+++++++task: ${task} (MULTIMODAL)+++++++"
                echo "+++++++train_task: ${train_task}+++++++"
                echo "+++++++model: LLaVA-Llama-3-8B+++++++"
                echo "====================================================="

                export save_path="${POD_save_dir}SCIQA${task}${train_num}${train_time}${train_task}"

                # 关键修改5：调用 lord_train_mul.py（多模态训练脚本）
                $python ${root_dir}/sciqa_lord6_lora/lord_train_mul.py\
                    --dataset_task=$task \
                    --use_lora=$use_lora \
                    --use_4bit=$use_4bit \
                    --rank=$rank \
                    --lora_alpha=$lora_alpha \
                    --from_path=$from_path \
                    --victim_path=$victim_model \
                    --is_black_box=$is_black_box \
                    --sub_set_num=$sub_set_num \
                    --sub_stage_num=$sub_stage_num\
                    --infer_batch_size=$infer_batch_size\
                    --tau1=$tau1 \
                    --tau2=$tau2 \
                    --task=$train_task \
                    --device="cuda" \
                    --epoch=$epoch \
                    --period_num=$period \
                    --acc_step=1 \
                    --log_step=10 \
                    --save_step=32 \
                    --train_num=$train_num \
                    --max_new_tokens=$max_new_tokens \
                    --LR=$LR \
                    --beta=$beta \
                    --temperature=$temperature \
                    --batch_size=$batch_size \
                    --use_old_logits=$use_old_logits\
                    --use_vic_logits=$use_vic_logits\
                    --use_kld=$use_kld\
                    --max_length=$max_length \
                    --with_early_shut=0 \
                    --save_path=$save_path
                    
                echo "DONE FOR ONE TRAINING RUN...."
            done
        done
    done
done

echo "NOW BEGIN TO INFERENCE ON ScienceQA..."
# 关键修改6：调用多模态推理（如果需要）
# $python ${root_dir}sciqa_process.py

echo "RUNNING 6.0.sciqa_lord6_lora.sh DONE."
# 6.0.sciqa_lord6_lora.sh ends here
