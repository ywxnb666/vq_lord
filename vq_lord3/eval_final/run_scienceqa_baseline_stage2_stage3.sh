#!/usr/bin/env bash
set -euo pipefail

unset OMP_NUM_THREADS || true
mkdir -p /root/workspace/vq_lord/vq_lord_test_results/manual_scienceqa_baseline
mkdir -p /root/workspace/vq_lord/vq_lord_test_results/manual_scienceqa_logs

/root/autodl-tmp/conda/envs/align_vq/bin/python -u /root/workspace/vq_lord/vq_lord3/eval_final/mm_scienceqa_control_eval_fast.py \
  --model_path /root/autodl-tmp/models/llama3-llava-next-8b-hf \
  --adapter_path /root/workspace/vq_lord/vq_lord_ckpts/stage2_vision_epoch15 \
  --use_vq 1 \
  --vq_codebook_size 1024 \
  --vq_codebook_path /root/workspace/vq_lord/vq_lord_ckpts/stage2_vision_epoch15/vq_codebook.pt \
  --scienceqa_path /root/autodl-tmp/datasets/ScienceQA \
  --split test \
  --max_samples 0 \
  --controls baseline \
  --prompt_style simple \
  --mcq_batch_size 8 \
  --save_path /root/workspace/vq_lord/vq_lord_test_results/manual_scienceqa_baseline/scienceqa_baseline_stage2_current.json \
  > /root/workspace/vq_lord/vq_lord_test_results/manual_scienceqa_logs/scienceqa_baseline_stage2_current.log 2>&1

unset OMP_NUM_THREADS || true
/root/autodl-tmp/conda/envs/align_vq/bin/python -u /root/workspace/vq_lord/vq_lord3/eval_final/mm_scienceqa_control_eval_fast.py \
  --model_path /root/autodl-tmp/models/llama3-llava-next-8b-hf \
  --adapter_path /root/workspace/vq_lord/vq_lord_ckpts/stage3_sub1_period7 \
  --use_vq 1 \
  --vq_codebook_size 1024 \
  --vq_codebook_path /root/workspace/vq_lord/vq_lord_ckpts/stage3_sub1_period7/vq_codebook.pt \
  --scienceqa_path /root/autodl-tmp/datasets/ScienceQA \
  --split test \
  --max_samples 0 \
  --controls baseline \
  --prompt_style legacy \
  --mcq_batch_size 8 \
  --save_path /root/workspace/vq_lord/vq_lord_test_results/manual_scienceqa_baseline/scienceqa_baseline_stage3_current.json \
  > /root/workspace/vq_lord/vq_lord_test_results/manual_scienceqa_logs/scienceqa_baseline_stage3_current.log 2>&1
