#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/align_vq/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/llama3-llava-next-8b-hf}"
SCIENCEQA_PATH="${SCIENCEQA_PATH:-/root/autodl-tmp/datasets/ScienceQA}"
CONTROLS="${CONTROLS:-baseline,text_only_blank,hint_ablation,option_shuffle,random_image_swap,image_blur,image_downsample}"

STAGE2_ADAPTER="${STAGE2_ADAPTER:-/root/workspace/vq_lord/vq_lord_ckpts/stage2_vision_epoch15}"
STAGE3_ADAPTER="${STAGE3_ADAPTER:-/root/workspace/vq_lord/vq_lord_ckpts/stage3_sub1_period7}"
STAGE2_VQ="${STAGE2_VQ:-/root/workspace/vq_lord/vq_lord_ckpts/stage2_vision_epoch15/vq_codebook.pt}"
STAGE3_VQ="${STAGE3_VQ:-/root/workspace/vq_lord/vq_lord_ckpts/stage3_sub1_period7/vq_codebook.pt}"

RESULT_DIR="${RESULT_DIR:-/root/workspace/vq_lord/vq_lord_test_results/manual_scienceqa_ablation_exact}"
LOG_DIR="${LOG_DIR:-/root/workspace/vq_lord/vq_lord_test_results/manual_scienceqa_ablation_logs_exact}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
export PYTHONUNBUFFERED=1

unset OMP_NUM_THREADS || true
"${PYTHON_BIN}" /root/workspace/vq_lord/vq_lord3/eval_final/mm_scienceqa_control_eval_fast.py \
  --model_path "${MODEL_PATH}" \
  --adapter_path "${STAGE2_ADAPTER}" \
  --scienceqa_path "${SCIENCEQA_PATH}" \
  --split test \
  --max_samples 0 \
  --controls "${CONTROLS}" \
  --prompt_style simple \
  --answer_mode logits \
  --max_new_tokens 128 \
  --use_4bit 0 \
  --use_vq 1 \
  --vq_codebook_size 1024 \
  --freeze_vision_tower 0 \
  --vq_codebook_path "${STAGE2_VQ}" \
  --save_path "${RESULT_DIR}/scienceqa_ablation_stage2_exact.json" \
  > "${LOG_DIR}/scienceqa_ablation_stage2_exact.log" 2>&1

unset OMP_NUM_THREADS || true
"${PYTHON_BIN}" /root/workspace/vq_lord/vq_lord3/eval_final/mm_scienceqa_control_eval_fast.py \
  --model_path "${MODEL_PATH}" \
  --adapter_path "${STAGE3_ADAPTER}" \
  --scienceqa_path "${SCIENCEQA_PATH}" \
  --split test \
  --max_samples 0 \
  --controls "${CONTROLS}" \
  --prompt_style legacy \
  --answer_mode logits \
  --max_new_tokens 128 \
  --use_4bit 0 \
  --use_vq 1 \
  --vq_codebook_size 1024 \
  --freeze_vision_tower 0 \
  --vq_codebook_path "${STAGE3_VQ}" \
  --save_path "${RESULT_DIR}/scienceqa_ablation_stage3_exact.json" \
  > "${LOG_DIR}/scienceqa_ablation_stage3_exact.log" 2>&1
