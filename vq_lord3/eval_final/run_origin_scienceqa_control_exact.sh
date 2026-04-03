#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/align_vq/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/llama3-llava-next-8b-hf}"
SCIENCEQA_PATH="${SCIENCEQA_PATH:-/root/autodl-tmp/datasets/ScienceQA}"
RESULT_DIR="${RESULT_DIR:-/root/workspace/vq_lord/vq_lord_test_results/final_origin}"
CONTROLS="${CONTROLS:-baseline,text_only_blank,hint_ablation,option_shuffle,random_image_swap,image_blur,image_downsample}"
SAVE_PATH="${SAVE_PATH:-${RESULT_DIR}/origin_scienceqa_control_suite_exact.json}"

mkdir -p "${RESULT_DIR}"
unset OMP_NUM_THREADS || true
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -u /root/workspace/vq_lord/vq_lord3/eval_final/mm_scienceqa_control_eval_fast.py \
  --model_path "${MODEL_PATH}" \
  --adapter_path "" \
  --scienceqa_path "${SCIENCEQA_PATH}" \
  --split test \
  --max_samples 0 \
  --controls "${CONTROLS}" \
  --prompt_style legacy \
  --answer_mode logits \
  --max_new_tokens 64 \
  --use_4bit 0 \
  --use_vq 0 \
  --vq_codebook_size 1024 \
  --freeze_vision_tower 0 \
  --vq_codebook_path "" \
  --save_path "${SAVE_PATH}"
