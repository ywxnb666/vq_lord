#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/align_vq/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/llama3-llava-next-8b-hf}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/workspace/vq_lord/vq_lord_ckpts/stage3_sub1_period7}"
SCIENCEQA_PATH="${SCIENCEQA_PATH:-/root/autodl-tmp/datasets/ScienceQA}"
VQ_CODEBOOK_PATH="${VQ_CODEBOOK_PATH:-/root/workspace/vq_lord/vq_lord_ckpts/stage3_sub1_period7/vq_codebook.pt}"
RESULT_DIR="${RESULT_DIR:-/root/workspace/vq_lord/vq_lord_test_results}"

mkdir -p "${RESULT_DIR}"
unset OMP_NUM_THREADS || true
export PYTHONUNBUFFERED=1

printf '[Launcher] Starting accelerated full evaluation\n'
printf '[Launcher] Result dir: %s\n' "${RESULT_DIR}"

printf '[Launcher] Stage 1: special benchmarks\n'
"${PYTHON_BIN}" -u /root/workspace/vq_lord/vq_lord3/eval_final/mm_special_benchmark_eval_fast.py \
  --model_path "${MODEL_PATH}" \
  --adapter_path "${ADAPTER_PATH}" \
  --use_4bit 0 \
  --use_vq 1 \
  --vq_codebook_size 1024 \
  --vq_codebook_path "${VQ_CODEBOOK_PATH}" \
  --benchmarks ai2d,chartqa \
  --prompt_style legacy \
  --max_new_tokens 32 \
  --mcq_batch_size 8 \
  --open_batch_size 4 \
  --save_path "${RESULT_DIR}/mm_special_benchmarks_full_fast.json"

printf '[Launcher] Stage 2: scienceqa controls\n'
"${PYTHON_BIN}" -u /root/workspace/vq_lord/vq_lord3/eval_final/mm_scienceqa_control_eval_fast.py \
  --model_path "${MODEL_PATH}" \
  --adapter_path "${ADAPTER_PATH}" \
  --use_vq 1 \
  --vq_codebook_size 1024 \
  --vq_codebook_path "${VQ_CODEBOOK_PATH}" \
  --scienceqa_path "${SCIENCEQA_PATH}" \
  --split test \
  --prompt_style legacy \
  --mcq_batch_size 8 \
  --save_path "${RESULT_DIR}/scienceqa_control_suite_full_fast.json"

printf '[Launcher] Stage 3: report aggregation\n'
"${PYTHON_BIN}" -u /root/workspace/vq_lord/vq_lord3/eval_final/mm_eval_suite_report.py \
  --benchmark_result "${RESULT_DIR}/mm_special_benchmarks_full_fast.json" \
  --control_result "${RESULT_DIR}/scienceqa_control_suite_full_fast.json" \
  --save_json "${RESULT_DIR}/mm_eval_suite_report_full_fast.json" \
  --save_md "${RESULT_DIR}/mm_eval_suite_report_full_fast.md"

printf '[Launcher] All stages completed\n'
