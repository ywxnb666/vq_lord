#!/bin/bash
set -euo pipefail

source /root/workspace/vq_lord/scripts2/common.sh
align_vq_init_paths
align_vq_ensure_runtime_dirs

VICTIM_MODEL="${VICTIM_MODEL:-qwen3.5-flash-2026-02-23}"
TEACHER_API_BASE="${TEACHER_API_BASE:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}}}"
TEACHER_API_KEY="${TEACHER_API_KEY:-${OPENAI_API_KEY:-}}"
TEACHER_ENABLE_THINKING="${TEACHER_ENABLE_THINKING:-false}"
RESULT_DIR="${RESULT_DIR:-/root/workspace/vq_lord/vq_lord_test_results/teacher_qwen35flash_compare}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
PROMPT_STYLE="${PROMPT_STYLE:-legacy}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SLEEP_SEC="${SLEEP_SEC:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
SCIENCEQA_PATH="${SCIENCEQA_PATH:-/root/autodl-tmp/datasets/ScienceQA}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/align_vq/bin/python}"

export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
unset DATASETS_OFFLINE || true
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

mkdir -p "${RESULT_DIR}"

printf '[Launcher] Starting teacher full evaluation\n'
printf '[Launcher] Result dir: %s\n' "${RESULT_DIR}"
printf '[Launcher] Victim model: %s\n' "${VICTIM_MODEL}"
printf '[Launcher] Teacher thinking: %s\n' "${TEACHER_ENABLE_THINKING}"
printf '[Launcher] Max concurrency: %s\n' "${MAX_CONCURRENCY}"
printf '[Launcher] HF_DATASETS_OFFLINE=%s HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s\n' "${HF_DATASETS_OFFLINE}" "${HF_HUB_OFFLINE}" "${TRANSFORMERS_OFFLINE}"

echo "[Launcher] Stage 1: special benchmarks"
"${PYTHON_BIN}" -u /root/workspace/vq_lord/vq_lord3/eval_final/mm_special_benchmark_eval_teacher.py \
  --victim_model "${VICTIM_MODEL}" \
  --teacher_api_base "${TEACHER_API_BASE}" \
  --teacher_api_key "${TEACHER_API_KEY}" \
  --teacher_enable_thinking "${TEACHER_ENABLE_THINKING}" \
  --max_retries "${MAX_RETRIES}" \
  --sleep_sec "${SLEEP_SEC}" \
  --max_concurrency "${MAX_CONCURRENCY}" \
  --benchmarks ai2d,chartqa \
  --max_samples_per_benchmark 0 \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --prompt_style "${PROMPT_STYLE}" \
  --save_path "${RESULT_DIR}/mm_special_benchmarks_teacher_full.json"

echo "[Launcher] Stage 2: ScienceQA controls"
"${PYTHON_BIN}" -u /root/workspace/vq_lord/vq_lord3/eval_final/mm_scienceqa_control_eval_teacher.py \
  --scienceqa_path "${SCIENCEQA_PATH}" \
  --split test \
  --max_samples 0 \
  --controls baseline,text_only_blank,hint_ablation,option_shuffle,random_image_swap,image_blur,image_downsample \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --prompt_style "${PROMPT_STYLE}" \
  --victim_model "${VICTIM_MODEL}" \
  --teacher_api_base "${TEACHER_API_BASE}" \
  --teacher_api_key "${TEACHER_API_KEY}" \
  --teacher_enable_thinking "${TEACHER_ENABLE_THINKING}" \
  --max_retries "${MAX_RETRIES}" \
  --sleep_sec "${SLEEP_SEC}" \
  --max_concurrency "${MAX_CONCURRENCY}" \
  --save_path "${RESULT_DIR}/scienceqa_control_suite_teacher_full.json"

echo "[Launcher] Stage 3: report aggregation"
"${PYTHON_BIN}" -u /root/workspace/vq_lord/vq_lord3/eval_final/mm_eval_suite_report.py \
  --benchmark_result "${RESULT_DIR}/mm_special_benchmarks_teacher_full.json" \
  --control_result "${RESULT_DIR}/scienceqa_control_suite_teacher_full.json" \
  --save_json "${RESULT_DIR}/mm_eval_suite_report_teacher_full.json" \
  --save_md "${RESULT_DIR}/mm_eval_suite_report_teacher_full.md"

echo "[Launcher] All stages completed"
