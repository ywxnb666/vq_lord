#!/bin/bash
set -euo pipefail

SCRIPT_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_SOURCE_DIR}/common.sh"

ROOT_DIR="${ROOT_DIR:-$ROOT_DEFAULT}"
ROOT_DIR="${ROOT_DIR%/}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_DEFAULT}"
CKPT_DIR="${ROOT_DIR}/vq_lord_ckpts"
DATA_DIR="${ROOT_DIR}/vq_lord_data"
PREPROCESS_DIR="${DATA_DIR}/preprocess"
TEST_RESULT_DIR="${ROOT_DIR}/vq_lord_test_results"
LOG_DIR="${ROOT_DIR}/logs"

align_vq_setup_env
align_vq_ensure_runtime_dirs
align_vq_setup_logging "eval_base_qwen_iconqa"

PREPROCESS_ENTRY="${ROOT_DIR}/vq_lord3/data/preprocess/sciqa_preprocess.py"
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/evaluation/sciqa_process2_parallel.py"

DATASET_NAME="iconqa"
DATASET_PATH="${DATASET_PATH_DEFAULT_ICONQA}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
STUDENT_MODEL_TYPE="qwen2_vl"
MODEL_PATH="${QWEN2_VL_MODEL_DEFAULT}"
SCIENCEQA_SEED="${SCIENCEQA_SEED:-20240306}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_BUCKET_BATCH_SIZE="${EVAL_BUCKET_BATCH_SIZE:-4}"
PREPROCESS_SHUFFLE="${PREPROCESS_SHUFFLE:-1}"
EVAL_ANSWER_MODE="${EVAL_ANSWER_MODE:-generate_readable}"
USE_4BIT="${USE_4BIT:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
PREPROCESS_FORCE_REBUILD="${PREPROCESS_FORCE_REBUILD:-1}"

OUT_DIR="${OUT_DIR:-${TEST_RESULT_DIR}/base_qwen_iconqa_$(date -u +%Y%m%d_%H%M%S)}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUT_DIR}/summary.tsv}"
RUN_NAME="qwen2_vl_iconqa_${EVAL_SPLIT}_n${EVAL_MAX_SAMPLES}"
BUCKET_PLAN_PATH="${BUCKET_PLAN_PATH:-${PREPROCESS_DIR}/base_${RUN_NAME}_seed${SCIENCEQA_SEED}_patches_bs${EVAL_BUCKET_BATCH_SIZE}.json}"
SHARD_RESULT_DIR="${SHARD_RESULT_DIR:-${OUT_DIR}/${RUN_NAME}_shards}"
RESULT_PATH="${RESULT_PATH:-${OUT_DIR}/${RUN_NAME}.json}"

mkdir -p "${OUT_DIR}" "${SHARD_RESULT_DIR}" "${PREPROCESS_DIR}" "$(dirname "${RESULT_PATH}")"

echo -e "student\tdataset\tsplit\traw_acc\tmetric_acc\tformat_rate\tanswer_parse_rate\tcorrect\tn\tstatus\tresult_path" > "${SUMMARY_PATH}"

read -r -a GPU_ID_ARR <<< "${GPU_IDS}"
if [ "${#GPU_ID_ARR[@]}" -lt "${NUM_SHARDS}" ]; then
    echo "错误: GPU_IDS 数量不足，至少需要 ${NUM_SHARDS} 个 GPU id，当前只有 ${#GPU_ID_ARR[@]}"
    exit 1
fi

append_metrics() {
    local status="$1"
    if [ "${status}" != "OK" ] || [ ! -f "${RESULT_PATH}" ]; then
        echo -e "qwen2_vl\ticonqa\t${EVAL_SPLIT}\tNA\tNA\tNA\tNA\tNA\tNA\t${status}\t${RESULT_PATH}" | tee -a "${SUMMARY_PATH}"
        return
    fi
    "${PYTHON_BIN}" - "${RESULT_PATH}" "${EVAL_SPLIT}" <<'PY' | tee -a "${SUMMARY_PATH}"
import json
import sys
result_path, split = sys.argv[1:3]
with open(result_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
metrics = payload.get("metrics", {})
correct = int(metrics.get("correct", 0) or 0)
total = int(metrics.get("total", 0) or 0)
raw_acc = correct / total if total else 0.0
metric_acc = float(metrics.get("accuracy", 0.0) or 0.0)
format_rate = float(metrics.get("format_rate", 0.0) or 0.0)
answer_parse_rate = float(metrics.get("answer_parse_rate", 0.0) or 0.0)
print(
    f"qwen2_vl\ticonqa\t{split}\t{raw_acc:.6f}\t{metric_acc:.6f}\t"
    f"{format_rate:.6f}\t{answer_parse_rate:.6f}\t{correct}\t{total}\tOK\t{result_path}"
)
PY
}

render_progress_bar() {
    local done_samples="$1"
    local total_samples="$2"
    local running_shards="$3"
    local width=40
    local filled=0
    local percent=0
    if [ "${total_samples}" -gt 0 ]; then
        filled=$((done_samples * width / total_samples))
        percent=$((done_samples * 100 / total_samples))
    fi
    local bar=""
    for ((i=0; i<width; i++)); do
        if [ "${i}" -lt "${filled}" ]; then
            bar="${bar}#"
        else
            bar="${bar}-"
        fi
    done
    printf '\r[Eval Progress] [%s] %3d%% %s/%s samples, running_shards=%s/%s' \
        "${bar}" "${percent}" "${done_samples}" "${total_samples}" "${running_shards}" "${NUM_SHARDS}"
}

align_vq_require_file "${PREPROCESS_ENTRY}" "预处理分桶入口"
align_vq_require_file "${EVAL_ENTRY}" "并行评测入口"
align_vq_require_path "${MODEL_PATH}" "Qwen2-VL 模型路径"
align_vq_require_path "${DATASET_PATH}" "IconQA 数据集路径"

echo "======================================================"
echo "Base Qwen2-VL IconQA Eval"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "DATASET_PATH: ${DATASET_PATH}"
echo "EVAL_SPLIT: ${EVAL_SPLIT}"
echo "EVAL_MAX_SAMPLES: ${EVAL_MAX_SAMPLES}"
echo "BUCKET_PLAN_PATH: ${BUCKET_PLAN_PATH}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "LOG_FILE: ${LOG_FILE}"
echo "======================================================"

if [ ! -f "${BUCKET_PLAN_PATH}" ] || [ "${PREPROCESS_FORCE_REBUILD}" = "1" ]; then
    if ! "${PYTHON_BIN}" "${PREPROCESS_ENTRY}" \
        --dataset-name="${DATASET_NAME}" \
        --dataset-path="${DATASET_PATH}" \
        --model-path="${MODEL_PATH}" \
        --split="${EVAL_SPLIT}" \
        --train-num="${EVAL_MAX_SAMPLES}" \
        --seed="${SCIENCEQA_SEED}" \
        --bucket-by="patches" \
        --bucket-batch-size="${EVAL_BUCKET_BATCH_SIZE}" \
        --bucket-drop-last="0" \
        --shuffle="${PREPROCESS_SHUFFLE}" \
        --save-json="${BUCKET_PLAN_PATH}"
    then
        append_metrics "PREPROCESS_FAIL"
        exit 1
    fi
fi

if [ ! -f "${BUCKET_PLAN_PATH}" ]; then
    append_metrics "NO_BUCKET"
    exit 1
fi

TOTAL_EVAL_SAMPLES="$("${PYTHON_BIN}" -c 'import json, sys; p=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(sum(len(b.get("sample_ids", [])) for b in p.get("batch_plan", [])))' "${BUCKET_PLAN_PATH}")"

PIDS=()
SHARD_LOGS=()
FINISHED=()
FAILED=0
for (( shard_id=0; shard_id<NUM_SHARDS; shard_id++ )); do
    gpu_id="${GPU_ID_ARR[$shard_id]}"
    shard_result_path="${SHARD_RESULT_DIR}/shard_$(printf '%02d' "${shard_id}").json"
    shard_log_path="${OUT_DIR}/${RUN_NAME}_shard$(printf '%02d' "${shard_id}").log"
    SHARD_LOGS+=("${shard_log_path}")
    FINISHED[$shard_id]=0
    echo "[Launch] shard=${shard_id}/${NUM_SHARDS} gpu=${gpu_id}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        "${PYTHON_BIN}" "${EVAL_ENTRY}" \
            --model_path="${MODEL_PATH}" \
            --student_model_type="${STUDENT_MODEL_TYPE}" \
            --adapter_path="" \
            --dataset_name="${DATASET_NAME}" \
            --scienceqa_path="${DATASET_PATH}" \
            --split="${EVAL_SPLIT}" \
            --max_samples="${EVAL_MAX_SAMPLES}" \
            --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
            --enable_second_pass="0" \
            --second_pass_max_new_tokens="0" \
            --use_4bit="${USE_4BIT}" \
            --use_vq="0" \
            --vq_codebook_size="1024" \
            --freeze_vision_tower="0" \
            --vq_codebook_path="" \
            --answer_mode="${EVAL_ANSWER_MODE}" \
            --stage2_llava_remove_context="0" \
            --bucket_plan_path="${BUCKET_PLAN_PATH}" \
            --num_shards="${NUM_SHARDS}" \
            --shard_id="${shard_id}" \
            --save_path="${shard_result_path}"
    ) > "${shard_log_path}" 2>&1 &
    PIDS+=("$!")
done

all_finished=0
while [ "${all_finished}" -eq 0 ]; do
    running_shards=0
    for idx in "${!PIDS[@]}"; do
        if [ "${FINISHED[$idx]}" -eq 1 ]; then
            continue
        fi
        pid="${PIDS[$idx]}"
        if kill -0 "${pid}" 2>/dev/null; then
            running_shards=$((running_shards + 1))
        else
            if wait "${pid}"; then
                printf '\n[Done] shard=%s log=%s\n' "${idx}" "${SHARD_LOGS[$idx]}"
            else
                printf '\n[Error] shard=%s failed, log=%s\n' "${idx}" "${SHARD_LOGS[$idx]}"
                FAILED=1
            fi
            FINISHED[$idx]=1
        fi
    done
    done_samples="$({ grep -h -E '^\[Shard [0-9]+\] progress_done ' "${SHARD_LOGS[@]}" 2>/dev/null || true; } | wc -l | tr -d ' ')"
    render_progress_bar "${done_samples}" "${TOTAL_EVAL_SAMPLES}" "${running_shards}"
    all_finished=1
    for idx in "${!PIDS[@]}"; do
        if [ "${FINISHED[$idx]}" -eq 0 ]; then
            all_finished=0
        fi
    done
    if [ "${all_finished}" -eq 1 ]; then
        printf '\n'
    else
        sleep 5
    fi
done

if [ "${FAILED}" -ne 0 ]; then
    append_metrics "FAIL"
    exit 1
fi

"${PYTHON_BIN}" "${EVAL_ENTRY}" \
    --model_path="${MODEL_PATH}" \
    --student_model_type="${STUDENT_MODEL_TYPE}" \
    --adapter_path="" \
    --dataset_name="${DATASET_NAME}" \
    --scienceqa_path="${DATASET_PATH}" \
    --split="${EVAL_SPLIT}" \
    --max_samples="${EVAL_MAX_SAMPLES}" \
    --max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    --enable_second_pass="0" \
    --second_pass_max_new_tokens="0" \
    --use_4bit="${USE_4BIT}" \
    --use_vq="0" \
    --vq_codebook_size="1024" \
    --freeze_vision_tower="0" \
    --vq_codebook_path="" \
    --answer_mode="${EVAL_ANSWER_MODE}" \
    --stage2_llava_remove_context="0" \
    --bucket_plan_path="${BUCKET_PLAN_PATH}" \
    --num_shards="${NUM_SHARDS}" \
    --shard_result_dir="${SHARD_RESULT_DIR}" \
    --merge_only=1 \
    --save_path="${RESULT_PATH}"

append_metrics "OK"

echo
echo "summary: ${SUMMARY_PATH}"
column -t -s $'\t' "${SUMMARY_PATH}" || cat "${SUMMARY_PATH}"
