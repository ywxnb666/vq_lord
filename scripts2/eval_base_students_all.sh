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
align_vq_setup_logging "eval_base_students_all"

PREPROCESS_ENTRY="${ROOT_DIR}/vq_lord3/data/preprocess/sciqa_preprocess.py"
EVAL_ENTRY="${ROOT_DIR}/vq_lord3/evaluation/sciqa_process2_parallel.py"

SCIENCEQA_SEED="${SCIENCEQA_SEED:-20240306}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_BUCKET_BATCH_SIZE="${EVAL_BUCKET_BATCH_SIZE:-4}"
PREPROCESS_SHUFFLE="${PREPROCESS_SHUFFLE:-1}"
EVAL_ANSWER_MODE="${EVAL_ANSWER_MODE:-generate_readable}"
USE_4BIT="${USE_4BIT:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"

OUT_DIR="${OUT_DIR:-${TEST_RESULT_DIR}/base_students_all_$(date -u +%Y%m%d_%H%M%S)}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUT_DIR}/summary.tsv}"

mkdir -p "${OUT_DIR}" "${PREPROCESS_DIR}" "${LOG_DIR}"

align_vq_require_file "${PREPROCESS_ENTRY}" "预处理分桶入口"
align_vq_require_file "${EVAL_ENTRY}" "并行评测入口"

read -r -a GPU_ID_ARR <<< "${GPU_IDS}"
if [ "${#GPU_ID_ARR[@]}" -lt "${NUM_SHARDS}" ]; then
    echo "错误: GPU_IDS 数量不足，至少需要 ${NUM_SHARDS} 个 GPU id，当前只有 ${#GPU_ID_ARR[@]}"
    exit 1
fi

echo -e "student\tdataset\tsplit\traw_acc\tmetric_acc\tformat_rate\tanswer_parse_rate\tcorrect\tn\tstatus\tresult_path" > "${SUMMARY_PATH}"

dataset_path_for() {
    case "$1" in
        scienceqa) echo "${DATASET_PATH_DEFAULT_SCIENCEQA}" ;;
        aokvqa) echo "${DATASET_PATH_DEFAULT_AOKVQA}" ;;
        iconqa) echo "${DATASET_PATH_DEFAULT_ICONQA}" ;;
        *) echo "unknown dataset: $1" >&2; return 1 ;;
    esac
}

split_for() {
    case "$1" in
        scienceqa) echo "test" ;;
        aokvqa) echo "validation" ;;
        iconqa) echo "test" ;;
        *) echo "unknown dataset: $1" >&2; return 1 ;;
    esac
}

model_path_for() {
    case "$1" in
        qwen2_vl) echo "${QWEN2_VL_MODEL_DEFAULT}" ;;
        llava_next) echo "${MODEL_DEFAULT}" ;;
        *) echo "unknown student: $1" >&2; return 1 ;;
    esac
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

append_metrics() {
    local student="$1"
    local dataset="$2"
    local split="$3"
    local result_path="$4"
    local status="$5"

    if [ "${status}" != "OK" ] || [ ! -f "${result_path}" ]; then
        echo -e "${student}\t${dataset}\t${split}\tNA\tNA\tNA\tNA\tNA\tNA\t${status}\t${result_path}" | tee -a "${SUMMARY_PATH}"
        return
    fi

    "${PYTHON_BIN}" - "${student}" "${dataset}" "${split}" "${status}" "${result_path}" <<'PY' | tee -a "${SUMMARY_PATH}"
import json
import sys

student, dataset, split, status, result_path = sys.argv[1:6]
with open(result_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
metrics = payload.get("metrics", {})
correct = int(metrics.get("correct", 0) or 0)
total = int(metrics.get("total", 0) or 0)
raw_acc = correct / total if total else 0.0
metric_acc = float(metrics.get("accuracy", 0.0) or 0.0)
format_rate = metrics.get("format_rate", "NA")
answer_parse_rate = metrics.get("answer_parse_rate", "NA")
if isinstance(format_rate, float):
    format_rate = f"{format_rate:.6f}"
if isinstance(answer_parse_rate, float):
    answer_parse_rate = f"{answer_parse_rate:.6f}"
print(
    f"{student}\t{dataset}\t{split}\t{raw_acc:.6f}\t{metric_acc:.6f}\t"
    f"{format_rate}\t{answer_parse_rate}\t{correct}\t{total}\t{status}\t{result_path}"
)
PY
}

run_one() {
    local student="$1"
    local dataset="$2"
    local split
    local dataset_path
    local model_path
    local stage2_llava_remove_context="0"

    split="$(split_for "${dataset}")"
    dataset_path="$(dataset_path_for "${dataset}")"
    model_path="$(model_path_for "${student}")"
    if [ "${student}" = "llava_next" ]; then
        stage2_llava_remove_context="${STAGE2_LLAVA_REMOVE_CONTEXT:-1}"
    fi

    local run_name="${student}_${dataset}_${split}_n${EVAL_MAX_SAMPLES}"
    local bucket_plan_path="${PREPROCESS_DIR}/base_${run_name}_seed${SCIENCEQA_SEED}_patches_bs${EVAL_BUCKET_BATCH_SIZE}.json"
    local shard_result_dir="${OUT_DIR}/${run_name}_shards"
    local result_path="${OUT_DIR}/${run_name}.json"

    mkdir -p "${shard_result_dir}"

    echo
    echo "======================================================"
    echo "Base student eval: student=${student}, dataset=${dataset}, split=${split}"
    echo "MODEL_PATH: ${model_path}"
    echo "DATASET_PATH: ${dataset_path}"
    echo "BUCKET_PLAN_PATH: ${bucket_plan_path}"
    echo "RESULT_PATH: ${result_path}"
    echo "======================================================"

    if [ ! -f "${bucket_plan_path}" ] || [ "${PREPROCESS_FORCE_REBUILD:-0}" = "1" ]; then
        if ! "${PYTHON_BIN}" "${PREPROCESS_ENTRY}" \
            --dataset-name="${dataset}" \
            --dataset-path="${dataset_path}" \
            --model-path="${model_path}" \
            --split="${split}" \
            --train-num="${EVAL_MAX_SAMPLES}" \
            --seed="${SCIENCEQA_SEED}" \
            --bucket-by="patches" \
            --bucket-batch-size="${EVAL_BUCKET_BATCH_SIZE}" \
            --bucket-drop-last="0" \
            --shuffle="${PREPROCESS_SHUFFLE}" \
            --save-json="${bucket_plan_path}"
        then
            echo "[Error] 预处理失败: ${run_name}"
            append_metrics "${student}" "${dataset}" "${split}" "${result_path}" "PREPROCESS_FAIL"
            return 1
        fi
    fi

    if [ ! -f "${bucket_plan_path}" ]; then
        echo "[Error] 分桶文件不存在: ${bucket_plan_path}"
        append_metrics "${student}" "${dataset}" "${split}" "${result_path}" "NO_BUCKET"
        return 1
    fi

    local total_eval_samples
    total_eval_samples="$("${PYTHON_BIN}" -c 'import json, sys; p=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(sum(len(b.get("sample_ids", [])) for b in p.get("batch_plan", [])))' "${bucket_plan_path}")"

    local pids=()
    local shard_logs=()
    local failed=0
    local finished=()

    for (( shard_id=0; shard_id<NUM_SHARDS; shard_id++ )); do
        local gpu_id="${GPU_ID_ARR[$shard_id]}"
        local shard_result_path="${shard_result_dir}/shard_$(printf '%02d' "${shard_id}").json"
        local shard_log_path="${OUT_DIR}/${run_name}_shard$(printf '%02d' "${shard_id}").log"
        shard_logs+=("${shard_log_path}")
        finished[$shard_id]=0

        echo "[Launch] ${run_name} shard=${shard_id}/${NUM_SHARDS} gpu=${gpu_id}"
        (
            export CUDA_VISIBLE_DEVICES="${gpu_id}"
            "${PYTHON_BIN}" "${EVAL_ENTRY}" \
                --model_path="${model_path}" \
                --student_model_type="${student}" \
                --adapter_path="" \
                --dataset_name="${dataset}" \
                --scienceqa_path="${dataset_path}" \
                --split="${split}" \
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
                --stage2_llava_remove_context="${stage2_llava_remove_context}" \
                --bucket_plan_path="${bucket_plan_path}" \
                --num_shards="${NUM_SHARDS}" \
                --shard_id="${shard_id}" \
                --save_path="${shard_result_path}"
        ) > "${shard_log_path}" 2>&1 &
        pids+=("$!")
    done

    local all_finished=0
    while [ "${all_finished}" -eq 0 ]; do
        local running_shards=0
        for idx in "${!pids[@]}"; do
            if [ "${finished[$idx]}" -eq 1 ]; then
                continue
            fi
            local pid="${pids[$idx]}"
            if kill -0 "${pid}" 2>/dev/null; then
                running_shards=$((running_shards + 1))
            else
                if wait "${pid}"; then
                    printf '\n[Done] %s shard=%s log=%s\n' "${run_name}" "${idx}" "${shard_logs[$idx]}"
                else
                    printf '\n[Error] %s shard=%s failed, log=%s\n' "${run_name}" "${idx}" "${shard_logs[$idx]}"
                    failed=1
                fi
                finished[$idx]=1
            fi
        done

        local done_samples
        done_samples="$({ grep -h -E '^\[Shard [0-9]+\] progress_done ' "${shard_logs[@]}" 2>/dev/null || true; } | wc -l | tr -d ' ')"
        render_progress_bar "${done_samples}" "${total_eval_samples}" "${running_shards}"

        all_finished=1
        for idx in "${!pids[@]}"; do
            if [ "${finished[$idx]}" -eq 0 ]; then
                all_finished=0
            fi
        done
        if [ "${all_finished}" -eq 1 ]; then
            printf '\n'
        else
            sleep 5
        fi
    done

    if [ "${failed}" -ne 0 ]; then
        append_metrics "${student}" "${dataset}" "${split}" "${result_path}" "FAIL"
        return 1
    fi

    "${PYTHON_BIN}" "${EVAL_ENTRY}" \
        --model_path="${model_path}" \
        --student_model_type="${student}" \
        --adapter_path="" \
        --dataset_name="${dataset}" \
        --scienceqa_path="${dataset_path}" \
        --split="${split}" \
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
        --stage2_llava_remove_context="${stage2_llava_remove_context}" \
        --bucket_plan_path="${bucket_plan_path}" \
        --num_shards="${NUM_SHARDS}" \
        --shard_result_dir="${shard_result_dir}" \
        --merge_only=1 \
        --save_path="${result_path}"

    append_metrics "${student}" "${dataset}" "${split}" "${result_path}" "OK"
}

FAILED_ANY=0
for student in qwen2_vl llava_next; do
    for dataset in scienceqa aokvqa iconqa; do
        if ! run_one "${student}" "${dataset}"; then
            FAILED_ANY=1
        fi
    done
done

echo
echo "summary: ${SUMMARY_PATH}"
column -t -s $'\t' "${SUMMARY_PATH}" || cat "${SUMMARY_PATH}"

if [ "${FAILED_ANY}" -ne 0 ]; then
    exit 1
fi
