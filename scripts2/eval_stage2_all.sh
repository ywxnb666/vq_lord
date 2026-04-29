#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/songxinhao/workspace/vq_lord}"
TEST_SH="${ROOT_DIR}/scripts2/test_vq_lord_stage2_parallel.sh"
CKPT_DIR="${ROOT_DIR}/vq_lord_ckpts/stage2"
OUT_DIR="${ROOT_DIR}/vq_lord_test_results/stage2_batch_$(date -u +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}"
SUM="${OUT_DIR}/summary.tsv"
echo -e "ckpt\tacc\tformat_rate\tn\tstatus" > "${SUM}"

mapfile -t CKPTS < <(find "${CKPT_DIR}" -maxdepth 1 -type d -name 'stage2_vision_epoch*' | sort -V)
for ckpt in "${CKPTS[@]}"; do
  name="$(basename "${ckpt}")"
  runlog="${OUT_DIR}/${name}.log"
  if STAGE2_CKPT_PATH="${ckpt}" RESULT_PATH="${OUT_DIR}/${name}.json" SHARD_RESULT_DIR="${OUT_DIR}/${name}_shards" LOG_FILE="${runlog}" bash "${TEST_SH}" > "${runlog}" 2>&1; then
    acc="$(awk -F= '/^ACCURACY=/{v=$2} END{print v}' "${runlog}")"
    fmt="$(awk -F= '/^FORMAT_RATE=/{v=$2} END{print v}' "${runlog}")"
    n="$(awk -F= '/^N=/{v=$2} END{print v}' "${runlog}")"
    echo -e "${name}\t${acc:-NA}\t${fmt:-NA}\t${n:-NA}\tOK" | tee -a "${SUM}"
  else
    echo -e "${name}\tNA\tNA\tNA\tFAIL" | tee -a "${SUM}"
  fi
done

echo "summary: ${SUM}"
column -t -s $'\t' "${SUM}" || cat "${SUM}"
