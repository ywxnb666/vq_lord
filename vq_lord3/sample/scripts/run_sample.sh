#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python3}"
CONFIG_PATH="${1:-${SCRIPT_ROOT}/configs/example.openai.json}"

cd "${SCRIPT_ROOT}"
"${PYTHON_EXE}" -m sampler.cli sample --config "${CONFIG_PATH}"
