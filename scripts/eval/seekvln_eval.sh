#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${SEEKVLN_PYTHON:-python}
export PYTHONPATH="${ROOT}/runtime_compat:${ROOT}:${ROOT}/evaluation:${ROOT}/scripts/eval${PYTHONPATH:+:${PYTHONPATH}}"
export SEEKVLN_MINIMAL_VLNCE_IMPORT=1
export SEEKVLN_HABITAT_DISCRETE_ZERO_COMPAT=1
export SEEKVLN_TENSORBOARD_NO_TF=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export USE_TF=0 TRANSFORMERS_NO_TF=1 PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"
exec "${PYTHON_BIN}" "${ROOT}/scripts/eval/run_eval.py" "$@"
