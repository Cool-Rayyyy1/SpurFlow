#!/usr/bin/env bash
# Smoke test GPT scoring API — no GPU, single Basic sample.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/SpurFlow}"
EVAL_DIR="${PROJECT_DIR}/evaluation/imgedit_bench"
CONDA_ROOT="${CONDA_ROOT:-/path/to/miniconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python -c "import openai" 2>/dev/null || pip install -q openai tenacity

if [[ -f "${EVAL_DIR}/openai.env" ]]; then
  # shellcheck source=/dev/null
  source "${EVAL_DIR}/openai.env"
fi

python "${EVAL_DIR}/smoke_test_score_api.py" "$@"
