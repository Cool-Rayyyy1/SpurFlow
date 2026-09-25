#!/usr/bin/env bash
# Activate the training environment from this repository.
#   CONDA_ROOT=/path/to/anaconda3 source setup_env.sh
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-${HOME}/anaconda3}"
ENV_NAME="${ENV_NAME:-arcflow}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${PROJECT_DIR}"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/env.sh"
