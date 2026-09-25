#!/usr/bin/env bash
# EditFlow environment. Activate with:
#   source <project_dir>/setup_env.sh
set -euo pipefail

CONDA_ROOT="/mnt/afs_gaochengmin/anaconda3"
ENV_NAME="arcflow"
# Resolve the project dir from this file's own location (works after copies/renames).
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_ENDPOINT='https://hf-mirror.com'

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${PROJECT_DIR}"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/env.sh"
