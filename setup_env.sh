#!/usr/bin/env bash
# EditFlow environment. Activate with:
#   source /mnt/afs_zhangyunzhe/EditFlow/setup_env.sh
set -euo pipefail

CONDA_ROOT="/mnt/afs_zhangyunzhe/miniconda3"
ENV_NAME="arcflow"
PROJECT_DIR="/mnt/afs_zhangyunzhe/EditFlow"

export HF_ENDPOINT='https://hf-mirror.com'

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${PROJECT_DIR}"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/env.sh"
