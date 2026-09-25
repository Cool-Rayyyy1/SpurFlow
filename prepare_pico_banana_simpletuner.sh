#!/usr/bin/env bash
# Build SimpleTuner-compatible Kontext pairs from pico-banana-400k.
# Independent of EditFlow distillation train_*.sh scripts.
#
#   bash prepare_pico_banana_simpletuner.sh
#   MAX_SAMPLES=2000 bash prepare_pico_banana_simpletuner.sh
#   USE_SUMMARIZED=1 bash prepare_pico_banana_simpletuner.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_gaochengmin/data/pico-banana-400k}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/simpletuner_pairs}"
JSONL="${JSONL:-jsonl/sft_with_local_source_image_path.jsonl}"
VAL_SIZE="${VAL_SIZE:-128}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
USE_SUMMARIZED="${USE_SUMMARIZED:-0}"
FORCE="${FORCE:-0}"

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
ENV_NAME="${SIMPLETUNER_ENV_NAME:-simpletuner}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        conda activate "${ENV_NAME}"
    elif conda env list | awk '{print $1}' | grep -qx "arcflow"; then
        conda activate arcflow
    fi
fi

ARGS=(
    --data-root "${DATA_ROOT}"
    --jsonl "${JSONL}"
    --out-root "${OUT_ROOT}"
    --val-size "${VAL_SIZE}"
    --max-samples "${MAX_SAMPLES}"
)
if [[ "${USE_SUMMARIZED}" == "1" ]]; then
    ARGS+=(--use-summarized)
fi
if [[ "${FORCE}" == "1" ]]; then
    ARGS+=(--force)
fi

echo "[prepare] DATA_ROOT=${DATA_ROOT}"
echo "[prepare] OUT_ROOT=${OUT_ROOT}  MAX_SAMPLES=${MAX_SAMPLES}  VAL_SIZE=${VAL_SIZE}"
python "${PROJECT_DIR}/tools/prepare_pico_banana_simpletuner_pairs.py" "${ARGS[@]}"
