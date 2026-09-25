#!/usr/bin/env bash
# Build SimpleTuner-compatible Kontext pairs from oss_edit.
#
#   bash prepare_oss_edit_simpletuner.sh
#   MAX_SAMPLES=2000 bash prepare_oss_edit_simpletuner.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSS_ROOT="${OSS_ROOT:-/mnt/afs_gaochengmin/data/oss_edit}"
OUT_ROOT="${OUT_ROOT:-${OSS_ROOT}/simpletuner_pairs}"
JSONL="${JSONL:-${OSS_ROOT}/metadata.jsonl}"
VAL_SIZE="${VAL_SIZE:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
FORCE="${FORCE:-0}"
EDITFLOW_OSS_BUILD="${EDITFLOW_OSS_BUILD:-${PROJECT_DIR}/tools/build_oss_edit_jsonl.py}"

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

if [[ ! -f "${JSONL}" ]]; then
    echo "[data] building ${JSONL}"
    python "${EDITFLOW_OSS_BUILD}" --root "${OSS_ROOT}" --out "${JSONL}"
fi

ARGS=(
    --data-root "${OSS_ROOT}"
    --jsonl "${JSONL}"
    --out-root "${OUT_ROOT}"
    --source-column "input_path,local_input_image"
    --target-column "output_path,output_image"
    --prompt-column "instruction,Edit_Instruction,gemma_instruction_separate,text"
    --val-size "${VAL_SIZE}"
    --max-samples "${MAX_SAMPLES}"
)
if [[ "${FORCE}" == "1" ]]; then
    ARGS+=(--force)
fi

echo "[prepare-oss] OSS_ROOT=${OSS_ROOT}"
echo "[prepare-oss] OUT_ROOT=${OUT_ROOT}  MAX_SAMPLES=${MAX_SAMPLES}  VAL_SIZE=${VAL_SIZE}"
python "${PROJECT_DIR}/tools/prepare_pico_banana_simpletuner_pairs.py" "${ARGS[@]}"
