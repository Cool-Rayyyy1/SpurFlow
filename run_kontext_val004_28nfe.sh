#!/usr/bin/env bash
# Run FLUX Kontext teacher (28 NFE) on training val sample 004.
# Uses 1 GPU by default on this 2-GPU dev machine.
#
#   bash run_kontext_val004_28nfe.sh
#   GPU_ID=1 bash run_kontext_val004_28nfe.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/setup_env.sh"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
NUM_STEPS="${NUM_STEPS:-28}"
GUIDANCE="${GUIDANCE:-2.5}"
SEED="${SEED:-42}"

SAMPLE_DIR="${SAMPLE_DIR:-${PROJECT_DIR}/work_dirs/gmkontext_k16_2nfe_pico400k/20260610_011516/samples/iter_900/004}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/kontext_teacher_28nfe/val004}"

SRC="${SAMPLE_DIR}/src.png"
PROMPT_FILE="${SAMPLE_DIR}/prompt.txt"
OUT_IMAGE="${OUT_DIR}/kontext_28nfe.png"

if [[ ! -f "${SRC}" ]]; then
    echo "Missing source image: ${SRC}" >&2
    exit 1
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "Missing prompt file: ${PROMPT_FILE}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
cp -f "${SRC}" "${OUT_DIR}/src.png"
cp -f "${PROMPT_FILE}" "${OUT_DIR}/prompt.txt"
if [[ -f "${SAMPLE_DIR}/target.png" ]]; then
    cp -f "${SAMPLE_DIR}/target.png" "${OUT_DIR}/target.png"
fi

echo "Running FLUX Kontext teacher on GPU ${GPU_ID}"
echo "  sample: ${SAMPLE_DIR}"
echo "  steps:  ${NUM_STEPS}, guidance: ${GUIDANCE}"
echo "  output: ${OUT_IMAGE}"

python "${PROJECT_DIR}/tools/run_kontext_teacher_edit.py" \
    --model-path "${KONTEXT_MODEL}" \
    --source "${SRC}" \
    --prompt-file "${PROMPT_FILE}" \
    --output "${OUT_IMAGE}" \
    --num-inference-steps "${NUM_STEPS}" \
    --guidance-scale "${GUIDANCE}" \
    --seed "${SEED}" \
    --device "cuda:0"

echo "Done."
echo "  result: ${OUT_IMAGE}"
echo "  folder: ${OUT_DIR}"
