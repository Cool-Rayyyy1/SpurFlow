#!/usr/bin/env bash
# Random pico-banana images + FLUX Kontext with prompt "keep the same" (28 NFE).
# Writes velocity_report.txt: per-step velocity u (= noise_pred) stats.
#
#   bash run_kontext_keep_same_velocity.sh
#   GPU_ID=0 NUM_SAMPLES=3 bash run_kontext_keep_same_velocity.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/setup_env.sh"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
DATASET_DIR="${DATASET_DIR:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
MANIFEST="${MANIFEST:-${DATASET_DIR}/jsonl/sft_with_local_source_image_path.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
NUM_STEPS="${NUM_STEPS:-28}"
GUIDANCE="${GUIDANCE:-2.5}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-keep the same}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/kontext_keep_same_velocity}"

mkdir -p "${OUT_DIR}"

echo "FLUX Kontext velocity check on GPU ${GPU_ID}"
echo "  prompt:  ${PROMPT}"
echo "  steps:   ${NUM_STEPS}"
echo "  samples: ${NUM_SAMPLES}"
echo "  output:  ${OUT_DIR}"

python "${PROJECT_DIR}/tools/kontext_keep_same_velocity_check.py" \
    --model-path "${KONTEXT_MODEL}" \
    --dataset-dir "${DATASET_DIR}" \
    --manifest "${MANIFEST}" \
    --num-samples "${NUM_SAMPLES}" \
    --num-inference-steps "${NUM_STEPS}" \
    --guidance-scale "${GUIDANCE}" \
    --seed "${SEED}" \
    --prompt "${PROMPT}" \
    --out-dir "${OUT_DIR}" \
    --report-txt "${OUT_DIR}/velocity_report.txt" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo "Done."
echo "  report: ${OUT_DIR}/velocity_report.txt"
echo "  log:    ${OUT_DIR}/run.log"
