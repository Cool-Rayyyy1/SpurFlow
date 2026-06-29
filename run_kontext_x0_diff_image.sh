#!/usr/bin/env bash
# 2nd GPU on this machine (physical GPU index 1): decoded-image x0_hat vs x_src and x_edit
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

CONDA_ROOT="/mnt/afs_zhangyunzhe/miniconda3"
ENV_NAME="arcflow"
PROJECT_DIR="/mnt/afs_zhangyunzhe/ArcFlow"
DATASET_ROOT="/mnt/afs_zhangyunzhe/dataset/pico-banana-400k"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${PROJECT_DIR}"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/env.sh" 2>/dev/null || true

if ! python -c "from diffusers import FluxKontextPipeline" 2>/dev/null; then
  echo "[setup] Fixing huggingface-hub for transformers/diffusers ..."
  pip install -q "huggingface-hub>=0.34.0,<1.0"
fi

MODEL_PATH="${KONTEXT_MODEL_PATH:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
MANIFEST="${MANIFEST:-${DATASET_ROOT}/manifests/kontext_x0_diff_10.jsonl}"
OUT_DIR="${PROJECT_DIR}/work_dirs/kontext_x0_diff"
mkdir -p "${OUT_DIR}"

EXTRA=()
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[warn] ${MANIFEST} missing — falling back to --demo."
  EXTRA+=(--demo)
fi

python tools/kontext_x0_ref_diff.py \
  --metric image \
  --model-path "${MODEL_PATH}" \
  --dataset-dir "${DATASET_ROOT}" \
  --manifest "${MANIFEST}" \
  --edited-images-dir "${DATASET_ROOT}/edited_images" \
  --num-samples 10 \
  --num-inference-steps 25 \
  --guidance-scale 2.5 \
  --seed 42 \
  --output-json "${OUT_DIR}/image_metrics.json" \
  "${EXTRA[@]}" \
  "$@" \
  2>&1 | tee "${OUT_DIR}/image_run.log"

echo "Done. Log: ${OUT_DIR}/image_run.log"
