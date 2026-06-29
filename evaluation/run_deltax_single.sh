#!/usr/bin/env bash
# Single-image delta_x visualization for the 2-NFE gmkontext_uedit_fixedeps student.
#
# Runs one local image + one prompt and dumps into one folder:
#   0_src.png 1_edited.png 2_deltax_step1.png 3_deltax_step2.png 4_x0_step1.png
#   5_src_noised_step1.png 6_xt_after_step1.png 7_step2_start.png 8_v_step1.png 9_v_step2.png
#   noise.pt noise_vis.png tensors.pt prompt.txt
#
# Usage:
#   bash evaluation/run_deltax_single.sh
#   IMAGE=/path/to/img.png PROMPT="..." bash evaluation/run_deltax_single.sh
#   GPU=1 CKPT=/path/iter_xxxx.pth bash evaluation/run_deltax_single.sh

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"
CONDA_ROOT="${CONDA_ROOT:-${WORKSPACE_ROOT}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-${WORKSPACE_ROOT}/pretrained_models/FLUX.1-Kontext-dev}"
export PYTHONPATH="${EDITFLOW_DIR}:${PYTHONPATH:-}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"

CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data.py}"
CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k/20260618_055623/iter_20000.pth}"
IMAGE="${IMAGE:-${EDITFLOW_DIR}/pikachu.png}"
PROMPT="${PROMPT:-Put an Ash Ketchum-style Pokemon Trainer cap on Pikachu, with a red-and-white design and a green emblem on the front.}"
OUTPUT_DIR="${OUTPUT_DIR:-${EDITFLOW_DIR}/evaluation/imgedit_bench/outputs/deltax_vis/pikachu_single}"

NFE="${NFE:-2}"
GUIDANCE="${GUIDANCE:-3.5}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
# Background for transparent (RGBA) inputs: white / black / "r,g,b".
BG_COLOR="${BG_COLOR:-white}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/setup_env.sh" 2>/dev/null || source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 DIFFUSERS_OFFLINE=0

cd "${EDITFLOW_DIR}"

echo "[deltax-single] IMAGE=${IMAGE}"
echo "[deltax-single] PROMPT=${PROMPT}"
echo "[deltax-single] CKPT=${CKPT}"
echo "[deltax-single] OUTPUT_DIR=${OUTPUT_DIR} GPU=${GPU} NFE=${NFE} GUIDANCE=${GUIDANCE} SEED=${SEED}"

python evaluation/imgedit_bench/visualize_deltax_single.py \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --image "${IMAGE}" \
  --prompt "${PROMPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --nfe "${NFE}" \
  --guidance "${GUIDANCE}" \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --bg_color "${BG_COLOR}"

echo ""
echo "[deltax-single] Done. Outputs under: ${OUTPUT_DIR}"
