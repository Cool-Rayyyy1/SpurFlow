#!/usr/bin/env bash
# Single-image FLUX.1-Kontext (teacher) edit inference.
#
# Default: pikachu.png + Ash cap prompt, 2 inference steps, guidance 2.5.
# Output folder:
#   evaluation/imgedit_bench/outputs/teacher_single/pikachu_2step/
#     0_src.png 1_edited.png 2_x0_step1.png 3_x0_step2.png prompt.txt
# 0_src.png keeps RGBA transparency when the input has alpha (no white composite).
#
# Usage:
#   bash evaluation/run_teacher_single.sh
#   STEPS=28 GUIDANCE=2.5 bash evaluation/run_teacher_single.sh
#   IMAGE=/path/to.png PROMPT="..." OUTPUT_DIR=/path/out bash evaluation/run_teacher_single.sh

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

MODEL_PATH="${MODEL_PATH:-${KONTEXT_MODEL_PATH}}"
IMAGE="${IMAGE:-${EDITFLOW_DIR}/pikachu.png}"
PROMPT="${PROMPT:-Put an Ash Ketchum-style Pokemon Trainer cap on Pikachu, with a red-and-white design and a green emblem on the front.}"
OUTPUT_DIR="${OUTPUT_DIR:-${EDITFLOW_DIR}/evaluation/imgedit_bench/outputs/teacher_single/pikachu_2step}"

STEPS="${STEPS:-2}"
GUIDANCE="${GUIDANCE:-2.5}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/setup_env.sh" 2>/dev/null || source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 DIFFUSERS_OFFLINE=0

cd "${EDITFLOW_DIR}"

echo "[teacher-single] MODEL_PATH=${MODEL_PATH}"
echo "[teacher-single] IMAGE=${IMAGE}"
echo "[teacher-single] PROMPT=${PROMPT}"
echo "[teacher-single] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[teacher-single] STEPS=${STEPS} GUIDANCE=${GUIDANCE} SEED=${SEED} GPU=${GPU}"

cmd=(
  python evaluation/imgedit_bench/run_teacher_single.py
  --model_path "${MODEL_PATH}"
  --image "${IMAGE}"
  --prompt "${PROMPT}"
  --output_dir "${OUTPUT_DIR}"
  --num_inference_steps "${STEPS}"
  --guidance_scale "${GUIDANCE}"
  --seed "${SEED}"
  --gpu "${GPU}"
)
if [[ "${CPU_OFFLOAD}" == "1" ]]; then
  cmd+=(--cpu_offload)
fi

"${cmd[@]}"

echo ""
echo "[teacher-single] Done. Outputs under: ${OUTPUT_DIR}"
