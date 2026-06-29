#!/usr/bin/env bash
# Visualize per-step predicted delta_x of the 2-NFE gmkontext_uedit_fixedeps student.
#
# For each ImgEdit-Bench basic edit_type (action/add/adjust/background/compose/
# extract/remove/replace/style) it dumps, per example: src, edited, and the
# delta_x of each of the 2 sampling steps (+ prompts.txt) into one folder/category.
#
# Usage:
#   bash evaluation/run_deltax_vis.sh                     # full run, 5 examples/cat, GPUs 0,1
#   MAX_TASKS=1 GPUS=0 bash evaluation/run_deltax_vis.sh  # quick smoke test (1 example, 1 GPU)
#   NUM_PER_CAT=3 CKPT=/path/iter_6000.pth bash evaluation/run_deltax_vis.sh

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"
CONDA_ROOT="${CONDA_ROOT:-${WORKSPACE_ROOT}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-${WORKSPACE_ROOT}/pretrained_models/FLUX.1-Kontext-dev}"
export IMGEDIT_BENCH_ROOT="${IMGEDIT_BENCH_ROOT:-${WORKSPACE_ROOT}/dataset/imgedit/benchmark/Benchmark}"
export PYTHONPATH="${EDITFLOW_DIR}:${PYTHONPATH:-}"
# Match training-time ImageEdit(resize_mode='kontext') preprocessing.
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"

CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data.py}"
CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k/20260618_055623/iter_20000.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${EDITFLOW_DIR}/evaluation/imgedit_bench/outputs/deltax_vis/gmkontext_uedit_fixedeps_iter20000}"

NUM_PER_CAT="${NUM_PER_CAT:-5}"
MAX_TASKS="${MAX_TASKS:-0}"
NFE="${NFE:-2}"
GUIDANCE="${GUIDANCE:-3.5}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0,1}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/setup_env.sh" 2>/dev/null || source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 DIFFUSERS_OFFLINE=0

cd "${EDITFLOW_DIR}"

echo "[deltax-vis] CONFIG=${CONFIG}"
echo "[deltax-vis] CKPT=${CKPT}"
echo "[deltax-vis] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[deltax-vis] NUM_PER_CAT=${NUM_PER_CAT} MAX_TASKS=${MAX_TASKS} NFE=${NFE} GUIDANCE=${GUIDANCE} GPUS=${GPUS}"

python evaluation/imgedit_bench/visualize_deltax.py \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_per_cat "${NUM_PER_CAT}" \
  --max_tasks "${MAX_TASKS}" \
  --nfe "${NFE}" \
  --guidance "${GUIDANCE}" \
  --seed "${SEED}" \
  --gpus "${GPUS}"

echo ""
echo "[deltax-vis] Done. Outputs under: ${OUTPUT_DIR}"
