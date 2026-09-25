#!/usr/bin/env bash
# GEdit-v2 for train_flux2_klein_data.sh (vanilla ArcFlow, no uedit/alpha/GAN).
#
# Must match train:
#   CONFIG  = configs/flux2_klein/editflux2_klein_2nfe_k16_data_oss_pico.py
#   student = true 2-NFE, guidance=1.0 (cond-only), STUDENT_RESIZE_MODE=flux2
#   teacher CFG=4 is train-only; do not pass 4.0 at student infer.
# Scoring: GPT-4.1 VIEScore Q-SC / Q-PQ / Q-O (same as other GEdit runs here).
# No alpha vis / case packs — only GEdit_v2/{stem}.jpg + scores.
#
# Default ckpt: iter_18000 of
#   gmklein_base_k16_2nfe_shift3.0_teachercfg4.0_oss30_pico70_data / 20260824_234013
#
# Usage:
#   bash evaluation/run_flux2_klein_data_gedit_eval.sh
#   CKPT=/path/to/iter_XXXX.pth bash evaluation/run_flux2_klein_data_gedit_eval.sh
#   MAX_SAMPLES=8 GEN_ONLY=1 bash evaluation/run_flux2_klein_data_gedit_eval.sh
#   SCORE_ONLY=1 bash evaluation/run_flux2_klein_data_gedit_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="${EDITFLOW_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
GEN_DIR="${SCRIPT_DIR}/gedit_v2"
SCORE_DIR="${SCRIPT_DIR}/gedit_bench"

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

export PYTHONPATH="${EDITFLOW_DIR}:${GEN_DIR}:${SCRIPT_DIR}/imgedit_bench:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/afs_gaochengmin/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export KLEIN_MODEL_PATH="${KLEIN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-flux2}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-1.0}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4.1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
export NUM_GPUS

export RUN_NAME="${RUN_NAME:-gmklein_base_k16_2nfe_shift3.0_teachercfg4.0_oss30_pico70_data}"
CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/flux2_klein/editflux2_klein_2nfe_k16_data_oss_pico.py}"
CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/20260824_234013/iter_18000.pth}"
RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT}" .pth)_gedit}"

META_JSON="${META_JSON:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GEN_ONLY="${GEN_ONLY:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
FORCE_SCORE="${FORCE_SCORE:-0}"

RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${GEN_DIR}/outputs/runs/${RUN_TAG}}"
STUDENT_OUTPUT="${STUDENT_OUTPUT:-${RUN_OUTPUT_ROOT}/student}"
SCORES_DIR="${SCORES_DIR:-${RUN_OUTPUT_ROOT}/gpt41_eval}"
RUN_CONFIG_TXT="${RUN_OUTPUT_ROOT}/run_config.txt"
OPENAI_ENV="${OPENAI_ENV:-${SCORE_DIR}/openai.env}"

mkdir -p "${HF_HOME}/hub" /mnt/afs_gaochengmin/.cache/torch/hub/checkpoints

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/setup_env.sh" 2>/dev/null || source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true
# env.sh must not clobber Klein student settings.
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-flux2}"
export KLEIN_MODEL_PATH="${KLEIN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"

cd "${EDITFLOW_DIR}"

if [[ -f "${OPENAI_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${OPENAI_ENV}"
fi
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4.1}"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found: ${CONFIG}" >&2
    exit 1
  fi
  if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
  fi
fi
if [[ ! -f "${META_JSON}" ]]; then
  echo "ERROR: GEdit meta not found: ${META_JSON}" >&2
  exit 1
fi

write_run_config() {
  local status="${1:-started}"
  mkdir -p "${RUN_OUTPUT_ROOT}"
  {
    echo "GEdit-v2 FLUX.2-klein-base vanilla ArcFlow student"
    echo "================================================="
    echo "Status:              ${status}"
    echo "RUN_NAME:            ${RUN_NAME}"
    echo "RUN_TAG:             ${RUN_TAG}"
    echo "CONFIG:              ${CONFIG}"
    echo "CKPT:                ${CKPT}"
    echo "KLEIN_MODEL_PATH:    ${KLEIN_MODEL_PATH}"
    echo "META_JSON:           ${META_JSON}"
    echo "STUDENT_NFE:         ${STUDENT_NFE}"
    echo "STUDENT_GUIDANCE:    ${STUDENT_GUIDANCE}"
    echo "STUDENT_RESIZE_MODE: ${STUDENT_RESIZE_MODE}"
    echo "REQUIRE_ALPHA:       0"
    echo "WRITE_CASE_BUNDLES:  0"
    echo "STUDENT_OUTPUT:      ${STUDENT_OUTPUT}"
    echo "FLAT_DIR:            ${STUDENT_OUTPUT}/GEdit_v2"
    echo "SCORES_DIR:          ${SCORES_DIR}"
    echo "SCORE_MODEL:         ${OPENAI_SCORING_MODEL}"
    echo "METRICS:             Q-SC / Q-PQ / Q-O"
    echo "GEN_ONLY:            ${GEN_ONLY}"
    echo "SCORE_ONLY:          ${SCORE_ONLY}"
    echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS:            ${NUM_GPUS}"
    echo "MAX_SAMPLES:         ${MAX_SAMPLES:-all}"
  } > "${RUN_CONFIG_TXT}"
  echo "[config] ${RUN_CONFIG_TXT}"
}

write_run_config "started"

echo "[gmklein vanilla-ArcFlow data / GEdit-v2]"
echo "  CONFIG=${CONFIG}"
echo "  CKPT=${CKPT}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  NFE=${STUDENT_NFE}  guidance=${STUDENT_GUIDANCE}  resize=${STUDENT_RESIZE_MODE}"
echo "  score=${OPENAI_SCORING_MODEL}  gpus=${NUM_GPUS}  cuda=${CUDA_VISIBLE_DEVICES}"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  gen_cmd=(
    python "${GEN_DIR}/run_editflow_gedit_infer.py"
    --meta_json "${META_JSON}"
    --output_dir "${STUDENT_OUTPUT}"
    --model_path "${KLEIN_MODEL_PATH}"
    --run_name "${RUN_NAME}"
    --config "${CONFIG}"
    --ckpt "${CKPT}"
    --num_inference_steps "${STUDENT_NFE}"
    --guidance_scale "${STUDENT_GUIDANCE}"
    --skip_existing
    --num_gpus "${NUM_GPUS}"
    --no_case_bundles
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    gen_cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ "${CPU_OFFLOAD}" == "1" ]]; then
    gen_cmd+=(--cpu_offload)
  fi
  echo "[gen] GEdit-v2 student (${STUDENT_NFE} NFE, cfg=${STUDENT_GUIDANCE}, resize=${STUDENT_RESIZE_MODE})"
  echo "Running: ${gen_cmd[*]}"
  "${gen_cmd[@]}"
fi

if [[ "${GEN_ONLY}" != "1" ]]; then
  score_cmd=(
    python "${SCORE_DIR}/run_gedit_gpt41_score.py"
    --meta_json "${META_JSON}"
    --pred_dir "${STUDENT_OUTPUT}"
    --scores_dir "${SCORES_DIR}"
    --model_name "${OPENAI_SCORING_MODEL}"
    --num_workers "${NUM_PROCESSES}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    score_cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ "${FORCE_SCORE}" == "1" ]]; then
    score_cmd+=(--force)
  fi
  echo "[score] GPT-4.1 VIEScore Q-SC / Q-PQ / Q-O"
  echo "Running: ${score_cmd[*]}"
  "${score_cmd[@]}"
fi

write_run_config "completed"

echo ""
echo "Done."
echo "  run folder: ${RUN_OUTPUT_ROOT}"
echo "  images:     ${STUDENT_OUTPUT}/GEdit_v2"
if [[ "${GEN_ONLY}" != "1" ]]; then
  echo "  scores:     ${SCORES_DIR}/scores.txt"
fi
