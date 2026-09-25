#!/usr/bin/env bash
# ImgEdit-Bench inference for the trained FLUX.2-klein-base Softsign-01 alpha
# 2-NFE student (split-stage DINO GAN recipe).
#
# Training counterpart:
#   train_flux2_klein_edit_fixedeps_alpha_softsign_alph_dino_gan.sh
#
# One pass writes edited images and two per-step alpha visualizations:
#   student/basic/{Action,Add,...}/{key}/
#     src.png  edit.png  pred.png  prompt.txt
#     step{1,2}_{alpha,heatmap,overlay}.png
#   student/uge/{key}.png
#
# Usage:
#   bash evaluation/run_flux2_klein_uedit_alpha_infer.sh
#   CKPT=/path/to/iter_XXXX.pth bash evaluation/run_flux2_klein_uedit_alpha_infer.sh
#   SUITE=basic MAX_SAMPLES=8 GEN_ONLY=1 \
#     bash evaluation/run_flux2_klein_uedit_alpha_infer.sh
#   SCORE_ONLY=1 RUN_TAG=<existing-run-tag> \
#     bash evaluation/run_flux2_klein_uedit_alpha_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="${EDITFLOW_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EVAL_DIR="${EVAL_DIR:-${SCRIPT_DIR}/imgedit_bench}"

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/afs_gaochengmin/.cache/huggingface}"
export PYTHONPATH="${EDITFLOW_DIR}:${PYTHONPATH:-}"
export IMGEDIT_BENCH_ROOT="${IMGEDIT_BENCH_ROOT:-/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark}"
export KLEIN_MODEL_PATH="${KLEIN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"

RUN_NAME="${RUN_NAME:-gmklein_base_uedit_fixedeps_alpha_softsign01_k16_2nfe_shift3.2_teachercfg4.0_pico400k_split_stage_alph_dino_gan}"
CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/flux2_klein/editflux2_klein_uedit_fixedeps_2nfe_k16_alpha_softsign_split_stage_alph_dino_gan.py}"
CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/20260810_020525/iter_6000.pth}"
RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT}" .pth)}"

STUDENT_NFE="${STUDENT_NFE:-2}"
STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-1.0}"
export STUDENT_GUIDANCE
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-flux2}"

SUITE="${SUITE:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GEN_ONLY="${GEN_ONLY:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
FORCE_SCORE="${FORCE_SCORE:-0}"

RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${EVAL_DIR}/outputs/runs/${RUN_TAG}}"
STUDENT_OUTPUT="${STUDENT_OUTPUT:-${RUN_OUTPUT_ROOT}/student}"
SCORES_SUBDIR="${SCORES_SUBDIR:-scores}"
SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores.txt}"
STUDENT_SCORES_DIR="${STUDENT_OUTPUT}/${SCORES_SUBDIR}"
STUDENT_SCORES_TXT="${STUDENT_OUTPUT}/${SCORES_TXT_NAME}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
RUN_CONFIG_TXT="${RUN_OUTPUT_ROOT}/run_config.txt"

write_run_config() {
  local status="${1:-started}"
  mkdir -p "${RUN_OUTPUT_ROOT}"
  {
    echo "ImgEdit-Bench FLUX.2 Klein Softsign-01 Alpha Student (+ alpha v6)"
    echo "================================================================"
    echo "Status:          ${status}"
    echo "RUN_NAME:        ${RUN_NAME}"
    echo "RUN_TAG:         ${RUN_TAG}"
    echo "CONFIG:          ${CONFIG}"
    echo "CKPT:            ${CKPT}"
    echo "KLEIN_MODEL_PATH:${KLEIN_MODEL_PATH}"
    echo "STUDENT_NFE:     ${STUDENT_NFE}"
    echo "STUDENT_GUIDANCE:${STUDENT_GUIDANCE}"
    echo "STUDENT_RESIZE_MODE:${STUDENT_RESIZE_MODE}"
    echo "SUITE:           ${SUITE}"
    echo "STUDENT_OUTPUT:  ${STUDENT_OUTPUT}"
    echo "LAYOUT:          student/basic/<Category>/<key>/{src,edit,pred,prompt,step{1,2}_*}"
    echo "ALPHA_ACTIVATION:softsign01"
    echo "GEN_ONLY:        ${GEN_ONLY}"
    echo "SCORE_ONLY:      ${SCORE_ONLY}"
    echo "NUM_PROCESSES:   ${NUM_PROCESSES}"
    echo "FORCE_SCORE:     ${FORCE_SCORE}"
    echo "OPENAI_SCORING_MODEL: ${OPENAI_SCORING_MODEL}"
    echo "STUDENT_SCORES_DIR: ${STUDENT_SCORES_DIR}"
    echo "STUDENT_SCORES_TXT: ${STUDENT_SCORES_TXT}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS:        ${NUM_GPUS}"
  } > "${RUN_CONFIG_TXT}"
  echo "[config] ${RUN_CONFIG_TXT}"
}

run_student_generation() {
  local -a cmd=(
    python "${EVAL_DIR}/run_editflow_imgedit_infer_alpha.py"
    --role student
    --suite "${SUITE}"
    --bench_root "${IMGEDIT_BENCH_ROOT}"
    --output_dir "${STUDENT_OUTPUT}"
    --model_path "${KLEIN_MODEL_PATH}"
    --run_name "${RUN_NAME}"
    --config "${CONFIG}"
    --ckpt "${CKPT}"
    --num_inference_steps "${STUDENT_NFE}"
    --guidance_scale "${STUDENT_GUIDANCE}"
    --skip_existing
    --num_gpus "${NUM_GPUS}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ "${CPU_OFFLOAD}" == "1" ]]; then
    cmd+=(--cpu_offload)
  fi
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

run_score() {
  local -a cmd=(
    python "${EVAL_DIR}/run_imgedit_score.py"
    --role student
    --model_output "${STUDENT_OUTPUT}"
    --scores_dir "${STUDENT_SCORES_DIR}"
    --scores_txt "${STUDENT_SCORES_TXT}"
    --bench_root "${IMGEDIT_BENCH_ROOT}"
    --annotations_dir "${EVAL_DIR}/annotations"
    --num_processes "${NUM_PROCESSES}"
  )
  if [[ "${FORCE_SCORE}" == "1" ]]; then
    cmd+=(--force)
  fi
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

mkdir -p "${HOME}/.cache/huggingface/hub" "${HOME}/.cache/torch/hub/checkpoints"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/setup_env.sh" 2>/dev/null || source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true

cd "${EDITFLOW_DIR}"

OPENAI_ENV="${EVAL_DIR}/openai.env"
if [[ -f "${OPENAI_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${OPENAI_ENV}"
fi

if [[ "${SCORE_ONLY}" != "1" ]]; then
  if [[ ! -d "${KLEIN_MODEL_PATH}" ]]; then
    echo "ERROR: FLUX.2 Klein model not found: ${KLEIN_MODEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found: ${CONFIG}" >&2
    exit 1
  fi
  if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    echo "Set CKPT=/path/to/iter_XXXX.pth" >&2
    exit 1
  fi
  bash "${EVAL_DIR}/download_imgedit_bench.sh"
fi

write_run_config "started"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  echo "[gen] FLUX.2 Klein Softsign-01 alpha student (${STUDENT_NFE} NFE, guidance=${STUDENT_GUIDANCE}, resize=${STUDENT_RESIZE_MODE})"
  run_student_generation
fi

if [[ "${GEN_ONLY}" != "1" ]]; then
  echo "[score] student -> ${STUDENT_SCORES_TXT} (model=${OPENAI_SCORING_MODEL}, processes=${NUM_PROCESSES})"
  run_score
fi

write_run_config "completed"

echo ""
echo "Done."
echo "  run folder:      ${RUN_OUTPUT_ROOT}"
echo "  student outputs: ${STUDENT_OUTPUT}"
echo "  basic cases:     ${STUDENT_OUTPUT}/basic/{Action,Add,...}/<key>/{src,edit,pred,prompt,step{1,2}_*}"
if [[ "${GEN_ONLY}" != "1" ]]; then
  echo "  student scores:  ${STUDENT_SCORES_TXT}"
fi
