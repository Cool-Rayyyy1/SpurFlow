#!/usr/bin/env bash
# GEdit-v2 official FLUX.2-klein-base-9B baseline, then GPT-4.1 VIEScore (Q-SC / Q-PQ / Q-O).
#
# Official FLUX.2-klein-base-9B settings (local model card):
#   num_inference_steps=50
#   guidance_scale=4.0
# This is the undistilled base teacher, not the 4-step distilled klein-9B.
#
# Usage:
#   bash evaluation/run_flux2_klein_base_gedit_eval.sh
#   GEN_ONLY=1 bash evaluation/run_flux2_klein_base_gedit_eval.sh
#   SCORE_ONLY=1 bash evaluation/run_flux2_klein_base_gedit_eval.sh
#   MAX_SAMPLES=8 GEN_ONLY=1 bash evaluation/run_flux2_klein_base_gedit_eval.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash evaluation/run_flux2_klein_base_gedit_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${EVAL_DIR:-${SCRIPT_DIR}/gedit_bench}"

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

export PYTHONPATH="${EDITFLOW_DIR}:${EVAL_DIR}:${SCRIPT_DIR}/imgedit_bench:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/env.sh"

export KLEIN_MODEL_PATH="${KLEIN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"
export KLEIN_STEPS="${KLEIN_STEPS:-50}"
export KLEIN_GUIDANCE="${KLEIN_GUIDANCE:-4.0}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4.1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
export NUM_GPUS

META_JSON="${META_JSON:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
RUN_NAME="${RUN_NAME:-flux2_klein_base_9b}"
RUN_TAG="${RUN_TAG:-${RUN_NAME}_${KLEIN_STEPS}step_cfg${KLEIN_GUIDANCE}_geditv2}"
SEED="${SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GEN_ONLY="${GEN_ONLY:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-32}"
FORCE_SCORE="${FORCE_SCORE:-0}"

RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${EVAL_DIR}/outputs/runs/${RUN_TAG}}"
MODEL_OUTPUT="${MODEL_OUTPUT:-${RUN_OUTPUT_ROOT}/model}"
RUN_CONFIG_TXT="${RUN_OUTPUT_ROOT}/run_config.txt"
OPENAI_ENV="${OPENAI_ENV:-${EVAL_DIR}/openai.env}"

cd "${EDITFLOW_DIR}"

if [[ -f "${OPENAI_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${OPENAI_ENV}"
fi
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4.1}"

if [[ ! -d "${KLEIN_MODEL_PATH}" ]]; then
  echo "ERROR: Klein-base model not found: ${KLEIN_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${META_JSON}" ]]; then
  echo "ERROR: GEdit meta not found: ${META_JSON}" >&2
  exit 1
fi

write_run_config() {
  local status="${1:-started}"
  mkdir -p "${RUN_OUTPUT_ROOT}"
  {
    echo "GEdit-v2 FLUX.2-klein-base-9B official baseline"
    echo "================================================"
    echo "Status:              ${status}"
    echo "RUN_TAG:             ${RUN_TAG}"
    echo "MODEL:               ${KLEIN_MODEL_PATH}"
    echo "META_JSON:           ${META_JSON}"
    echo "STEPS:               ${KLEIN_STEPS}   # official base 50, not distilled 4"
    echo "GUIDANCE_SCALE:      ${KLEIN_GUIDANCE}"
    echo "SEED:                ${SEED}"
    echo "MODEL_OUTPUT:        ${MODEL_OUTPUT}"
    echo "FLAT_DIR:            ${MODEL_OUTPUT}/GEdit_v2"
    echo "SCORE_MODEL:         ${OPENAI_SCORING_MODEL}"
    echo "METRICS:             Q-SC / Q-PQ / Q-O"
    echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS:            ${NUM_GPUS}"
    echo "MAX_SAMPLES:         ${MAX_SAMPLES:-all}"
  } > "${RUN_CONFIG_TXT}"
  echo "[config] ${RUN_CONFIG_TXT}"
}

write_run_config "started"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  python -c "from diffusers import Flux2KleinPipeline" || {
    echo "ERROR: Flux2KleinPipeline is unavailable in ${CONDA_ENV}." >&2
    echo "Need diffusers>=0.37." >&2
    exit 1
  }
  gen_cmd=(
    python "${EVAL_DIR}/run_editflow_gedit_infer.py"
    --role klein
    --meta_json "${META_JSON}"
    --output_dir "${MODEL_OUTPUT}"
    --model_path "${KLEIN_MODEL_PATH}"
    --run_name "${RUN_NAME}"
    --num_inference_steps "${KLEIN_STEPS}"
    --guidance_scale "${KLEIN_GUIDANCE}"
    --seed "${SEED}"
    --num_gpus "${NUM_GPUS}"
    --skip_existing
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    gen_cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ "${CPU_OFFLOAD}" == "1" ]]; then
    gen_cmd+=(--cpu_offload)
  fi
  echo "[gen] FLUX.2-klein-base-9B GEdit-v2 (${KLEIN_STEPS} steps, cfg=${KLEIN_GUIDANCE})"
  echo "Running: ${gen_cmd[*]}"
  "${gen_cmd[@]}"
fi

if [[ "${GEN_ONLY}" != "1" ]]; then
  score_cmd=(
    python "${EVAL_DIR}/run_gedit_gpt41_score.py"
    --meta_json "${META_JSON}"
    --pred_dir "${MODEL_OUTPUT}"
    --scores_dir "${RUN_OUTPUT_ROOT}/gpt41_eval"
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
echo "  images:     ${MODEL_OUTPUT}/GEdit_v2"
echo "  scores:     ${RUN_OUTPUT_ROOT}/gpt41_eval/scores.txt"
