#!/usr/bin/env bash
# GEdit-v2 official Qwen-Image-Edit-2511 baseline, then GPT-4.1 VIEScore (Q-SC / Q-PQ / Q-O).
#
# Official Qwen-Image-Edit-2511 settings (model card / README):
#   num_inference_steps=40
#   true_cfg_scale=4.0
#   guidance_scale=1.0
#   negative_prompt=" "
#
# Usage:
#   bash evaluation/run_qwen_image_edit_2511_gedit_eval.sh
#   GEN_ONLY=1 bash evaluation/run_qwen_image_edit_2511_gedit_eval.sh
#   SCORE_ONLY=1 bash evaluation/run_qwen_image_edit_2511_gedit_eval.sh
#   MAX_SAMPLES=8 GEN_ONLY=1 bash evaluation/run_qwen_image_edit_2511_gedit_eval.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash evaluation/run_qwen_image_edit_2511_gedit_eval.sh

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

export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511}"
export QWEN_STEPS="${QWEN_STEPS:-40}"
export QWEN_TRUE_CFG_SCALE="${QWEN_TRUE_CFG_SCALE:-4.0}"
export QWEN_GUIDANCE="${QWEN_GUIDANCE:-1.0}"
export QWEN_NEGATIVE_PROMPT="${QWEN_NEGATIVE_PROMPT:- }"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4.1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
export NUM_GPUS

META_JSON="${META_JSON:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
RUN_NAME="${RUN_NAME:-qwen_image_edit_2511}"
RUN_TAG="${RUN_TAG:-${RUN_NAME}_${QWEN_STEPS}step_truecfg${QWEN_TRUE_CFG_SCALE}_geditv2}"
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

if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
  echo "ERROR: Qwen model not found: ${QWEN_MODEL_PATH}" >&2
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
    echo "GEdit-v2 Qwen-Image-Edit-2511 official baseline"
    echo "================================================"
    echo "Status:              ${status}"
    echo "RUN_TAG:             ${RUN_TAG}"
    echo "MODEL:               ${QWEN_MODEL_PATH}"
    echo "META_JSON:           ${META_JSON}"
    echo "STEPS:               ${QWEN_STEPS}   # official 40, not 50"
    echo "TRUE_CFG_SCALE:      ${QWEN_TRUE_CFG_SCALE}"
    echo "GUIDANCE_SCALE:      ${QWEN_GUIDANCE}"
    echo "NEGATIVE_PROMPT:     <single space>"
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
  python -c "from diffusers import QwenImageEditPlusPipeline" || {
    echo "ERROR: QwenImageEditPlusPipeline is unavailable in ${CONDA_ENV}." >&2
    exit 1
  }
  gen_cmd=(
    python "${EVAL_DIR}/run_editflow_gedit_infer.py"
    --role qwen
    --meta_json "${META_JSON}"
    --output_dir "${MODEL_OUTPUT}"
    --model_path "${QWEN_MODEL_PATH}"
    --run_name "${RUN_NAME}"
    --num_inference_steps "${QWEN_STEPS}"
    --guidance_scale "${QWEN_GUIDANCE}"
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
  echo "[gen] Qwen-Image-Edit-2511 GEdit-v2 (${QWEN_STEPS} steps, true_cfg=${QWEN_TRUE_CFG_SCALE})"
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
