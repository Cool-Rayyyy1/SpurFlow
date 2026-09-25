#!/usr/bin/env bash
# GEdit-v2 for official Qwen-Image-Edit-2511 Lightning 4-step LoRA, then
# GPT-4.1 VIEScore via evaluate_gedit_gpt41.sh (resize to 512-area before GPT).
#
# Generation matches ModelTC/Qwen-Image-Lightning generate_with_diffusers.py:
#   QwenImageEditPlusPipeline + Lightning LoRA
#   FlowMatchEulerDiscreteScheduler (exponential, shift=3)
#   steps=4, true_cfg_scale=1.0, negative_prompt=" ", seed=42
#
# Usage (1 GPU):
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
#     bash evaluation/run_qwen_image_edit_2511_lightning_4step_gedit_eval.sh
#
#   MAX_SAMPLES=8 GEN_ONLY=1 \
#     bash evaluation/run_qwen_image_edit_2511_lightning_4step_gedit_eval.sh
#   SCORE_ONLY=1 \
#     bash evaluation/run_qwen_image_edit_2511_lightning_4step_gedit_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="${EDITFLOW_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EVAL_DIR="${EVAL_DIR:-${SCRIPT_DIR}/gedit_bench}"

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/afs_gaochengmin/.cache/huggingface}"
export PYTHONPATH="${EDITFLOW_DIR}:${EVAL_DIR}:${SCRIPT_DIR}/imgedit_bench:${PYTHONPATH:-}"

export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511}"
export QWEN_LORA_PATH="${QWEN_LORA_PATH:-/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors}"
export QWEN_STEPS="${QWEN_STEPS:-4}"
export QWEN_TRUE_CFG_SCALE="${QWEN_TRUE_CFG_SCALE:-1.0}"
export QWEN_GUIDANCE="${QWEN_GUIDANCE:-1.0}"
export QWEN_NEGATIVE_PROMPT="${QWEN_NEGATIVE_PROMPT:- }"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

META_JSON="${META_JSON:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
RUN_NAME="${RUN_NAME:-qwen_image_edit_2511_lightning_4step}"
RUN_TAG="${RUN_TAG:-${RUN_NAME}_geditv2}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GEN_ONLY="${GEN_ONLY:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
WORKER_NUM="${WORKER_NUM:-32}"

RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${SCRIPT_DIR}/gedit_v2/outputs/runs/${RUN_TAG}}"
MODEL_OUTPUT="${MODEL_OUTPUT:-${RUN_OUTPUT_ROOT}/model}"
RESULT_FOLDER="${RESULT_FOLDER:-${MODEL_OUTPUT}/GEdit_v2}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_OUTPUT_ROOT}/gpt41_eval}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4.1}"
OPENAI_ENV="${OPENAI_ENV:-${SCRIPT_DIR}/imgedit_bench/openai.env}"
SCORE_SH="${SCORE_SH:-/mnt/afs_gaochengmin/A-backup/new/scripts/evaluate_gedit_gpt41.sh}"
RUN_CONFIG_TXT="${RUN_OUTPUT_ROOT}/run_config.txt"

write_run_config() {
  local status="${1:-started}"
  mkdir -p "${RUN_OUTPUT_ROOT}"
  {
    echo "GEdit-v2 Qwen-Image-Edit-2511 Lightning 4-step"
    echo "=============================================="
    echo "Status:              ${status}"
    echo "RUN_TAG:             ${RUN_TAG}"
    echo "BASE:                ${QWEN_MODEL_PATH}"
    echo "LORA:                ${QWEN_LORA_PATH}"
    echo "META_JSON:           ${META_JSON}"
    echo "STEPS:               ${QWEN_STEPS}"
    echo "TRUE_CFG_SCALE:      ${QWEN_TRUE_CFG_SCALE}"
    echo "GUIDANCE_SCALE:      ${QWEN_GUIDANCE}"
    echo "NEGATIVE_PROMPT:     <single space>"
    echo "SEED:                ${SEED}"
    echo "SCHEDULER:           FlowMatchEulerDiscreteScheduler exponential shift=3"
    echo "MODEL_OUTPUT:        ${MODEL_OUTPUT}"
    echo "RESULT_FOLDER:       ${RESULT_FOLDER}"
    echo "EVAL_OUTPUT_DIR:     ${EVAL_OUTPUT_DIR}"
    echo "SCORE_SH:            ${SCORE_SH}"
    echo "SCORE_RESIZE:        512-area (Step1X-Edit VIEScore)"
    echo "SCORE_MODEL:         ${OPENAI_SCORING_MODEL}"
    echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS:            ${NUM_GPUS}"
    echo "MAX_SAMPLES:         ${MAX_SAMPLES:-all}"
  } > "${RUN_CONFIG_TXT}"
  echo "[config] ${RUN_CONFIG_TXT}"
}

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/setup_env.sh" 2>/dev/null || source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true

cd "${EDITFLOW_DIR}"

if [[ -f "${OPENAI_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${OPENAI_ENV}"
fi
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4.1}"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
    echo "ERROR: Qwen model not found: ${QWEN_MODEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${QWEN_LORA_PATH}" ]]; then
    echo "ERROR: Lightning LoRA not found: ${QWEN_LORA_PATH}" >&2
    exit 1
  fi
fi
if [[ ! -f "${META_JSON}" ]]; then
  echo "ERROR: GEdit meta not found: ${META_JSON}" >&2
  exit 1
fi

write_run_config "started"

echo "[qwen lightning 4-step / GEdit-v2]"
echo "  BASE=${QWEN_MODEL_PATH}"
echo "  LORA=${QWEN_LORA_PATH}"
echo "  steps=${QWEN_STEPS}  true_cfg=${QWEN_TRUE_CFG_SCALE}  seed=${SEED}"
echo "  score=${OPENAI_SCORING_MODEL}  512-area  gpus=${NUM_GPUS}"

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
  echo "[gen] ${gen_cmd[*]}"
  "${gen_cmd[@]}"
fi

if [[ "${GEN_ONLY}" != "1" ]]; then
  if [[ ! -d "${RESULT_FOLDER}" ]]; then
    echo "ERROR: generated GEdit_v2 folder missing: ${RESULT_FOLDER}" >&2
    exit 1
  fi
  if [[ ! -f "${SCORE_SH}" ]]; then
    echo "ERROR: score script not found: ${SCORE_SH}" >&2
    exit 1
  fi
  echo "[score] 512-area GPT-4.1 via ${SCORE_SH}"
  RESULT_FOLDER="${RESULT_FOLDER}" \
  META_JSON="${META_JSON}" \
  EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
  WORKER_NUM="${WORKER_NUM}" \
  OPENAI_ENV="${OPENAI_ENV}" \
  OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL}" \
  bash "${SCORE_SH}"
fi

write_run_config "completed"

echo ""
echo "Done."
echo "  run folder: ${RUN_OUTPUT_ROOT}"
echo "  images:     ${RESULT_FOLDER}"
echo "  scores:     ${EVAL_OUTPUT_DIR}"
