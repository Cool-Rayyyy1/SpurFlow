#!/usr/bin/env bash
# GEdit-v2 for official TeleStyleV2 on Qwen-Image-Edit-2511, then
# GPT-4.1 VIEScore via evaluate_gedit_gpt41.sh (resize to 512-area before GPT).
#
# Generation matches HuggingFace demo witcherderivia/TeleStyleV2-QIE2511:
#   QwenImageEditPlusPipeline (local 2511)
#   fuse style LoRA + Lightning 4-step LoRA (scale=1.0)
#   steps=4, true_cfg_scale=1.0, guidance_scale=1.0, negative_prompt=" ", seed=123
#
# Usage (this 开发机, 1x H100):
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
#     bash evaluation/run_telestyle_qie2511_gedit_eval.sh

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
STYLE_LORA="${STYLE_LORA:-/mnt/afs_gaochengmin/checkpoints/TeleStyleV2/diffusers-TeleStyleV2-QIE-2511-Lora-bf16.safetensors}"
LIGHTNING_LORA="${LIGHTNING_LORA:-/mnt/afs_gaochengmin/checkpoints/TeleStyleV2/QIE-2511-Lightning-4steps-V1.0-bf16.safetensors}"
export QWEN_LORA_PATHS="${QWEN_LORA_PATHS:-${STYLE_LORA},${LIGHTNING_LORA}}"
export QWEN_LORA_ADAPTER_NAMES="${QWEN_LORA_ADAPTER_NAMES:-style,dmd}"
export QWEN_FUSE_LORA="${QWEN_FUSE_LORA:-1}"
export QWEN_LIGHTNING_SCHEDULER="${QWEN_LIGHTNING_SCHEDULER:-0}"
export QWEN_STEPS="${QWEN_STEPS:-4}"
export QWEN_TRUE_CFG_SCALE="${QWEN_TRUE_CFG_SCALE:-1.0}"
export QWEN_GUIDANCE="${QWEN_GUIDANCE:-1.0}"
export QWEN_NEGATIVE_PROMPT="${QWEN_NEGATIVE_PROMPT:- }"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

META_JSON="${META_JSON:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
RUN_NAME="${RUN_NAME:-telestyle_qie2511_official_4step}"
RUN_TAG="${RUN_TAG:-${RUN_NAME}_geditv2}"
SEED="${SEED:-123}"
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
    echo "GEdit-v2 TeleStyleV2 QIE-2511 official"
    echo "======================================"
    echo "Status:              ${status}"
    echo "RUN_TAG:             ${RUN_TAG}"
    echo "BASE:                ${QWEN_MODEL_PATH}"
    echo "LORA_PATHS:          ${QWEN_LORA_PATHS}"
    echo "ADAPTERS:            ${QWEN_LORA_ADAPTER_NAMES}"
    echo "FUSE_LORA:           ${QWEN_FUSE_LORA}"
    echo "LIGHTNING_SCHED:     ${QWEN_LIGHTNING_SCHEDULER}"
    echo "META_JSON:           ${META_JSON}"
    echo "STEPS:               ${QWEN_STEPS}"
    echo "TRUE_CFG_SCALE:      ${QWEN_TRUE_CFG_SCALE}"
    echo "GUIDANCE_SCALE:      ${QWEN_GUIDANCE}"
    echo "NEGATIVE_PROMPT:     <single space>"
    echo "SEED:                ${SEED}"
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

PYTHON="${PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found: ${PYTHON}" >&2
  exit 1
fi
export PATH="$(dirname "${PYTHON}"):${PATH}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true

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
  IFS=',' read -r -a _loras <<< "${QWEN_LORA_PATHS}"
  for _lora in "${_loras[@]}"; do
    _lora="${_lora#"${_lora%%[![:space:]]*}"}"
    _lora="${_lora%"${_lora##*[![:space:]]}"}"
    if [[ ! -e "${_lora}" ]]; then
      echo "ERROR: LoRA not found: ${_lora}" >&2
      exit 1
    fi
  done
fi
if [[ ! -f "${META_JSON}" ]]; then
  echo "ERROR: GEdit meta not found: ${META_JSON}" >&2
  exit 1
fi

write_run_config "started"

echo "[TeleStyleV2 QIE-2511 / GEdit-v2]"
echo "  BASE=${QWEN_MODEL_PATH}"
echo "  LORAS=${QWEN_LORA_PATHS}"
echo "  steps=${QWEN_STEPS}  true_cfg=${QWEN_TRUE_CFG_SCALE}  seed=${SEED}  fuse=${QWEN_FUSE_LORA}"
echo "  score=${OPENAI_SCORING_MODEL}  512-area  gpus=${NUM_GPUS}"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  "${PYTHON}" -c "from diffusers import QwenImageEditPlusPipeline" || {
    echo "ERROR: QwenImageEditPlusPipeline is unavailable in ${PYTHON}." >&2
    exit 1
  }
  gen_cmd=(
    "${PYTHON}" "${EVAL_DIR}/run_editflow_gedit_infer.py"
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
