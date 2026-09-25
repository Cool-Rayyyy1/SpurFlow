#!/usr/bin/env bash
# ImgEdit-Bench for official TeleStyleV2 on Qwen-Image-Edit-2511.
#
# Matches HuggingFace demo witcherderivia/TeleStyleV2-QIE2511:
#   QwenImageEditPlusPipeline (local 2511)
#   fuse style LoRA + Lightning 4-step LoRA (scale=1.0)
#   steps=4, true_cfg_scale=1.0, guidance_scale=1.0, negative_prompt=" ", seed=123
# Then GPT-4.1 scoring.
#
# Usage (this 开发机, 1x H100):
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
#     bash evaluation/run_telestyle_qie2511_imgedit_eval.sh

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
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511}"
STYLE_LORA="${STYLE_LORA:-/mnt/afs_gaochengmin/checkpoints/TeleStyleV2/diffusers-TeleStyleV2-QIE-2511-Lora-bf16.safetensors}"
LIGHTNING_LORA="${LIGHTNING_LORA:-/mnt/afs_gaochengmin/checkpoints/TeleStyleV2/QIE-2511-Lightning-4steps-V1.0-bf16.safetensors}"
export QWEN_LORA_PATHS="${QWEN_LORA_PATHS:-${STYLE_LORA},${LIGHTNING_LORA}}"
export QWEN_LORA_ADAPTER_NAMES="${QWEN_LORA_ADAPTER_NAMES:-style,dmd}"
export QWEN_FUSE_LORA="${QWEN_FUSE_LORA:-1}"
export QWEN_LIGHTNING_SCHEDULER="${QWEN_LIGHTNING_SCHEDULER:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

export QWEN_STEPS="${QWEN_STEPS:-4}"
export QWEN_TRUE_CFG_SCALE="${QWEN_TRUE_CFG_SCALE:-1.0}"
export QWEN_GUIDANCE="${QWEN_GUIDANCE:-1.0}"
export QWEN_NEGATIVE_PROMPT="${QWEN_NEGATIVE_PROMPT:- }"
QWEN_SEED="${QWEN_SEED:-123}"

export RUN_NAME="${RUN_NAME:-telestyle_qie2511_official_4step}"
export RUN_TAG="${RUN_TAG:-${RUN_NAME}}"

SUITE="${SUITE:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GEN_ONLY="${GEN_ONLY:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
FORCE_SCORE="${FORCE_SCORE:-0}"

RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${EVAL_DIR}/outputs/runs/${RUN_TAG}}"
MODEL_OUTPUT="${MODEL_OUTPUT:-${RUN_OUTPUT_ROOT}/model}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4.1}"
RUN_CONFIG_TXT="${RUN_OUTPUT_ROOT}/run_config.txt"

write_run_config() {
  local status="${1:-started}"
  mkdir -p "${RUN_OUTPUT_ROOT}"
  {
    echo "ImgEdit-Bench TeleStyleV2 QIE-2511 official"
    echo "==========================================="
    echo "Status:          ${status}"
    echo "RUN_TAG:         ${RUN_TAG}"
    echo "BASE:            ${QWEN_MODEL_PATH}"
    echo "LORA_PATHS:      ${QWEN_LORA_PATHS}"
    echo "ADAPTERS:        ${QWEN_LORA_ADAPTER_NAMES}"
    echo "FUSE_LORA:       ${QWEN_FUSE_LORA}"
    echo "LIGHTNING_SCHED: ${QWEN_LIGHTNING_SCHEDULER}"
    echo "STEPS:           ${QWEN_STEPS}"
    echo "TRUE_CFG_SCALE:  ${QWEN_TRUE_CFG_SCALE}"
    echo "GUIDANCE:        ${QWEN_GUIDANCE}"
    echo "NEGATIVE_PROMPT: <single space>"
    echo "SEED:            ${QWEN_SEED}"
    echo "SUITE:           ${SUITE}"
    echo "MODEL_OUTPUT:    ${MODEL_OUTPUT}"
    echo "OPENAI_SCORING_MODEL: ${OPENAI_SCORING_MODEL}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS:        ${NUM_GPUS}"
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

OPENAI_ENV="${EVAL_DIR}/openai.env"
if [[ -f "${OPENAI_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${OPENAI_ENV}"
fi

if [[ "${SCORE_ONLY}" != "1" ]]; then
  if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
    echo "ERROR: Qwen-Image-Edit model not found: ${QWEN_MODEL_PATH}" >&2
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
  bash "${EVAL_DIR}/download_imgedit_bench.sh"
  "${PYTHON}" -c "from diffusers import QwenImageEditPlusPipeline" || {
    echo "ERROR: QwenImageEditPlusPipeline is unavailable in ${PYTHON}." >&2
    exit 1
  }
fi

write_run_config "started"

echo "[TeleStyleV2 QIE-2511 / ImgEdit]"
echo "  BASE=${QWEN_MODEL_PATH}"
echo "  LORAS=${QWEN_LORA_PATHS}"
echo "  steps=${QWEN_STEPS}  true_cfg=${QWEN_TRUE_CFG_SCALE}  seed=${QWEN_SEED}  fuse=${QWEN_FUSE_LORA}"
echo "  score=${OPENAI_SCORING_MODEL}  gpus=${NUM_GPUS}  cuda=${CUDA_VISIBLE_DEVICES}"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  gen_cmd=(
    "${PYTHON}" "${EVAL_DIR}/run_editflow_imgedit_infer.py"
    --role qwen
    --suite "${SUITE}"
    --bench_root "${IMGEDIT_BENCH_ROOT}"
    --output_dir "${MODEL_OUTPUT}"
    --model_path "${QWEN_MODEL_PATH}"
    --run_name "${RUN_NAME}"
    --num_inference_steps "${QWEN_STEPS}"
    --guidance_scale "${QWEN_GUIDANCE}"
    --seed "${QWEN_SEED}"
    --num_gpus "${NUM_GPUS}"
    --skip_existing
    --write_case_bundles
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
  score_cmd=(
    "${PYTHON}" "${EVAL_DIR}/run_imgedit_score.py"
    --role qwen
    --model_output "${MODEL_OUTPUT}"
    --bench_root "${IMGEDIT_BENCH_ROOT}"
    --annotations_dir "${EVAL_DIR}/annotations"
    --num_processes "${NUM_PROCESSES}"
  )
  if [[ "${FORCE_SCORE}" == "1" ]]; then
    score_cmd+=(--force)
  fi
  echo "[score] ${score_cmd[*]}"
  "${score_cmd[@]}"
fi

write_run_config "completed"

echo ""
echo "Done."
echo "  run folder: ${RUN_OUTPUT_ROOT}"
echo "  outputs:    ${MODEL_OUTPUT}"
if [[ "${GEN_ONLY}" != "1" ]]; then
  echo "  scores:     ${MODEL_OUTPUT}/scores.txt"
fi
