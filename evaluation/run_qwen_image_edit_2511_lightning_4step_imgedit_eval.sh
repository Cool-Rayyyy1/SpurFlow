#!/usr/bin/env bash
# ImgEdit-Bench for official Qwen-Image-Edit-2511 Lightning 4-step LoRA.
#
# Matches ModelTC/Qwen-Image-Lightning generate_with_diffusers.py:
#   QwenImageEditPlusPipeline + Lightning LoRA
#   FlowMatchEulerDiscreteScheduler (exponential, shift=3)
#   steps=4, true_cfg_scale=1.0, negative_prompt=" ", seed=42
# Then GPT-4.1 scoring.
#
# Usage (1 GPU):
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
#     bash evaluation/run_qwen_image_edit_2511_lightning_4step_imgedit_eval.sh
#
#   SUITE=basic MAX_SAMPLES=8 GEN_ONLY=1 \
#     bash evaluation/run_qwen_image_edit_2511_lightning_4step_imgedit_eval.sh

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
export QWEN_LORA_PATH="${QWEN_LORA_PATH:-/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

export QWEN_STEPS="${QWEN_STEPS:-4}"
export QWEN_TRUE_CFG_SCALE="${QWEN_TRUE_CFG_SCALE:-1.0}"
export QWEN_GUIDANCE="${QWEN_GUIDANCE:-1.0}"
export QWEN_NEGATIVE_PROMPT="${QWEN_NEGATIVE_PROMPT:- }"
QWEN_SEED="${QWEN_SEED:-42}"

export RUN_NAME="${RUN_NAME:-qwen_image_edit_2511_lightning_4step}"
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
    echo "ImgEdit-Bench Qwen-Image-Edit-2511 Lightning 4-step"
    echo "=================================================="
    echo "Status:          ${status}"
    echo "RUN_TAG:         ${RUN_TAG}"
    echo "BASE:            ${QWEN_MODEL_PATH}"
    echo "LORA:            ${QWEN_LORA_PATH}"
    echo "STEPS:           ${QWEN_STEPS}"
    echo "TRUE_CFG_SCALE:  ${QWEN_TRUE_CFG_SCALE}"
    echo "GUIDANCE:        ${QWEN_GUIDANCE}"
    echo "NEGATIVE_PROMPT: <single space>"
    echo "SEED:            ${QWEN_SEED}"
    echo "SCHEDULER:       FlowMatchEulerDiscreteScheduler exponential shift=3"
    echo "SUITE:           ${SUITE}"
    echo "MODEL_OUTPUT:    ${MODEL_OUTPUT}"
    echo "OPENAI_SCORING_MODEL: ${OPENAI_SCORING_MODEL}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS:        ${NUM_GPUS}"
  } > "${RUN_CONFIG_TXT}"
  echo "[config] ${RUN_CONFIG_TXT}"
}

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
  if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
    echo "ERROR: Qwen-Image-Edit model not found: ${QWEN_MODEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${QWEN_LORA_PATH}" ]]; then
    echo "ERROR: Lightning LoRA not found: ${QWEN_LORA_PATH}" >&2
    exit 1
  fi
  bash "${EVAL_DIR}/download_imgedit_bench.sh"
  python -c "from diffusers import QwenImageEditPlusPipeline" || {
    echo "ERROR: QwenImageEditPlusPipeline is unavailable in ${CONDA_ENV}." >&2
    exit 1
  }
fi

write_run_config "started"

echo "[qwen lightning 4-step / ImgEdit]"
echo "  BASE=${QWEN_MODEL_PATH}"
echo "  LORA=${QWEN_LORA_PATH}"
echo "  steps=${QWEN_STEPS}  true_cfg=${QWEN_TRUE_CFG_SCALE}  seed=${QWEN_SEED}"
echo "  score=${OPENAI_SCORING_MODEL}  gpus=${NUM_GPUS}  cuda=${CUDA_VISIBLE_DEVICES}"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  gen_cmd=(
    python "${EVAL_DIR}/run_editflow_imgedit_infer.py"
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
    python "${EVAL_DIR}/run_imgedit_score.py"
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
