#!/usr/bin/env bash
# Two-GPU Qwen-Image-Edit-2511 baseline on ImgEdit-Bench: generate, then score.
#
# Usage:
#   bash evaluation/run_qwen_image_edit_2511_imgedit_eval.sh
#   GEN_ONLY=1 bash evaluation/run_qwen_image_edit_2511_imgedit_eval.sh
#   SCORE_ONLY=1 bash evaluation/run_qwen_image_edit_2511_imgedit_eval.sh
#   MAX_SAMPLES=8 GEN_ONLY=1 bash evaluation/run_qwen_image_edit_2511_imgedit_eval.sh

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"
EVAL_DIR="${EVAL_DIR:-${EDITFLOW_DIR}/evaluation/imgedit_bench}"
CONDA_ROOT="${CONDA_ROOT:-${WORKSPACE_ROOT}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

export PYTHONPATH="${EDITFLOW_DIR}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export IMGEDIT_BENCH_ROOT="${IMGEDIT_BENCH_ROOT:-${WORKSPACE_ROOT}/dataset/imgedit/benchmark/Benchmark}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${WORKSPACE_ROOT}/pretrained_models/Qwen-Image-Edit-2511}"
RUN_TAG="${RUN_TAG:-qwen_image_edit_2511_40step_baseline}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${EVAL_DIR}/outputs/runs/${RUN_TAG}}"
MODEL_OUTPUT="${MODEL_OUTPUT:-${RUN_OUTPUT_ROOT}/model}"

SUITE="${SUITE:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GEN_ONLY="${GEN_ONLY:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-64}"
FORCE_SCORE="${FORCE_SCORE:-0}"

# Official Qwen-Image-Edit-2511 baseline settings.
export QWEN_STEPS="${QWEN_STEPS:-40}"
export QWEN_TRUE_CFG_SCALE="${QWEN_TRUE_CFG_SCALE:-4.0}"
export QWEN_GUIDANCE="${QWEN_GUIDANCE:-1.0}"
export QWEN_NEGATIVE_PROMPT="${QWEN_NEGATIVE_PROMPT:- }"
QWEN_SEED="${QWEN_SEED:-0}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${EDITFLOW_DIR}"

if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
  echo "ERROR: Qwen model not found: ${QWEN_MODEL_PATH}" >&2
  exit 1
fi

OPENAI_ENV="${EVAL_DIR}/openai.env"
if [[ -f "${OPENAI_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${OPENAI_ENV}"
fi

mkdir -p "${RUN_OUTPUT_ROOT}"
{
  echo "ImgEdit-Bench Qwen-Image-Edit-2511 Baseline"
  echo "MODEL: ${QWEN_MODEL_PATH}"
  echo "OUTPUT: ${MODEL_OUTPUT}"
  echo "SUITE: ${SUITE}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "NUM_GPUS: ${NUM_GPUS}"
  echo "STEPS: ${QWEN_STEPS}"
  echo "TRUE_CFG_SCALE: ${QWEN_TRUE_CFG_SCALE}"
  echo "GUIDANCE_SCALE: ${QWEN_GUIDANCE}"
  echo "NEGATIVE_PROMPT: <single space>"
  echo "SEED: ${QWEN_SEED}"
} > "${RUN_OUTPUT_ROOT}/run_config.txt"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  bash "${EVAL_DIR}/download_imgedit_bench.sh"

  python -c "from diffusers import QwenImageEditPlusPipeline" || {
    echo "ERROR: QwenImageEditPlusPipeline is unavailable in ${CONDA_ENV}." >&2
    echo "Install the latest diffusers version before running this evaluation." >&2
    exit 1
  }

  gen_cmd=(
    python "${EVAL_DIR}/run_editflow_imgedit_infer.py"
    --role qwen
    --suite "${SUITE}"
    --bench_root "${IMGEDIT_BENCH_ROOT}"
    --output_dir "${MODEL_OUTPUT}"
    --model_path "${QWEN_MODEL_PATH}"
    --run_name "qwen_image_edit_2511"
    --num_inference_steps "${QWEN_STEPS}"
    --guidance_scale "${QWEN_GUIDANCE}"
    --seed "${QWEN_SEED}"
    --num_gpus "${NUM_GPUS}"
    --skip_existing
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    gen_cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ "${CPU_OFFLOAD}" == "1" ]]; then
    gen_cmd+=(--cpu_offload)
  fi
  echo "[generate] ${gen_cmd[*]}"
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

echo "Done: ${RUN_OUTPUT_ROOT}"
