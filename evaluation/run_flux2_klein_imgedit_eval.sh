#!/usr/bin/env bash
# ImgEdit-Bench for official distilled FLUX.2-klein-9B (4-step, Flux2KleinPipeline).
#
# This is NOT FLUX.2-klein-base-9B (undistilled, 50-step CFG=4).
# Official distilled card: num_inference_steps=4, guidance_scale=1.0.
#
# Output:
#   evaluation/imgedit_bench/outputs/runs/flux2_klein_9b_4step/
#     model/basic|uge|multiturn/
#     model/scores_gpt4o.txt
#     model/scores_gpt4o/
#
# Usage:
#   bash evaluation/run_flux2_klein_imgedit_eval.sh
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash evaluation/run_flux2_klein_imgedit_eval.sh
#   SUITE=basic MAX_SAMPLES=8 bash evaluation/run_flux2_klein_imgedit_eval.sh
#   GEN_ONLY=1 bash evaluation/run_flux2_klein_imgedit_eval.sh
#   SCORE_ONLY=1 bash evaluation/run_flux2_klein_imgedit_eval.sh

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
export NUM_GPUS

# Distilled 4-step Klein (not klein-base).
KLEIN_MODEL_PATH="${KLEIN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-9B}"
RUN_NAME="${RUN_NAME:-flux2_klein_9b}"
RUN_TAG="${RUN_TAG:-${RUN_NAME}_4step}"
KLEIN_STEPS="${KLEIN_STEPS:-4}"
KLEIN_GUIDANCE="${KLEIN_GUIDANCE:-1.0}"
export KLEIN_STEPS KLEIN_GUIDANCE

REF_TEACHER_OUTPUT="${REF_TEACHER_OUTPUT:-}"

SUITE="${SUITE:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GEN_ONLY="${GEN_ONLY:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
BUILD_COMPARISONS="${BUILD_COMPARISONS:-1}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
FORCE_SCORE="${FORCE_SCORE:-0}"

RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${EVAL_DIR}/outputs/runs/${RUN_TAG}}"
MODEL_OUTPUT="${MODEL_OUTPUT:-${RUN_OUTPUT_ROOT}/model}"
COMPARISON_OUTPUT="${COMPARISON_OUTPUT:-${RUN_OUTPUT_ROOT}/comparisons}"
SCORES_SUBDIR="${SCORES_SUBDIR:-scores_gpt4o}"
SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores_gpt4o.txt}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
SCORES_TXT="${MODEL_OUTPUT}/${SCORES_TXT_NAME}"
RUN_CONFIG_TXT="${RUN_OUTPUT_ROOT}/run_config.txt"

ensure_klein_diffusers() {
  python - <<'PY' || {
import sys
try:
    from diffusers import Flux2KleinPipeline  # noqa: F401
except ImportError:
    sys.exit(1)
PY
    echo "[deps] installing diffusers>=0.37 for Flux2KleinPipeline ..."
    pip install -q 'diffusers>=0.37.0'
  }
}

write_run_config() {
  local status="${1:-started}"
  mkdir -p "${RUN_OUTPUT_ROOT}"
  {
    echo "ImgEdit-Bench FLUX.2-klein-9B (distilled 4-step) Evaluation"
    echo "=========================================================="
    echo "Status:          ${status}"
    echo "RUN_TAG:         ${RUN_TAG}"
    echo "KLEIN_MODEL:     ${KLEIN_MODEL_PATH}"
    echo "KLEIN_STEPS:     ${KLEIN_STEPS}   # official distilled 4"
    echo "KLEIN_GUIDANCE:  ${KLEIN_GUIDANCE}   # official distilled CFG=1"
    echo "SUITE:           ${SUITE}"
    echo "MODEL_OUTPUT:    ${MODEL_OUTPUT}"
    echo "COMPARISONS:     ${COMPARISON_OUTPUT}"
    echo "SCORING_MODEL:   ${OPENAI_SCORING_MODEL}"
    echo "SCORES_TXT:      ${SCORES_TXT}"
    echo "GEN_ONLY:        ${GEN_ONLY}"
    echo "NUM_PROCESSES:   ${NUM_PROCESSES}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS:        ${NUM_GPUS}"
  } > "${RUN_CONFIG_TXT}"
  echo "[config] ${RUN_CONFIG_TXT}"
}

run_klein_generation() {
  local -a cmd=(
    python "${EVAL_DIR}/run_editflow_imgedit_infer.py"
    --role klein
    --suite "${SUITE}"
    --bench_root "${IMGEDIT_BENCH_ROOT}"
    --output_dir "${MODEL_OUTPUT}"
    --model_path "${KLEIN_MODEL_PATH}"
    --run_name "${RUN_NAME}"
    --num_inference_steps "${KLEIN_STEPS}"
    --guidance_scale "${KLEIN_GUIDANCE}"
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

run_comparisons() {
  if [[ "${BUILD_COMPARISONS}" != "1" ]]; then
    return 0
  fi
  local -a cmd=(
    python "${EVAL_DIR}/build_comparison_grids.py"
    --bench_root "${IMGEDIT_BENCH_ROOT}"
    --annotations_dir "${EVAL_DIR}/annotations"
    --output_dir "${COMPARISON_OUTPUT}"
    --student_output "${MODEL_OUTPUT}"
    --require_student
    --suite "${SUITE}"
    --skip_existing
  )
  if [[ -n "${REF_TEACHER_OUTPUT}" && -d "${REF_TEACHER_OUTPUT}" ]]; then
    cmd+=(--teacher_output "${REF_TEACHER_OUTPUT}")
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

run_score() {
  local -a cmd=(
    python "${EVAL_DIR}/run_imgedit_score.py"
    --role klein
    --model_output "${MODEL_OUTPUT}"
    --scores_dir "${MODEL_OUTPUT}/${SCORES_SUBDIR}"
    --scores_txt "${SCORES_TXT}"
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

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${EDITFLOW_DIR}"
# Keep HF online so a missing local Klein dir can still be fetched if needed.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 DIFFUSERS_OFFLINE=0

OPENAI_ENV="${EVAL_DIR}/openai.env"
if [[ -f "${OPENAI_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${OPENAI_ENV}"
fi

if [[ ! -d "${KLEIN_MODEL_PATH}" ]] || [[ ! -f "${KLEIN_MODEL_PATH}/model_index.json" ]]; then
  echo "ERROR: Distilled FLUX.2-klein-9B not found: ${KLEIN_MODEL_PATH}" >&2
  echo "This eval needs the 4-step distilled Klein, not FLUX.2-klein-base-9B." >&2
  echo "Download with:" >&2
  echo "  huggingface-cli download black-forest-labs/FLUX.2-klein-9B --local-dir ${KLEIN_MODEL_PATH}" >&2
  exit 1
fi

write_run_config "started"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  if [[ ! -d "${IMGEDIT_BENCH_ROOT}/singleturn" ]]; then
    bash "${EVAL_DIR}/download_imgedit_bench.sh"
  fi
  ensure_klein_diffusers
  echo "[gen] FLUX.2-klein-9B distilled (${KLEIN_STEPS} steps, guidance=${KLEIN_GUIDANCE})"
  run_klein_generation
  run_comparisons
fi

if [[ "${GEN_ONLY}" != "1" ]]; then
  echo "[score] klein gpt-4o -> ${SCORES_TXT}"
  run_score
fi

write_run_config "completed"

echo ""
echo "Done."
echo "  run folder:   ${RUN_OUTPUT_ROOT}"
echo "  images:       ${MODEL_OUTPUT}"
echo "  comparisons:  ${COMPARISON_OUTPUT}"
if [[ "${GEN_ONLY}" != "1" ]]; then
  echo "  scores:       ${SCORES_TXT}  (${OPENAI_SCORING_MODEL})"
fi
