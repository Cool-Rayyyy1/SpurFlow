#!/usr/bin/env bash
# ImgEdit-Bench student inference for gmkontext_uedit (4-head + epsilon in proj).
#
# Uses:
#   CONFIG=configs/kontext/editflux_uedit_2nfe_k16_data.py
#   CKPT=checkpoints/gmkontext_uedit_k16_2nfe_pico400k/.../iter_5000.pth
#
# For fixedeps (NO proj_out_epsilon), use:
#   bash evaluation/run_gmkontext_uedit_fixedeps_infer.sh
#
# For fixedeps split-stage (ArcFlowEditImitationSplitStage), use:
#   bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_infer.sh
#
# Teacher generation is skipped. Comparison panels use an existing teacher run
# (default: 20260610_011516_iter_5000) plus this student output.
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_infer.sh
#   SUITE=basic MAX_SAMPLES=8 bash evaluation/run_gmkontext_uedit_infer.sh
#   GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_infer.sh   # skip GPT scoring
#   SKIP_STUDENT_GEN=1 GEN_ONLY=1 bash ...                      # comparisons only (student already done)

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"
EVAL_DIR="${EVAL_DIR:-${EDITFLOW_DIR}/evaluation/imgedit_bench}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/dataset/imgedit}"

CONDA_ROOT="${CONDA_ROOT:-${WORKSPACE_ROOT}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-${WORKSPACE_ROOT}/pretrained_models/FLUX.1-Kontext-dev}"
export IMGEDIT_BENCH_ROOT="${IMGEDIT_BENCH_ROOT:-${DATA_ROOT}/benchmark/Benchmark}"
export PYTHONPATH="${EDITFLOW_DIR}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
export NUM_GPUS

RUN_NAME="${RUN_NAME:-gmkontext_uedit_k16_2nfe_pico400k}"
CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_2nfe_k16_data.py}"
CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_k16_2nfe_pico400k/20260612_145638/iter_5000.pth}"
RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT}" .pth)}"
REF_TEACHER_OUTPUT="${REF_TEACHER_OUTPUT:-${EVAL_DIR}/outputs/runs/20260610_011516_iter_5000/teacher}"

STUDENT_NFE="${STUDENT_NFE:-2}"
STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export STUDENT_GUIDANCE

SUITE="${SUITE:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GEN_ONLY="${GEN_ONLY:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
SKIP_STUDENT_GEN="${SKIP_STUDENT_GEN:-0}"
BUILD_COMPARISONS="${BUILD_COMPARISONS:-1}"
WRITE_CASE_BUNDLES="${WRITE_CASE_BUNDLES:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-64}"
FORCE_SCORE="${FORCE_SCORE:-0}"

RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${EVAL_DIR}/outputs/runs/${RUN_TAG}}"
STUDENT_OUTPUT="${STUDENT_OUTPUT:-${RUN_OUTPUT_ROOT}/student}"
COMPARISON_OUTPUT="${COMPARISON_OUTPUT:-${RUN_OUTPUT_ROOT}/comparisons}"
SCORES_SUBDIR="${SCORES_SUBDIR:-scores}"
SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores.txt}"
STUDENT_SCORES_DIR="${STUDENT_OUTPUT}/${SCORES_SUBDIR}"
STUDENT_SCORES_TXT="${STUDENT_OUTPUT}/${SCORES_TXT_NAME}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
RUN_CONFIG_TXT="${RUN_OUTPUT_ROOT}/run_config.txt"

resolve_ckpt() {
  if [[ -n "${CKPT}" && -f "${CKPT}" ]]; then
    echo "${CKPT}"
    return 0
  fi
  local cand latest_iter=-1 latest_ckpt=""
  shopt -s nullglob
  for cand in \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/latest.pth" \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/*/latest.pth \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/model/${RUN_NAME}/latest.pth" \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/model/${RUN_NAME}/iter_8500.pth" \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/*/iter_8500.pth \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/iter_8500.pth \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/*/iter_5000.pth \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/iter_5000.pth; do
    if [[ -f "${cand}" ]]; then
      echo "${cand}"
      shopt -u nullglob
      return 0
    fi
  done
  for cand in \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/model/${RUN_NAME}"/iter_*.pth \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/*/iter_*.pth \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/iter_*.pth; do
    if [[ -f "${cand}" && "${cand}" =~ iter_([0-9]+)\.pth$ ]]; then
      if [[ "${BASH_REMATCH[1]}" -gt "${latest_iter}" ]]; then
        latest_iter="${BASH_REMATCH[1]}"
        latest_ckpt="${cand}"
      fi
    fi
  done
  shopt -u nullglob
  if [[ -n "${latest_ckpt}" && -f "${latest_ckpt}" ]]; then
    echo "${latest_ckpt}"
    return 0
  fi
  return 1
}

write_run_config() {
  local status="${1:-started}"
  mkdir -p "${RUN_OUTPUT_ROOT}"
  {
    echo "ImgEdit-Bench gmkontext_uedit Student Inference"
    echo "==============================================="
    echo "Status:          ${status}"
    echo "RUN_NAME:        ${RUN_NAME}"
    echo "RUN_TAG:         ${RUN_TAG}"
    echo "CONFIG:          ${CONFIG}"
    echo "CKPT:            ${CKPT_PATH:-${CKPT}}"
    echo "STUDENT_NFE:     ${STUDENT_NFE}"
    echo "STUDENT_GUIDANCE:${STUDENT_GUIDANCE}"
    echo "STUDENT_RESIZE_MODE:${STUDENT_RESIZE_MODE:-center_crop}"
    echo "SUITE:           ${SUITE}"
    echo "REF_TEACHER:     ${REF_TEACHER_OUTPUT}"
    echo "STUDENT_OUTPUT:  ${STUDENT_OUTPUT}"
    echo "COMPARISONS:     ${COMPARISON_OUTPUT}"
    echo "BUILD_COMPARISONS:${BUILD_COMPARISONS}"
    echo "WRITE_CASE_BUNDLES:${WRITE_CASE_BUNDLES}"
    echo "GEN_ONLY:        ${GEN_ONLY}"
    echo "SKIP_STUDENT_GEN:${SKIP_STUDENT_GEN}"
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
    python "${EVAL_DIR}/run_editflow_imgedit_infer.py"
    --role student
    --suite "${SUITE}"
    --bench_root "${IMGEDIT_BENCH_ROOT}"
    --output_dir "${STUDENT_OUTPUT}"
    --model_path "${KONTEXT_MODEL_PATH}"
    --run_name "${RUN_NAME}"
    --config "${CONFIG}"
    --ckpt "${CKPT_PATH}"
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
  if [[ "${WRITE_CASE_BUNDLES}" == "1" ]]; then
    cmd+=(--write_case_bundles)
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
    --student_output "${STUDENT_OUTPUT}"
    --require_student
    --suite "${SUITE}"
    --skip_existing
  )
  if [[ -d "${REF_TEACHER_OUTPUT}" ]]; then
    cmd+=(--teacher_output "${REF_TEACHER_OUTPUT}")
  else
    echo "[warn] REF_TEACHER_OUTPUT not found: ${REF_TEACHER_OUTPUT}"
    echo "       comparisons will include source + student only"
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

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/setup_env.sh" 2>/dev/null || source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 DIFFUSERS_OFFLINE=0

cd "${EDITFLOW_DIR}"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  bash "${EVAL_DIR}/download_imgedit_bench.sh"
fi

CKPT_PATH=""
if [[ "${SCORE_ONLY}" != "1" ]]; then
  if ! CKPT_PATH="$(resolve_ckpt)"; then
    echo "ERROR: checkpoint not found. Set CKPT=/path/to/iter_5000.pth" >&2
    exit 1
  fi
  echo "[ckpt] ${CKPT_PATH}"
fi

write_run_config "started"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  echo "[gen] student ${RUN_NAME} (${STUDENT_NFE} NFE, guidance=${STUDENT_GUIDANCE})"
  if [[ "${SKIP_STUDENT_GEN}" != "1" ]]; then
    run_student_generation
  else
    echo "[skip] SKIP_STUDENT_GEN=1 — reusing existing student outputs in ${STUDENT_OUTPUT}"
  fi
  run_comparisons
fi

if [[ "${GEN_ONLY}" != "1" ]]; then
  echo "[score] student -> ${STUDENT_SCORES_TXT}"
  run_score
fi

write_run_config "completed"

echo ""
echo "Done."
echo "  run folder:      ${RUN_OUTPUT_ROOT}"
echo "  student outputs: ${STUDENT_OUTPUT}"
if [[ "${WRITE_CASE_BUNDLES}" == "1" ]]; then
  echo "  basic cases:     ${STUDENT_OUTPUT}/basic/{Action,Add,...}/<key>/{src,pred,prompt}"
fi
echo "  comparisons:     ${COMPARISON_OUTPUT}"
if [[ "${GEN_ONLY}" != "1" ]]; then
  echo "  student scores:  ${STUDENT_SCORES_TXT}"
fi