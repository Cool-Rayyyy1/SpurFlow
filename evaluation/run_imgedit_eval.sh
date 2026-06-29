#!/usr/bin/env bash
# ImgEdit-Bench for EditFlow: distilled student eval (single-GPU default).
#
# Default: 4-NFE student (iter_10000) + GPT scoring; teacher generation is skipped.
#
#   bash evaluation/run_imgedit_eval.sh
#
# Flags:
#   GEN_ONLY=1          generation only, no GPT scoring
#   SCORE_ONLY=1        scoring only (respects SKIP_STUDENT for teacher vs student)
#   SKIP_STUDENT=1      teacher only
#   SKIP_TEACHER=0      also run teacher baseline generation/scoring
#   FORCE_TEACHER=1     re-run teacher generation
#   FORCE_SCORE=1       re-run GPT scoring
#
# GPT credentials: evaluation/imgedit_bench/openai.env

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

RUN_NAME="${RUN_NAME:-gmkontext_arcflow_4nfe_pico400k}"
CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_4nfe_k16.py}"
CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_arcflow_4nfe_pico400k/iter_10000.pth}"
RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT}" .pth)}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
GEN_ONLY="${GEN_ONLY:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"
SKIP_TEACHER="${SKIP_TEACHER:-1}"
FORCE_TEACHER="${FORCE_TEACHER:-0}"
FORCE_SCORE="${FORCE_SCORE:-0}"
SKIP_STUDENT="${SKIP_STUDENT:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SUITE="${SUITE:-all}"
BUILD_COMPARISONS="${BUILD_COMPARISONS:-1}"

TEACHER_STEPS="${TEACHER_STEPS:-28}"
TEACHER_GUIDANCE="${TEACHER_GUIDANCE:-2.5}"
STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-${TEACHER_GUIDANCE}}"
export TEACHER_STEPS TEACHER_GUIDANCE STUDENT_GUIDANCE

parse_nfe_from_run_name() {
  local name="$1"
  if [[ "${name}" =~ ([0-9]+)nfe ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  echo "2"
}

STUDENT_NFE="${STUDENT_NFE:-$(parse_nfe_from_run_name "${RUN_NAME}")}"

RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${EVAL_DIR}/outputs/runs/${RUN_TAG}}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-${RUN_OUTPUT_ROOT}/teacher}"
STUDENT_OUTPUT="${STUDENT_OUTPUT:-${RUN_OUTPUT_ROOT}/student}"
COMPARISON_OUTPUT="${COMPARISON_OUTPUT:-${RUN_OUTPUT_ROOT}/comparisons}"
RUN_CONFIG_TXT="${RUN_OUTPUT_ROOT}/run_config.txt"
TEACHER_SCORES_TXT="${TEACHER_OUTPUT}/scores.txt"
STUDENT_SCORES_TXT="${STUDENT_OUTPUT}/scores.txt"

resolve_ckpt() {
  if [[ -n "${CKPT}" && -f "${CKPT}" ]]; then
    echo "${CKPT}"
    return 0
  fi
  local cand
  shopt -s nullglob
  for cand in \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/latest.pth" \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/iter_*.pth; do
    if [[ -f "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

resolve_adapter_dir() {
  local ckpt_path="$1"
  local ckpt_tag ckpt_dir
  ckpt_tag="$(basename "${ckpt_path}" .pth)"
  ckpt_dir="$(dirname "${ckpt_path}")"
  local adapter="${ckpt_dir}/diffusers_adapter_${ckpt_tag}"
  if [[ -f "${adapter}/diffusion_pytorch_model.safetensors" ]]; then
    echo "${adapter}"
    return 0
  fi
  adapter="${ckpt_dir}/diffusers_adapter_latest"
  if [[ -f "${adapter}/diffusion_pytorch_model.safetensors" ]]; then
    echo "${adapter}"
    return 0
  fi
  echo "${ckpt_dir}/diffusers_adapter_${ckpt_tag}"
}

write_run_config() {
  local status="${1:-started}"
  mkdir -p "${RUN_OUTPUT_ROOT}"
  {
    echo "ImgEdit-Bench Run Configuration"
    echo "==============================="
    echo "Status:          ${status}"
    echo "Started:         ${RUN_STARTED_AT:-$(date '+%Y-%m-%d %H:%M:%S')}"
    echo "Updated:         $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "[Run identity]"
    echo "RUN_TAG:         ${RUN_TAG}"
    echo "RUN_OUTPUT_ROOT: ${RUN_OUTPUT_ROOT}"
    echo "RUN_CONFIG_TXT:  ${RUN_CONFIG_TXT}"
    echo ""
    echo "[GPU]"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS:             ${NUM_GPUS}"
    echo "CPU_OFFLOAD:          ${CPU_OFFLOAD}"
    echo ""
    echo "[Teacher — FLUX.1-Kontext baseline]"
    echo "Model:           ${KONTEXT_MODEL_PATH}"
    echo "Steps:           ${TEACHER_STEPS}"
    echo "Guidance:        ${TEACHER_GUIDANCE}"
    echo "Output:          ${TEACHER_OUTPUT}"
    echo "Scores:          ${TEACHER_SCORES_TXT}"
    echo ""
    echo "[Student — EditFlow distilled]"
    echo "RUN_NAME:        ${RUN_NAME}"
    echo "CONFIG:          ${CONFIG}"
    echo "CKPT:            ${CKPT}"
    echo "CKPT_RESOLVED:   ${CKPT_PATH:-pending}"
    echo "ADAPTER_DIR:     ${ADAPTER_DIR:-pending}"
    echo "STUDENT_NFE:     ${STUDENT_NFE}"
    echo "Guidance:        ${STUDENT_GUIDANCE}"
    echo "Output:          ${STUDENT_OUTPUT}"
    echo "Scores:          ${STUDENT_SCORES_TXT}"
    echo ""
    echo "[Benchmark]"
    echo "SUITE:           ${SUITE}"
    echo "MAX_SAMPLES:     ${MAX_SAMPLES:-all}"
    echo "BENCH_ROOT:      ${IMGEDIT_BENCH_ROOT}"
    echo ""
    echo "[Pipeline flags]"
    echo "GEN_ONLY:        ${GEN_ONLY}"
    echo "SCORE_ONLY:      ${SCORE_ONLY}"
    echo "SKIP_TEACHER:    ${SKIP_TEACHER}"
    echo "SKIP_STUDENT:    ${SKIP_STUDENT}"
    echo "SKIP_EXPORT:     ${SKIP_EXPORT}"
    echo "FORCE_TEACHER:   ${FORCE_TEACHER}"
    echo "FORCE_SCORE:     ${FORCE_SCORE}"
    echo "NUM_PROCESSES:   ${NUM_PROCESSES}"
    echo ""
    echo "[Environment]"
    echo "EDITFLOW_DIR:    ${EDITFLOW_DIR}"
    echo "EVAL_DIR:        ${EVAL_DIR}"
    echo "CONDA_ENV:       ${CONDA_ENV}"
    echo "HF_ENDPOINT:     ${HF_ENDPOINT}"
    echo "HF_HOME:         ${HF_HOME}"
    echo "OPENAI_ENV:      ${OPENAI_ENV:-${EVAL_DIR}/openai.env}"
    if [[ "${status}" == "completed" ]]; then
      echo ""
      echo "[Results]"
      if [[ -f "${TEACHER_SCORES_TXT}" ]]; then
        echo "--- teacher/scores.txt ---"
        cat "${TEACHER_SCORES_TXT}"
      fi
      if [[ -f "${STUDENT_SCORES_TXT}" ]]; then
        echo "--- student/scores.txt ---"
        cat "${STUDENT_SCORES_TXT}"
      fi
    fi
  } > "${RUN_CONFIG_TXT}"
  echo "[config] ${RUN_CONFIG_TXT}"
}

teacher_complete() {
  python "${EVAL_DIR}/run_editflow_imgedit_infer.py" \
    --role teacher \
    --suite "${SUITE}" \
    --bench_root "${IMGEDIT_BENCH_ROOT}" \
    --output_dir "${TEACHER_OUTPUT}" \
    --model_path "${KONTEXT_MODEL_PATH}" \
    --check_only
}

run_generation() {
  local role="$1"
  local output_dir="$2"
  local adapter_dir="${3:-}"
  local nfe="${4:-}"
  local -a cmd=(
    python "${EVAL_DIR}/run_editflow_imgedit_infer.py"
    --role "${role}"
    --suite "${SUITE}"
    --bench_root "${IMGEDIT_BENCH_ROOT}"
    --output_dir "${output_dir}"
    --model_path "${KONTEXT_MODEL_PATH}"
    --run_name "${RUN_NAME}"
    --skip_existing
  )
  if [[ "${role}" == "teacher" ]]; then
    cmd+=(--num_inference_steps "${TEACHER_STEPS}" --guidance_scale "${TEACHER_GUIDANCE}")
  else
    cmd+=(--config "${CONFIG}" --ckpt "${CKPT_PATH}")
    cmd+=(--num_inference_steps "${nfe}" --guidance_scale "${STUDENT_GUIDANCE}")
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ "${CPU_OFFLOAD}" == "1" ]]; then
    cmd+=(--cpu_offload)
  fi
  cmd+=(--num_gpus "${NUM_GPUS}")
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

run_score() {
  local role="$1"
  local model_output="$2"
  local -a cmd=(
    python "${EVAL_DIR}/run_imgedit_score.py"
    --role "${role}"
    --model_output "${model_output}"
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

run_comparisons() {
  if [[ "${BUILD_COMPARISONS}" != "1" || "${SCORE_ONLY}" == "1" ]]; then
    return 0
  fi
  local teacher_arg=() student_arg=() require_student_arg=()
  if [[ -d "${TEACHER_OUTPUT}" ]]; then
    teacher_arg=(--teacher_output "${TEACHER_OUTPUT}")
  fi
  if [[ "${SKIP_STUDENT}" != "1" && -d "${STUDENT_OUTPUT}" ]]; then
    student_arg=(--student_output "${STUDENT_OUTPUT}")
    require_student_arg=(--require_student)
  fi
  if [[ ${#teacher_arg[@]} -eq 0 && ${#student_arg[@]} -eq 0 ]]; then
    echo "[skip] comparisons: no teacher/student outputs found"
    return 0
  fi
  local -a cmd=(
    python "${EVAL_DIR}/build_comparison_grids.py"
    --bench_root "${IMGEDIT_BENCH_ROOT}"
    --annotations_dir "${EVAL_DIR}/annotations"
    --output_dir "${COMPARISON_OUTPUT}"
    --suite "${SUITE}"
    --skip_existing
  )
  cmd+=("${teacher_arg[@]}" "${student_arg[@]}" "${require_student_arg[@]}")
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

print_teacher_scores() {
  python "${EVAL_DIR}/run_imgedit_score.py" \
    --role teacher \
    --model_output "${TEACHER_OUTPUT}" \
    --bench_root "${IMGEDIT_BENCH_ROOT}" \
    --print_only
}

print_student_scores() {
  if [[ -f "${STUDENT_SCORES_TXT}" ]]; then
    python "${EVAL_DIR}/run_imgedit_score.py" \
      --role student \
      --model_output "${STUDENT_OUTPUT}" \
      --bench_root "${IMGEDIT_BENCH_ROOT}" \
      --print_only
  fi
}

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/setup_env.sh" 2>/dev/null || source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 DIFFUSERS_OFFLINE=0

cd "${EDITFLOW_DIR}"

OPENAI_ENV="${EVAL_DIR}/openai.env"
if [[ -f "${OPENAI_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${OPENAI_ENV}"
fi

RUN_STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
CKPT_PATH=""
ADAPTER_DIR=""
write_run_config "started"

if [[ "${SCORE_ONLY}" != "1" ]]; then
  bash "${EVAL_DIR}/download_imgedit_bench.sh"
fi

# ---------- generation ----------
if [[ "${SCORE_ONLY}" != "1" ]]; then
  if [[ "${SKIP_TEACHER}" != "1" ]]; then
    if [[ "${FORCE_TEACHER}" == "1" ]] || ! teacher_complete >/dev/null 2>&1; then
      echo "[gen] teacher (Kontext: steps=${TEACHER_STEPS}, guidance=${TEACHER_GUIDANCE})"
      run_generation teacher "${TEACHER_OUTPUT}" "" ""
    else
      echo "[skip] teacher generation already complete at ${TEACHER_OUTPUT}"
    fi
  fi

  if [[ "${SKIP_STUDENT}" != "1" ]]; then
    CKPT_PATH=""
    ADAPTER_DIR=""
    if CKPT_PATH="$(resolve_ckpt)"; then
      echo "[ckpt] ${CKPT_PATH}"
      ADAPTER_DIR="$(resolve_adapter_dir "${CKPT_PATH}")"
      if [[ "${SKIP_EXPORT}" != "1" && ! -f "${ADAPTER_DIR}/diffusion_pytorch_model.safetensors" ]]; then
        echo "[export] ${CKPT_PATH} -> ${ADAPTER_DIR} (optional diffusers export)"
        python export_arcflow_to_diffusers.py "${CONFIG}" --ckpt "${CKPT_PATH}" --out-dir "${ADAPTER_DIR}"
      fi
    else
      echo "ERROR: no checkpoint found for RUN_NAME=${RUN_NAME}. Set CKPT=/path/to/latest.pth" >&2
      exit 1
    fi
    echo "[gen] student ${RUN_NAME} (${STUDENT_NFE} NFE, guidance=${STUDENT_GUIDANCE}) via val_step"
    run_generation student "${STUDENT_OUTPUT}" "" "${STUDENT_NFE}"
  fi

  run_comparisons
fi

# ---------- scoring ----------
if [[ "${GEN_ONLY}" != "1" ]]; then
  if [[ "${SKIP_TEACHER}" != "1" ]]; then
    if [[ "${FORCE_SCORE}" == "1" || ! -f "${TEACHER_SCORES_TXT}" ]]; then
      echo "[score] teacher -> ${TEACHER_SCORES_TXT}"
      run_score teacher "${TEACHER_OUTPUT}"
    else
      echo "[skip] teacher scores already exist: ${TEACHER_SCORES_TXT}"
    fi
  fi

  if [[ "${SKIP_STUDENT}" != "1" ]]; then
    if [[ "${FORCE_SCORE}" == "1" || ! -f "${STUDENT_SCORES_TXT}" ]]; then
      echo "[score] student -> ${STUDENT_SCORES_TXT}"
      run_score student "${STUDENT_OUTPUT}"
    else
      echo "[skip] student scores already exist: ${STUDENT_SCORES_TXT}"
    fi
  fi
fi

write_run_config "completed"

# ---------- summary ----------
echo ""
echo "Done."
echo "  run folder:      ${RUN_OUTPUT_ROOT}"
echo "  run config:      ${RUN_CONFIG_TXT}"
echo "  GPUs:            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_GPUS=${NUM_GPUS}"
echo "  teacher outputs: ${TEACHER_OUTPUT}"
echo "  teacher scores:  ${TEACHER_SCORES_TXT}"
if [[ "${BUILD_COMPARISONS}" == "1" ]]; then
  echo "  comparisons:     ${COMPARISON_OUTPUT}"
fi
if [[ "${SKIP_STUDENT}" != "1" ]]; then
  echo "  student outputs: ${STUDENT_OUTPUT}"
  echo "  student scores:  ${STUDENT_SCORES_TXT}"
fi

if [[ -f "${TEACHER_SCORES_TXT}" ]]; then
  print_teacher_scores
fi
if [[ "${SKIP_STUDENT}" != "1" && -f "${STUDENT_SCORES_TXT}" ]]; then
  print_student_scores
fi
