#!/usr/bin/env bash
# GEdit-v2 student inference (generation only, no GPT scoring).
# Same layout as new2/evaluation/run_gmkontext_uedit_gedit_infer.sh.
#
# Wrappers set CONFIG / CKPT / RUN_NAME / STUDENT_RESIZE_MODE, then exec this.
# Vanilla ArcFlow (no proj_out_alpha): do not pass --require_alpha.
#
# Outputs:
#   student/GEdit_v2/{stem}.jpg
#   student/cases/{edit_type}/{stem}/{src.png,pred.png,prompt.txt}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="${EDITFLOW_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${EDITFLOW_DIR}}"
EVAL_DIR="${EVAL_DIR:-${SCRIPT_DIR}/gedit_v2}"

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/afs_gaochengmin/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev}"
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511}"
export KLEIN_MODEL_PATH="${KLEIN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"
export PYTHONPATH="${EDITFLOW_DIR}:${PYTHONPATH:-}"

mkdir -p "${HF_HOME}/hub"
mkdir -p /mnt/afs_gaochengmin/.cache/torch/hub/checkpoints

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
export NUM_GPUS

RUN_NAME="${RUN_NAME:-gmkontext_k16_2nfe_oss30_pico70_data}"
CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_2nfe_k16_data_oss_pico.py}"
CKPT="${CKPT:-}"
RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT:-iter.pth}" .pth)}"

STUDENT_NFE="${STUDENT_NFE:-2}"
STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export STUDENT_GUIDANCE
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"

META_JSON="${META_JSON:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
REQUIRE_ALPHA="${REQUIRE_ALPHA:-0}"
# Record per-NFE-step mixture pi (dominant_k) and compute ARI vs edit_type.
DUMP_MIXTURE_STATS="${DUMP_MIXTURE_STATS:-1}"

# Pick the matching pretrained backbone from CONFIG, not from leftover env vars.
STUDENT_PRETRAINED="${KONTEXT_MODEL_PATH}"
if [[ "${CONFIG}" == *qwen* ]]; then
  STUDENT_PRETRAINED="${QWEN_MODEL_PATH}"
fi
if [[ "${CONFIG}" == *klein* || "${CONFIG}" == *flux2_klein* ]]; then
  STUDENT_PRETRAINED="${KLEIN_MODEL_PATH}"
fi
export STUDENT_PRETRAINED

RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${EVAL_DIR}/outputs/runs/${RUN_TAG}}"
STUDENT_OUTPUT="${STUDENT_OUTPUT:-${RUN_OUTPUT_ROOT}/student}"
RUN_CONFIG_TXT="${RUN_OUTPUT_ROOT}/run_config.txt"

resolve_ckpt() {
  if [[ -n "${CKPT}" && -f "${CKPT}" ]]; then
    echo "${CKPT}"
    return 0
  fi
  local cand
  shopt -s nullglob
  for cand in \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/latest.pth" \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/*/latest.pth \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/*/iter_*.pth \
    "${EDITFLOW_DIR}/checkpoints/${RUN_NAME}"/iter_*.pth \
    "${EDITFLOW_DIR}/checkpoints"/20*/latest.pth; do
    if [[ -f "${cand}" ]]; then
      echo "${cand}"
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

write_run_config() {
  local status="${1:-started}"
  mkdir -p "${RUN_OUTPUT_ROOT}"
  {
    echo "GEdit-v2 EditFlow student inference (generation only)"
    echo "===================================================="
    echo "Status:          ${status}"
    echo "RUN_NAME:        ${RUN_NAME}"
    echo "RUN_TAG:         ${RUN_TAG}"
    echo "CONFIG:          ${CONFIG}"
    echo "CKPT:            ${CKPT_PATH:-${CKPT}}"
    echo "STUDENT_PRETRAINED: ${STUDENT_PRETRAINED}"
    echo "META_JSON:       ${META_JSON}"
    echo "STUDENT_NFE:     ${STUDENT_NFE}"
    echo "STUDENT_GUIDANCE:${STUDENT_GUIDANCE}"
    echo "STUDENT_RESIZE_MODE:${STUDENT_RESIZE_MODE}"
    echo "STUDENT_OUTPUT:  ${STUDENT_OUTPUT}"
    echo "FLAT_DIR:        ${STUDENT_OUTPUT}/GEdit_v2"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "NUM_GPUS:        ${NUM_GPUS}"
    echo "MAX_SAMPLES:     ${MAX_SAMPLES:-all}"
  } > "${RUN_CONFIG_TXT}"
  echo "[config] ${RUN_CONFIG_TXT}"
}

run_student_generation() {
  local -a cmd=(
    python "${EVAL_DIR}/run_editflow_gedit_infer.py"
    --meta_json "${META_JSON}"
    --output_dir "${STUDENT_OUTPUT}"
    --model_path "${STUDENT_PRETRAINED}"
    --run_name "${RUN_NAME}"
    --config "${CONFIG}"
    --ckpt "${CKPT_PATH}"
    --num_inference_steps "${STUDENT_NFE}"
    --guidance_scale "${STUDENT_GUIDANCE}"
    --skip_existing
    --num_gpus "${NUM_GPUS}"
  )
  if [[ "${REQUIRE_ALPHA}" == "1" ]]; then
    cmd+=(--require_alpha)
  fi
  if [[ "${DUMP_MIXTURE_STATS}" == "1" ]]; then
    cmd+=(--dump_mixture_stats)
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ "${CPU_OFFLOAD}" == "1" ]]; then
    cmd+=(--cpu_offload)
  fi
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/setup_env.sh" 2>/dev/null || source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true

cd "${EDITFLOW_DIR}"
rm -rf "${HOME}/.cache/YAPF"
python -c "import mmcv" >/dev/null

if [[ ! -f "${META_JSON}" ]]; then
  echo "ERROR: GEdit meta not found: ${META_JSON}" >&2
  exit 1
fi

if ! CKPT_PATH="$(resolve_ckpt)"; then
  echo "ERROR: checkpoint not found. Set CKPT=/path/to/iter_XXXX.pth" >&2
  exit 1
fi
echo "[ckpt] ${CKPT_PATH}"

write_run_config "started"

echo "[gen] GEdit-v2 student ${RUN_NAME} (${STUDENT_NFE} NFE, guidance=${STUDENT_GUIDANCE}, resize=${STUDENT_RESIZE_MODE})"
run_student_generation

if [[ "${DUMP_MIXTURE_STATS}" == "1" ]]; then
  echo "[ari] per-step ARI: mixture dominant_k vs edit_type"
  python "${EVAL_DIR}/compute_pi_ari.py" --output_dir "${STUDENT_OUTPUT}" \
    || echo "[ari] WARNING: ARI computation failed"
fi

write_run_config "completed"

echo ""
echo "Done (generation only, no scores)."
echo "  run folder:  ${RUN_OUTPUT_ROOT}"
echo "  flat preds:  ${STUDENT_OUTPUT}/GEdit_v2"
echo "  case dirs:   ${STUDENT_OUTPUT}/cases/{edit_type}/{stem}/{src,pred,prompt}"
