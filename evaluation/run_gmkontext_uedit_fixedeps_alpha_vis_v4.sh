#!/usr/bin/env bash
# Alpha v4: 4 images per sample (src / edit / heatmap / overlay on src).
# Output: .../alpha_vis_v4/{basic,uge}/<id>_src.png etc.
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_alpha_vis_v4.sh
#   SUITE=basic MAX_SAMPLES=4 bash evaluation/run_gmkontext_uedit_fixedeps_alpha_vis_v4.sh

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"
EVAL_DIR="${EDITFLOW_DIR}/evaluation/imgedit_bench"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/dataset/imgedit}"

CONDA_ROOT="${CONDA_ROOT:-${WORKSPACE_ROOT}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-${WORKSPACE_ROOT}/pretrained_models/FLUX.1-Kontext-dev}"
export IMGEDIT_BENCH_ROOT="${IMGEDIT_BENCH_ROOT:-${DATA_ROOT}/benchmark/Benchmark}"
export PYTHONPATH="${EDITFLOW_DIR}:${EVAL_DIR}:${PYTHONPATH:-}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"

RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k}"
CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data.py}"
CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_alpha/model/gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k/iter_8500.pth}"
RUN_TAG="${RUN_TAG:-gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_iter_8500}"

STUDENT_NFE="${STUDENT_NFE:-2}"
STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
SUITE="${SUITE:-basic_uge}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
FORCE="${FORCE:-0}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-0.1}"

OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/outputs/runs/${RUN_TAG}/alpha_vis_v4}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
source "${EDITFLOW_DIR}/setup_env.sh" 2>/dev/null || source "${EDITFLOW_DIR}/env.sh" 2>/dev/null || true

cd "${EDITFLOW_DIR}"
bash "${EVAL_DIR}/download_imgedit_bench.sh"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

SKIP_FLAG=()
if [[ "${FORCE}" != "1" ]]; then
  SKIP_FLAG=(--skip_existing)
fi

cmd=(
  python "${EVAL_DIR}/run_alpha_vis_bench_v4.py"
  --suite "${SUITE}"
  --bench_root "${IMGEDIT_BENCH_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --config "${CONFIG}"
  --ckpt "${CKPT}"
  --run_name "${RUN_NAME}"
  --num_inference_steps "${STUDENT_NFE}"
  --guidance_scale "${STUDENT_GUIDANCE}"
  --num_gpus "${NUM_GPUS}"
  --alpha_threshold "${ALPHA_THRESHOLD}"
  "${SKIP_FLAG[@]}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=(--max_samples "${MAX_SAMPLES}")
fi

echo "Running: ${cmd[*]}"
echo "Output:  ${OUTPUT_DIR}"
echo "Per sample: *_src.png  *_edit.png  *_heatmap.png  *_overlay.png"
"${cmd[@]}"

echo "Done. Alpha v4 -> ${OUTPUT_DIR}"
