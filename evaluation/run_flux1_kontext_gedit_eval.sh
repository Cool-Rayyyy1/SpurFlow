#!/usr/bin/env bash
# Official FLUX.1-Kontext-dev baseline on GEdit-v2.
# Model-card settings: 28 inference steps, guidance_scale=2.5, native dynamic scheduler.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${EVAL_DIR:-${SCRIPT_DIR}/gedit_bench}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
source "${EDITFLOW_DIR}/env.sh"

export PYTHONPATH="${EDITFLOW_DIR}:${EVAL_DIR}:${SCRIPT_DIR}/imgedit_bench:${PYTHONPATH:-}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

KONTEXT_STEPS="${KONTEXT_STEPS:-28}"
KONTEXT_GUIDANCE="${KONTEXT_GUIDANCE:-2.5}"
META_JSON="${META_JSON:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
RUN_TAG="${RUN_TAG:-flux1_kontext_dev_${KONTEXT_STEPS}step_cfg${KONTEXT_GUIDANCE}_geditv2}"
RUN_ROOT="${RUN_ROOT:-${EVAL_DIR}/outputs/runs/${RUN_TAG}}"
MODEL_OUTPUT="${MODEL_OUTPUT:-${RUN_ROOT}/model}"
SEED="${SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

mkdir -p "${RUN_ROOT}"
cat > "${RUN_ROOT}/run_config.txt" <<EOF
GEdit-v2 FLUX.1-Kontext-dev official baseline
RUN_TAG:             ${RUN_TAG}
MODEL:               ${KONTEXT_MODEL_PATH}
META_JSON:           ${META_JSON}
STEPS:               ${KONTEXT_STEPS}
GUIDANCE_SCALE:      ${KONTEXT_GUIDANCE}
SCHEDULER:           native model scheduler (dynamic shifting)
SEED:                ${SEED}
MODEL_OUTPUT:        ${MODEL_OUTPUT}
FLAT_DIR:            ${MODEL_OUTPUT}/GEdit_v2
CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES}
NUM_GPUS:            ${NUM_GPUS}
MAX_SAMPLES:         ${MAX_SAMPLES:-all}
EOF

cmd=(
  python "${EVAL_DIR}/run_editflow_gedit_infer.py"
  --role kontext
  --meta_json "${META_JSON}"
  --output_dir "${MODEL_OUTPUT}"
  --model_path "${KONTEXT_MODEL_PATH}"
  --run_name flux1_kontext_dev
  --num_inference_steps "${KONTEXT_STEPS}"
  --guidance_scale "${KONTEXT_GUIDANCE}"
  --seed "${SEED}"
  --num_gpus "${NUM_GPUS}"
  --skip_existing
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=(--max_samples "${MAX_SAMPLES}")
fi

cd "${EDITFLOW_DIR}"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
echo "Done: ${RUN_ROOT}"
