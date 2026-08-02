#!/usr/bin/env bash
# ImgEdit-Bench student inference for pure ArcFlow (train_flux_data.sh).
#
# Matches:
#   train_flux_data.sh
#   configs/kontext/editflux_2nfe_k16_data.py
#     LatentDiffusionImageEdit
#     + ArcFlowImitation (policy_type='ArcFlow')
#     + ArcFluxTransformer2DModel (proj_out_means / logweights / loggamma)
#   ImageEdit resize_mode default: center_crop (1024 square)
#   NFE=2, distilled_guidance=3.5
#
# No alpha head / no alpha visualization — flat student/{basic,uge}/ outputs + GPT score.
#
# Default ckpt (this machine's train_flux_data run):
#   checkpoints/20260728_220103/latest.pth  (-> iter_10000.pth)
# Note: checkpoint_config.out_dir='checkpoints/' saves under
#   EditFlow/checkpoints/<run_id>/ rather than checkpoints/<RUN_NAME>/<run_id>/.
#
# Usage:
#   bash evaluation/run_gmkontext_data_infer.sh
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash evaluation/run_gmkontext_data_infer.sh
#   CKPT=checkpoints/20260728_220103/iter_8000.pth \
#     RUN_TAG=gmkontext_k16_2nfe_pico400k_data_iter_8000 \
#     bash evaluation/run_gmkontext_data_infer.sh
#   GEN_ONLY=1 bash evaluation/run_gmkontext_data_infer.sh
#   SCORE_ONLY=1 bash evaluation/run_gmkontext_data_infer.sh
#   SUITE=basic MAX_SAMPLES=8 bash evaluation/run_gmkontext_data_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

DEFAULT_CKPT_DIR="${EDITFLOW_DIR}/checkpoints/20260728_220103"
DEFAULT_CKPT="${DEFAULT_CKPT_DIR}/latest.pth"

export RUN_NAME="${RUN_NAME:-gmkontext_k16_2nfe_pico400k_data}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_2nfe_k16_data.py}"
export CKPT="${CKPT:-${DEFAULT_CKPT}}"

# Resolve relative CKPT against EditFlow root; prefer absolute so shared
# resolve_ckpt does not fall back to an unrelated uedit checkpoint.
if [[ "${CKPT}" != /* ]]; then
  export CKPT="${EDITFLOW_DIR}/${CKPT}"
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  echo "  Expected e.g. ${DEFAULT_CKPT}" >&2
  echo "  Set CKPT=/path/to/iter_XXXX.pth" >&2
  exit 1
fi

CKPT_BASENAME="$(basename "$(readlink -f "${CKPT}")" .pth)"
export RUN_TAG="${RUN_TAG:-${RUN_NAME}_${CKPT_BASENAME}}"

# Match train_flux_data / ImageEdit defaults.
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-center_crop}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"

export SUITE="${SUITE:-all}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"
export NUM_PROCESSES="${NUM_PROCESSES:-16}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores.txt}"
export BUILD_COMPARISONS="${BUILD_COMPARISONS:-0}"
export WRITE_CASE_BUNDLES="${WRITE_CASE_BUNDLES:-0}"

echo "[gmkontext_data / pure ArcFlow] CONFIG=${CONFIG}"
echo "[gmkontext_data / pure ArcFlow] CKPT=${CKPT}"
echo "[gmkontext_data / pure ArcFlow] RUN_TAG=${RUN_TAG}"
echo "[gmkontext_data / pure ArcFlow] SUITE=${SUITE}  NFE=${STUDENT_NFE}  guidance=${STUDENT_GUIDANCE}  resize=${STUDENT_RESIZE_MODE}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
