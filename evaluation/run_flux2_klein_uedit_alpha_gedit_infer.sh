#!/usr/bin/env bash
# GEdit-v2 inference for the trained FLUX.2-klein-base Softsign-01 alpha 2-NFE student.
# Uses KLEIN_MODEL_PATH (not Kontext). GEdit meta stays on afs_caiqi.
#
# Training counterpart:
#   train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh
#
# Usage:
#   bash evaluation/run_flux2_klein_uedit_alpha_gedit_infer.sh
#   CKPT=/path/to/iter_20000.pth bash evaluation/run_flux2_klein_uedit_alpha_gedit_infer.sh
#   MAX_SAMPLES=8 bash evaluation/run_flux2_klein_uedit_alpha_gedit_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="${EDITFLOW_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-${EDITFLOW_DIR}}"
export EDITFLOW_DIR
export CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
export CONDA_ENV="${CONDA_ENV:-arcflow}"
export HF_HOME="${HF_HOME:-/mnt/afs_gaochengmin/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export RUN_NAME="${RUN_NAME:-gmklein_base_uedit_fixedeps_alpha_softsign01_k16_2nfe_shift3.2_teachercfg4.0_oss70_pico30}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/flux2_klein/editflux2_klein_uedit_fixedeps_2nfe_k16_alpha_softsign_data_oss_pico.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/20260818_033917/iter_20000.pth}"
export RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT}" .pth)_gedit}"

export KLEIN_MODEL_PATH="${KLEIN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-flux2}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-1.0}"
export REQUIRE_ALPHA="${REQUIRE_ALPHA:-1}"
export META_JSON="${META_JSON:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"

mkdir -p "${HF_HOME}/hub"
mkdir -p /mnt/afs_gaochengmin/.cache/torch/hub/checkpoints

echo "[gmklein_alpha / GEdit-v2] CONFIG=${CONFIG}"
echo "[gmklein_alpha / GEdit-v2] CKPT=${CKPT}"
echo "[gmklein_alpha / GEdit-v2] KLEIN_MODEL_PATH=${KLEIN_MODEL_PATH}"
echo "[gmklein_alpha / GEdit-v2] RUN_TAG=${RUN_TAG}"
echo "[gmklein_alpha / GEdit-v2] NFE=${STUDENT_NFE}  resize=${STUDENT_RESIZE_MODE}  require_alpha=${REQUIRE_ALPHA}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_gedit_infer.sh" "$@"
