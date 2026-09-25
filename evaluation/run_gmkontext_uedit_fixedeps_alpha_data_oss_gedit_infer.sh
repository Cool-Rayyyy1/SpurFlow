#!/usr/bin/env bash
# GEdit-v2 student inference for train_flux_edit_fixedeps_alpha_data.sh
# (Kontext sigmoid-alpha PIID, no GAN, oss 30% / pico 70%).
# Generation only — no GPT / VIEScore.
#
# Must match train:
#   CONFIG  = configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data_oss_pico.py
#   student = cond-only 2-NFE, guidance=3.5, STUDENT_RESIZE_MODE=kontext
#   shift   = 3.2 (already in the config)
#
# Default ckpt:
#   checkpoints/gmkontext_uedit_fixedeps_alpha_k16_2nfe_shift3.2_oss30_pico70/20260819_145946/iter_25000.pth
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_alpha_data_oss_gedit_infer.sh
#   CKPT=/path/to/iter_XXXX.pth RUN_TAG=... bash evaluation/run_gmkontext_uedit_fixedeps_alpha_data_oss_gedit_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="${EDITFLOW_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17}"
export EDITFLOW_DIR
export CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
export CONDA_ENV="${CONDA_ENV:-arcflow}"
export HF_HOME="${HF_HOME:-/mnt/afs_gaochengmin/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_alpha_k16_2nfe_shift3.2_oss30_pico70}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data_oss_pico.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/20260819_145946/iter_25000.pth}"
export RUN_TAG="${RUN_TAG:-${RUN_NAME}_20260819_145946_iter25000}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export META_JSON="${META_JSON:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
export REQUIRE_ALPHA=1

mkdir -p "${HF_HOME}/hub"
mkdir -p /mnt/afs_gaochengmin/.cache/torch/hub/checkpoints

echo "[gmkontext_uedit_fixedeps_alpha_data_oss / GEdit-v2] CONFIG=${CONFIG}"
echo "[gmkontext_uedit_fixedeps_alpha_data_oss / GEdit-v2] CKPT=${CKPT}"
echo "[gmkontext_uedit_fixedeps_alpha_data_oss / GEdit-v2] RUN_TAG=${RUN_TAG}"
echo "[gmkontext_uedit_fixedeps_alpha_data_oss / GEdit-v2] NFE=${STUDENT_NFE}  guidance=${STUDENT_GUIDANCE}  resize=${STUDENT_RESIZE_MODE}  require_alpha=1"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_gedit_infer.sh" "$@"
