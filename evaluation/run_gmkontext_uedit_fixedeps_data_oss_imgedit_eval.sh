#!/usr/bin/env bash
# ImgEdit-Bench for train_flux_edit_fixedeps_data_oss.sh
# (Kontext residual PIID, no alpha, no GAN, oss 30% / pico 70%).
#
# Must match train:
#   CONFIG  = configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_oss_pico.py
#   student = cond-only 2-NFE, guidance=3.5, STUDENT_RESIZE_MODE=kontext
#   shift   = 3.2 (already in the config)
#
# Default ckpt:
#   checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_shift3.2_oss30_pico70/20260902_195514/latest.pth
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_data_oss_imgedit_eval.sh
#   GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_data_oss_imgedit_eval.sh
#   CKPT=/path/to/iter_XXXX.pth bash evaluation/run_gmkontext_uedit_fixedeps_data_oss_imgedit_eval.sh
#   SUITE=basic MAX_SAMPLES=8 bash evaluation/run_gmkontext_uedit_fixedeps_data_oss_imgedit_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="${EDITFLOW_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17}"
export EDITFLOW_DIR
export EVAL_DIR="${EDITFLOW_DIR}/evaluation/imgedit_bench"
export DATA_ROOT="${DATA_ROOT:-/mnt/afs_gaochengmin/data/imgedit}"
export CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
export CONDA_ENV="${CONDA_ENV:-arcflow}"
export IMGEDIT_BENCH_ROOT="${IMGEDIT_BENCH_ROOT:-/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark}"
export HF_HOME="${HF_HOME:-/mnt/afs_gaochengmin/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_shift3.2_oss30_pico70}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_oss_pico.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/20260902_195514/latest.pth}"
export RUN_TAG="${RUN_TAG:-${RUN_NAME}_20260902_195514}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export NUM_PROCESSES="${NUM_PROCESSES:-32}"
export BUILD_COMPARISONS="${BUILD_COMPARISONS:-0}"
export WRITE_CASE_BUNDLES="${WRITE_CASE_BUNDLES:-1}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export REF_TEACHER_OUTPUT="${REF_TEACHER_OUTPUT:-}"
export REQUIRE_ALPHA=0

mkdir -p "${HF_HOME}/hub"
mkdir -p /mnt/afs_gaochengmin/.cache/torch/hub/checkpoints

echo "[gmkontext_uedit_fixedeps_data_oss / ImgEdit] CONFIG=${CONFIG}"
echo "[gmkontext_uedit_fixedeps_data_oss / ImgEdit] CKPT=${CKPT}"
echo "[gmkontext_uedit_fixedeps_data_oss / ImgEdit] RUN_TAG=${RUN_TAG}"
echo "[gmkontext_uedit_fixedeps_data_oss / ImgEdit] NFE=${STUDENT_NFE}  guidance=${STUDENT_GUIDANCE}  resize=${STUDENT_RESIZE_MODE}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
