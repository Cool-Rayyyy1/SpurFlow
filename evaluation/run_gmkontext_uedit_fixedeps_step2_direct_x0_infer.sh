#!/usr/bin/env bash
# ImgEdit-Bench student inference: fixedeps 2-NFE with step-2 direct x0.
#
# Differs from run_gmkontext_uedit_fixedeps_infer.sh only in the final step:
#   step-1: same as usual  v=(eps - x_ref - pred_delta), integrate x <- x - v dt
#   step-2: pred on x_t -> pred_delta -> x0 = x_ref + pred_delta  (NO integrate)
#
# Uses config:
#   configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_step2_direct_x0.py
# which sets test_cfg.step2_direct_x0=True. Other infer scripts are untouched.
#
# Default: 1 GPU.
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_step2_direct_x0_infer.sh
#   CKPT=/path/to/iter_xxx.pth bash evaluation/run_gmkontext_uedit_fixedeps_step2_direct_x0_infer.sh
#   SUITE=basic MAX_SAMPLES=8 GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_step2_direct_x0_infer.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash evaluation/run_gmkontext_uedit_fixedeps_step2_direct_x0_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_dino_gan}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_step2_direct_x0.py}"
# Same default ckpt family as run_gmkontext_uedit_fixedeps_infer.sh (student EMA).
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/model/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_dino_gan/model/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_dino_gan/iter_5000.pth}"
export RUN_TAG="${RUN_TAG:-gmkontext_uedit_fixedeps_step2_direct_x0_$(basename "${CKPT}" .pth)}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"

# Default single GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export NUM_PROCESSES="${NUM_PROCESSES:-16}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores_gpt4o}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores_gpt4o.txt}"
# Skip comparison grids by default (same as user's recent preference).
export BUILD_COMPARISONS="${BUILD_COMPARISONS:-0}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
