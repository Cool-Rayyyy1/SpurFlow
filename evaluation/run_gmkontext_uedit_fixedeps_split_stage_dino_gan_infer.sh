#!/usr/bin/env bash
# ImgEdit-Bench student inference for split_stage_dino_gan (3-head fixed-eps + split-stage 2-NFE).
#
# Model: ArcFlowEditImitationSplitStageGAN + DINO GAN training; inference uses student EMA only.
# 2-NFE val_step with split_stage_step2_x_ref_scale (default 1.0 for this run).
# Preprocessing: STUDENT_RESIZE_MODE=kontext.
#
# Do NOT use run_gmkontext_uedit_fixedeps_infer.sh — that uses standard ArcFlowEditImitation
# (non split-stage forward_test).
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dino_gan_infer.sh
#   SUITE=basic MAX_SAMPLES=8 GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dino_gan_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_dino_gan}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_split_stage_dino_gan.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/model/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_dino_gan/iter_3000.pth}"
export RUN_TAG="${RUN_TAG:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_dino_gan_iter_3000_step2xref1p0}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores_gpt4o}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores_gpt4o.txt}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
