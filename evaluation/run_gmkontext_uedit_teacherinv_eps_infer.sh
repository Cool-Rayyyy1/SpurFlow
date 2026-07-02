#!/usr/bin/env bash
# ImgEdit-Bench student inference for gmkontext_uedit_teacherinv_eps (4-head + proj_out_epsilon).
#
# Model: ArcFluxEditNewEpsTransformer2DModel + ArcFlowEditTeacherInvEpsImitation
#   (train_flux_edit_teacherinv_eps_data.sh):
#   pred_epsilon from epsilon head at t=1 (fixed trajectory)
#   pred_delta ~ x0_tgt - x_ref
#   student_u = pred_epsilon - x_ref - pred_delta
# Preprocessing: STUDENT_RESIZE_MODE=kontext (matches train_flux_edit_teacherinv_eps_data.sh).
# Default ckpt: checkpoints/gmkontext_uedit_teacherinv_eps_k16_2nfe_pico400k/20260624_174422/iter_20000.pth
#
# Do NOT use run_gmkontext_uedit_fixedeps_infer.sh for this checkpoint — that script targets
# editflux_uedit_fixedeps_2nfe_k16_data.py (3-head, NO proj_out_epsilon, fixed path epsilon).
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_teacherinv_eps_infer.sh
#   CKPT=/path/to/iter_20000.pth bash evaluation/run_gmkontext_uedit_teacherinv_eps_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_teacherinv_eps_k16_2nfe_pico400k}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_teacherinv_eps_2nfe_k16_data.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_teacherinv_eps_k16_2nfe_pico400k/20260624_174422/iter_20000.pth}"
export RUN_TAG="${RUN_TAG:-gmkontext_uedit_teacherinv_eps_k16_2nfe_pico400k_20260624_iter_20000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"
# Match training ImageEdit(resize_mode='kontext'): bucket by source aspect ratio, bicubic resize.
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
# 2-NFE student + guidance 3.5 (matches test_cfg distilled_guidance_scale). Keep scoring concurrency
# modest to avoid GPT API 429s.
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores_gpt4o}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores_gpt4o.txt}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
