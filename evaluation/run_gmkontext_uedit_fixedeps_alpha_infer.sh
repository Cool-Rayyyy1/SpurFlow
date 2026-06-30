#!/usr/bin/env bash
# ImgEdit-Bench student inference for gmkontext_uedit_fixedeps_alpha (4-head + proj_out_alpha).
#
# Model: ArcFluxEditNewAlphaTransformer2DModel + fixed-path epsilon training:
#   pred_delta ~ x0_tgt - x_ref
#   student_u = path_epsilon - alpha * x_ref - pred_delta
# Preprocessing: STUDENT_RESIZE_MODE=kontext (nearest FLUX Kontext bucket, matches training).
# Default ckpt dir: checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_alpha/.../iter_5000.pth
#
# Do NOT use run_gmkontext_uedit_fixedeps_infer.sh — that script targets the 3-head model
# without proj_out_alpha (editflux_uedit_fixedeps_2nfe_k16_data.py).
#
# Extra outputs (NOT used for GPT scoring):
#   student/alpha_vis/basic/<id>_alpha_step{1,2}.png  (blank grid + alpha labels)
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_alpha_infer.sh
#   CKPT=/path/to/iter_5000.pth bash evaluation/run_gmkontext_uedit_fixedeps_alpha_infer.sh
#   SUITE=basic MAX_SAMPLES=8 bash evaluation/run_gmkontext_uedit_fixedeps_alpha_infer.sh
#   GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_alpha_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_alpha/iter_5000.pth}"
export RUN_TAG="${RUN_TAG:-gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_iter_5000}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export NUM_PROCESSES="${NUM_PROCESSES:-16}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores.txt}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_alpha_infer.sh" "$@"
