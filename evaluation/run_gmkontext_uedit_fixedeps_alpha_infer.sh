#!/usr/bin/env bash
# ImgEdit-Bench student inference for gmkontext_uedit_fixedeps_alpha (4-head + proj_out_alpha).
#
# One-pass generation + alpha v6 visualization (no second model forward for vis).
#
# Model: ArcFluxEditNewAlphaTransformer2DModel + fixed-path epsilon:
#   pred_delta ~ x0_tgt - x_ref
#   alpha = sigmoid(raw head), four channels per 2x2 patch
#   student_u = path_epsilon - alpha * x_ref - pred_delta
# Preprocessing: STUDENT_RESIZE_MODE=kontext
#
# Default ckpt: step2 alph_dino_gan_new iter_7000 under checkpoints/model/final/...
#
# Outputs:
#   student/basic/{Action,Add,...}/{key}/
#     src.png  edit.png  pred.png  prompt.txt
#     step{1,2}_{alpha,heatmap,overlay}.png   # alpha v6
#   (NO flat student/basic/{key}.png duplicates)
#   student/uge/{key}.png                     # UGE still flat
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_alpha_infer.sh
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash evaluation/run_gmkontext_uedit_fixedeps_alpha_infer.sh
#   CKPT=.../iter_8000.pth RUN_TAG=..._iter_8000 bash evaluation/run_gmkontext_uedit_fixedeps_alpha_infer.sh
#   GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_alpha_infer.sh
#   SCORE_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_alpha_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

DEFAULT_CKPT_DIR="${EDITFLOW_DIR}/checkpoints/model/final/gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_step2_alph_dino_gan_new"
DEFAULT_CKPT="${DEFAULT_CKPT_DIR}/iter_7000.pth"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_step2_alph_dino_gan}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data.py}"
export CKPT="${CKPT:-${DEFAULT_CKPT}}"
export RUN_TAG="${RUN_TAG:-step2_alph_dino_gan_new_iter_7000}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export NUM_PROCESSES="${NUM_PROCESSES:-16}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores.txt}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_alpha_infer.sh" "$@"
