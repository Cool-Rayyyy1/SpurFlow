#!/usr/bin/env bash
# ImgEdit-Bench student inference for gmkontext_uedit_fixedeps split-stage (3-head, NO proj_out_epsilon).
#
# Model: ArcFlowEditImitationSplitStage + fixed-path epsilon training (same student velocity as fixedeps):
#   pred_delta ~ x0_tgt - x_ref
#   student_u = path_epsilon - x_ref - pred_delta
# Training uses a 2-step rollout with split-stage losses; inference is standard 2-NFE val_step.
# Preprocessing: STUDENT_RESIZE_MODE=kontext (matches train_flux_edit_fixedeps_data_split_stage.sh).
# Default ckpt: checkpoints/.../split_stage/model/.../iter_8500.pth
#
# Do NOT use run_gmkontext_uedit_fixedeps_infer.sh — that script targets the non-split-stage config
# (editflux_uedit_fixedeps_2nfe_k16_data.py / ArcFlowEditImitation).
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_infer.sh
#   CKPT=/path/to/iter_8500.pth bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_infer.sh
#   SUITE=basic MAX_SAMPLES=8 bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_infer.sh
#   GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_split_stage.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage/model/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage/iter_8500.pth}"
export RUN_TAG="${RUN_TAG:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_iter_8500}"
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
