#!/usr/bin/env bash
# ImgEdit-Bench student inference for gmkontext_uedit_fixedeps split-stage (3-head, NO proj_out_epsilon).
#
# Model: ArcFlowEditImitationSplitStage + fixed-path epsilon training (same student velocity as fixedeps):
#   pred_delta ~ x0_tgt - x_ref
#   student_u = path_epsilon - x_ref - pred_delta
# Training uses a 2-step rollout with split-stage losses; inference is 2-NFE val_step with
# step-2 policy x_ref scaled by split_stage_step2_x_ref_scale (default 0.5, matches train).
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
#   RUN_TAG=my_custom_tag bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_infer.sh
#
# Output: evaluation/imgedit_bench/outputs/runs/<RUN_TAG>/
# Default RUN_TAG includes step2_xref0p5 so re-infer after the val_step fix does not
# overwrite the earlier run (e.g. ..._split_stage_iter_8500 without step2 scaling).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_split_stage.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage/model/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage/iter_8500.pth}"
export RUN_TAG="${RUN_TAG:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_iter_8500_step2xref0p5}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores_gpt4o}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores_gpt4o.txt}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
