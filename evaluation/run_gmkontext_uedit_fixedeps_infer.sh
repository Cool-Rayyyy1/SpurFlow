#!/usr/bin/env bash
# ImgEdit-Bench student inference for gmkontext_uedit_fixedeps (3-head, NO proj_out_epsilon).
#
# Model: ArcFluxEditNewTransformer2DModel + fixed-path epsilon training:
#   pred_delta ~ x0_tgt - x_ref
#   student_u = path_epsilon - x_ref - pred_delta
# Preprocessing: STUDENT_RESIZE_MODE=kontext (matches train_flux_edit_fixedeps_data.sh).
# Default ckpt: checkpoints/.../20260618_055623/iter_38500.pth (latest.pth -> iter_38500)
#
# Do NOT use run_gmkontext_uedit_infer.sh for this checkpoint — that script defaults to
# editflux_uedit_2nfe_k16_data.py (ArcFluxEditTransformer2DModel with proj_out_epsilon).
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_infer.sh
#   CKPT=/path/to/iter_6000.pth bash evaluation/run_gmkontext_uedit_fixedeps_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data.py}"
# NOTE: current code implements the epsilon-delta residual velocity
#   student_u = path_epsilon - x_ref - pred_delta   (== teacher u = noise - x0)
# The 20260618_055623 run was trained with this same formulation, so train/inference match.
# Do NOT point this at the older 20260614_015130 run: that checkpoint was trained with the
# previous (delta-epsilon) sign convention and is INCONSISTENT with the current inference code.
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k/20260618_055623/iter_38500.pth}"
export RUN_TAG="${RUN_TAG:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_20260618_iter_38500}"
# Match training ImageEdit(resize_mode='kontext'): bucket by source aspect ratio, bicubic resize.
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
# Inference GPUs (student generation): 2 processes via mp.spawn, one model per GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
# 2-NFE student + guidance 3.5 (matches test_cfg distilled_guidance_scale). Keep scoring concurrency
# modest to avoid GPT API 429s.
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
# GPT scoring: gpt-4o into a separate folder so older gpt-4o-2024-11-20 scores stay intact.
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores_gpt4o}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores_gpt4o.txt}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
