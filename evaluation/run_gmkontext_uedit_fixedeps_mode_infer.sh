#!/usr/bin/env bash
# ImgEdit-Bench student inference for step2_dino_gan with GM mixture MODE reduce.
#
# Same checkpoint / config as run_gmkontext_uedit_fixedeps_infer.sh (the run that
# scored Basic overall 3.62 at iter_5500), but at each NFE step:
#   k* = argmax_k w_k
#   pred_delta = δ_{k*}   (no weighted average over K=16)
#
# Default ckpt: step2_dino_gan iter_5500
# Output tag:  ..._iter_5500_mode  (does not overwrite the mean-reduce run)
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_mode_infer.sh
#   SUITE=basic MAX_SAMPLES=8 GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_mode_infer.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash evaluation/run_gmkontext_uedit_fixedeps_mode_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_dino_gan}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/model/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_dino_gan/iter_5500.pth}"
export RUN_TAG="${RUN_TAG:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_dino_gan_iter_5500_mode}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores_gpt4o}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores_gpt4o.txt}"
export BUILD_COMPARISONS="${BUILD_COMPARISONS:-0}"
export WRITE_CASE_BUNDLES="${WRITE_CASE_BUNDLES:-1}"
# Argmax over mixture weights instead of mean.
export STUDENT_MIXTURE_REDUCE="${STUDENT_MIXTURE_REDUCE:-mode}"
# Dump per-image K=16 weight / mode-histogram files under student/mixture_stats/.
export DUMP_MIXTURE_STATS="${DUMP_MIXTURE_STATS:-1}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
