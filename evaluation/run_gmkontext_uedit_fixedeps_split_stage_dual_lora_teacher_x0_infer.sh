#!/usr/bin/env bash
# ImgEdit-Bench inference for split_stage_dual_lora_teacher_x0.
#
# Requires ArcFlowEditImitationSplitStageDualLoraTeacherX0 + dual_stage_lora:
#   step-1: integrate to mid (step1 LoRA)
#   step-2: x0 = x_ref + pred_delta (step2 LoRA; no velocity integrate)
#
# Output layout (WRITE_CASE_BUNDLES=1):
#   student/basic/{Action,Add,Adjust,Background,Compose,Extract,Remove,Replace,Style}/{key}/
#     src.png
#     pred.png
#     prompt.txt
#   student/basic/{key}.png   — flat copy of pred for GPT scoring
#   student/uge/{key}.png
#
# Do NOT use run_gmkontext_uedit_fixedeps_infer.sh for these checkpoints —
# that script loads the non-dual / non-teacher_x0 config.
#
# Default ckpt: 20260721_021700/iter_10000.pth (latest)
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh
#   CKPT=/path/to/iter_5000.pth bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh
#   SUITE=basic MAX_SAMPLES=8 GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_dual_lora_teacher_x0}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_split_stage_dual_lora_teacher_x0_infer.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_dual_lora_teacher_x0/20260721_021700/iter_10000.pth}"
export RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT}" .pth)}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export NUM_PROCESSES="${NUM_PROCESSES:-16}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores_gpt4o}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores_gpt4o.txt}"
export BUILD_COMPARISONS="${BUILD_COMPARISONS:-0}"
# Category / case folders with src+pred side-by-side (same layout as alpha infer).
export WRITE_CASE_BUNDLES="${WRITE_CASE_BUNDLES:-1}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
