#!/usr/bin/env bash
# ImgEdit-Bench (full) + GPT scoring for dual-LoRA teacher-x0.
#
# Matches train_flux_edit_fixedeps_data_split_stage_dual_lora_teacher_x0.sh:
#   ArcFlowEditImitationSplitStageDualLoraTeacherX0 + dual_stage_lora
#   step-1: integrate to mid (step1 LoRA)
#   step-2: x0 = x_ref + pred_delta (step2 LoRA; no velocity integrate)
#   NFE=2, distilled_guidance=3.5, resize_mode=kontext
#
# Default ckpt:
#   checkpoints/.../20260721_021700/latest.pth  (-> iter_20000.pth)
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh
#   CKPT=.../iter_15000.pth bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh
#   GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh
#   SCORE_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh
#   SUITE=basic MAX_SAMPLES=8 bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh
#
# Do NOT use run_gmkontext_uedit_fixedeps_infer.sh for these checkpoints —
# that loads the non-dual / non-teacher_x0 config.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

CKPT_DIR="${EDITFLOW_DIR}/checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_dual_lora_teacher_x0/20260721_021700"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_dual_lora_teacher_x0}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_split_stage_dual_lora_teacher_x0_infer.py}"
export CKPT="${CKPT:-${CKPT_DIR}/latest.pth}"
export RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT}" .pth)}"

# Match training sample_eval / test_cfg.
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"

# Full ImgEdit-Bench by default (basic + uge + multiturn).
export SUITE="${SUITE:-all}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
export NUM_PROCESSES="${NUM_PROCESSES:-16}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export SCORES_SUBDIR="${SCORES_SUBDIR:-scores_gpt4o}"
export SCORES_TXT_NAME="${SCORES_TXT_NAME:-scores_gpt4o.txt}"
export BUILD_COMPARISONS="${BUILD_COMPARISONS:-0}"
# Category / case folders with src+pred (same layout as alpha infer).
export WRITE_CASE_BUNDLES="${WRITE_CASE_BUNDLES:-1}"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  echo "  Set CKPT=/path/to/iter_XXXXX.pth" >&2
  exit 1
fi

echo "[dual_lora_teacher_x0] CONFIG=${CONFIG}"
echo "[dual_lora_teacher_x0] CKPT=${CKPT}"
echo "[dual_lora_teacher_x0] SUITE=${SUITE}  NFE=${STUDENT_NFE}  guidance=${STUDENT_GUIDANCE}  resize=${STUDENT_RESIZE_MODE}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
