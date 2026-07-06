#!/usr/bin/env bash
# ImgEdit-Bench inference for split_stage_dual_lora_dino_gan.
# Uses ArcFlowEditImitationSplitStageDualLoraGAN split forward_test (step1/step2 LoRA).
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_dino_gan_infer.sh
#   CKPT=/path/to/iter_5000.pth GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_dino_gan_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_dual_lora_dino_gan}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_split_stage_dual_lora_dino_gan.py}"
export CKPT="${CKPT:-}"
export RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT:-latest}" .pth)}"
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
