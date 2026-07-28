#!/usr/bin/env bash
# ImgEdit-Bench student inference for gmkontext_uedit_fixedeps (3-head, NO proj_out_epsilon).
#
# Model: ArcFluxEditNewTransformer2DModel + fixed-path epsilon training:
#   pred_delta ~ x0_tgt - x_ref
#   student_u = path_epsilon - x_ref - pred_delta
# Preprocessing: STUDENT_RESIZE_MODE=kontext (matches train_flux_edit_fixedeps_data.sh).
#
# Default ckpt: step2_dino_gan iter_5000 (student EMA only at inference; DINO D is unused).
# Uses editflux_uedit_fixedeps_2nfe_k16_data.py (ArcFlowEditImitation standard 2-NFE val_step),
# which matches step2_dino_gan training validation (Step2GAN val_step -> same 2-NFE forward_test).
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_infer.sh
#   CKPT=/path/to/iter_6000.pth bash evaluation/run_gmkontext_uedit_fixedeps_infer.sh
#   SUITE=basic MAX_SAMPLES=8 GEN_ONLY=1 bash evaluation/run_gmkontext_uedit_fixedeps_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"

export RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_dino_gan}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data.py}"
# step2_dino_gan ckpt includes discriminator keys; init_model loads diffusion_ema only (strict=False).
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/model/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_dino_gan/model/gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_dino_gan/iter_5000.pth}"
export RUN_TAG="${RUN_TAG:-gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_dino_gan_iter_5000}"
# Match training ImageEdit(resize_mode='kontext'): bucket by source aspect ratio, bicubic resize.
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
# Inference: 8 GPUs by default. CUDA_VISIBLE_DEVICES must expose at least NUM_GPUS devices.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
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
