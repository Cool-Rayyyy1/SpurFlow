#!/usr/bin/env bash
# ImgEdit-Bench (gen + gpt-4o score) for the FLUX.2-klein-base Softsign-01
# alpha step-2 DINO GAN student trained on oss30 / pico70.
#
# Training counterpart:
#   train_flux2_klein_edit_fixedeps_alpha_softsign_step2_alph_dino_gan_plus_oss.sh
#   (NFE=2, SHIFT=3.2, TEACHER_CFG=4.0, OSS_PROB=0.3, student CFG=1.0,
#    resize_mode=flux2)
#
# Must match train:
#   CONFIG  = editflux2_klein_uedit_fixedeps_2nfe_k16_alpha_softsign_step2_alph_dino_gan_oss_pico.py
#   student = 2-NFE, guidance=1.0, STUDENT_RESIZE_MODE=flux2
#   teacher CFG is train-only; do not pass 4.0 at student infer.
#
# Usage (1 GPU):
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
#     bash evaluation/run_flux2_klein_uedit_step2_alph_dino_gan_oss_pico_infer.sh
#
#   SUITE=basic MAX_SAMPLES=8 GEN_ONLY=1 \
#     bash evaluation/run_flux2_klein_uedit_step2_alph_dino_gan_oss_pico_infer.sh
#   SCORE_ONLY=1 \
#     bash evaluation/run_flux2_klein_uedit_step2_alph_dino_gan_oss_pico_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="${EDITFLOW_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export RUN_NAME="${RUN_NAME:-gmklein_base_uedit_fixedeps_alpha_softsign01_k16_2nfe_shift3.2_teachercfg4.0_oss30_pico70_step2_alph_dino_gan}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/flux2_klein/editflux2_klein_uedit_fixedeps_2nfe_k16_alpha_softsign_step2_alph_dino_gan_oss_pico.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/20260825_001651/iter_6000.pth}"
export RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT}" .pth)}"

export KLEIN_MODEL_PATH="${KLEIN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-flux2}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-1.0}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"

# This box is 1x H100; override NUM_GPUS if you have more.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

echo "[gmklein step2 alph_dino_gan oss30/pico70 / ImgEdit]"
echo "  CONFIG=${CONFIG}"
echo "  CKPT=${CKPT}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  NFE=${STUDENT_NFE}  guidance=${STUDENT_GUIDANCE}  resize=${STUDENT_RESIZE_MODE}"
echo "  score=${OPENAI_SCORING_MODEL}  gpus=${NUM_GPUS}  cuda=${CUDA_VISIBLE_DEVICES}"

exec bash "${SCRIPT_DIR}/run_flux2_klein_uedit_alpha_infer.sh" "$@"
