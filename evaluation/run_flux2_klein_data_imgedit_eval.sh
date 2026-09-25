#!/usr/bin/env bash
# ImgEdit-Bench for train_flux2_klein_data.sh (vanilla ArcFlow, no uedit/alpha/GAN).
#
# Must match train:
#   CONFIG  = configs/flux2_klein/editflux2_klein_2nfe_k16_data_oss_pico.py
#   student = true 2-NFE, guidance=1.0 (cond-only), STUDENT_RESIZE_MODE=flux2
#             (Flux2KleinPipeline area-cap ~1MP + multiple-of-16, same as ImageEdit)
#   teacher CFG=4 is train-only; do not pass 4.0 at student infer.
#   gpt-4o scoring via evaluation/imgedit_bench/openai.env
#
# Default ckpt: iter_18000 of
#   gmklein_base_k16_2nfe_shift3.0_teachercfg4.0_oss30_pico70_data / 20260824_234013
#
# Usage:
#   bash evaluation/run_flux2_klein_data_imgedit_eval.sh
#   CKPT=/path/to/iter_XXXX.pth bash evaluation/run_flux2_klein_data_imgedit_eval.sh
#   SUITE=basic MAX_SAMPLES=8 GEN_ONLY=1 \
#     bash evaluation/run_flux2_klein_data_imgedit_eval.sh
#   SCORE_ONLY=1 RUN_TAG=<existing-run-tag> \
#     bash evaluation/run_flux2_klein_data_imgedit_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="${EDITFLOW_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17}"
export EDITFLOW_DIR
export EVAL_DIR="${EDITFLOW_DIR}/evaluation/imgedit_bench"
export DATA_ROOT="${DATA_ROOT:-/mnt/afs_gaochengmin/data/imgedit}"
export CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
export CONDA_ENV="${CONDA_ENV:-arcflow}"
export IMGEDIT_BENCH_ROOT="${IMGEDIT_BENCH_ROOT:-/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark}"
export HF_HOME="${HF_HOME:-/mnt/afs_gaochengmin/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export KLEIN_MODEL_PATH="${KLEIN_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"
# Shared student infer entry reads --model_path from KONTEXT_MODEL_PATH; unused
# for role=student (config+ckpt load the Klein student). Keep it local/offline.
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-${KLEIN_MODEL_PATH}}"

export RUN_NAME="${RUN_NAME:-gmklein_base_k16_2nfe_shift3.0_teachercfg4.0_oss30_pico70_data}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/flux2_klein/editflux2_klein_2nfe_k16_data_oss_pico.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/${RUN_NAME}/20260824_234013/iter_18000.pth}"
export RUN_TAG="${RUN_TAG:-${RUN_NAME}_$(basename "${CKPT}" .pth)}"

# Train/infer alignment: flux2 resize, 2-NFE, student CFG=1.
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-flux2}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-1.0}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"
export NUM_PROCESSES="${NUM_PROCESSES:-16}"
export GEN_ONLY="${GEN_ONLY:-0}"
export BUILD_COMPARISONS="${BUILD_COMPARISONS:-0}"
# No alpha/heatmap vis, no per-case src/pred packs. Scoring reads flat basic/{key}.png.
export WRITE_CASE_BUNDLES="${WRITE_CASE_BUNDLES:-0}"
export DUMP_MIXTURE_STATS="${DUMP_MIXTURE_STATS:-0}"
export REF_TEACHER_OUTPUT="${REF_TEACHER_OUTPUT:-}"

mkdir -p "${HF_HOME}/hub"
mkdir -p /mnt/afs_gaochengmin/.cache/torch/hub/checkpoints

echo "[gmklein vanilla-ArcFlow data / ImgEdit]"
echo "  CONFIG=${CONFIG}"
echo "  CKPT=${CKPT}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  NFE=${STUDENT_NFE}  guidance=${STUDENT_GUIDANCE}  resize=${STUDENT_RESIZE_MODE}"
echo "  score=${OPENAI_SCORING_MODEL}  gpus=${NUM_GPUS}  cuda=${CUDA_VISIBLE_DEVICES}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
