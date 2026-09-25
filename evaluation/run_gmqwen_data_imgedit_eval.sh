#!/usr/bin/env bash
# ImgEdit-Bench for train_qwen_data.sh (vanilla ArcFlow, no alpha).
# All local paths under /mnt/afs_gaochengmin.
#
# Default ckpt: checkpoints/gmqwen_k16_2nfe_oss30_pico70_data/20260817_111001/latest.pth
#
# Usage (generation only by default; no GPT scoring):
#   bash evaluation/run_gmqwen_data_imgedit_eval.sh
#   SUITE=basic MAX_SAMPLES=8 bash evaluation/run_gmqwen_data_imgedit_eval.sh
#   GEN_ONLY=0 bash evaluation/run_gmqwen_data_imgedit_eval.sh   # also score

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17/EditFlow"

export WORKSPACE_ROOT="/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17"
export EDITFLOW_DIR
export EVAL_DIR="${EDITFLOW_DIR}/evaluation/imgedit_bench"
export DATA_ROOT="/mnt/afs_gaochengmin/data/imgedit"
export CONDA_ROOT="/mnt/afs_gaochengmin/anaconda3"
export CONDA_ENV="arcflow"
export IMGEDIT_BENCH_ROOT="/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark"
export HF_HOME="/mnt/afs_gaochengmin/.cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export QWEN_MODEL_PATH="/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511"
export KONTEXT_MODEL_PATH="${QWEN_MODEL_PATH}"

export RUN_NAME="${RUN_NAME:-gmqwen_k16_2nfe_oss30_pico70_data}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/qwen/editqwen_2nfe_k16_data_oss_pico.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmqwen_k16_2nfe_oss30_pico70_data/20260817_111001/latest.pth}"
export RUN_TAG="${RUN_TAG:-gmqwen_k16_2nfe_oss30_pico70_data_20260817_111001}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-qwen}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-1.0}"
export NUM_PROCESSES="${NUM_PROCESSES:-32}"
export GEN_ONLY="${GEN_ONLY:-1}"
export SUITE="${SUITE:-basic}"
export DUMP_MIXTURE_STATS="${DUMP_MIXTURE_STATS:-1}"
export BUILD_COMPARISONS="${BUILD_COMPARISONS:-0}"
export WRITE_CASE_BUNDLES="${WRITE_CASE_BUNDLES:-1}"
export OPENAI_SCORING_MODEL="${OPENAI_SCORING_MODEL:-gpt-4o}"
export REF_TEACHER_OUTPUT="${REF_TEACHER_OUTPUT:-}"

mkdir -p "${HF_HOME}/hub"
mkdir -p /mnt/afs_gaochengmin/.cache/torch/hub/checkpoints

echo "[gmqwen_data / ImgEdit] CONFIG=${CONFIG}"
echo "[gmqwen_data / ImgEdit] CKPT=${CKPT}"
echo "[gmqwen_data / ImgEdit] RUN_TAG=${RUN_TAG}"
echo "[gmqwen_data / ImgEdit] NFE=${STUDENT_NFE}  guidance=${STUDENT_GUIDANCE}  resize=${STUDENT_RESIZE_MODE}  GEN_ONLY=${GEN_ONLY}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_infer.sh" "$@"
