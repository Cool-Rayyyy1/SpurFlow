#!/usr/bin/env bash
# GEdit-v2 student inference for train_flux_data.sh (vanilla ArcFlow, no alpha).
# Local paths under /mnt/afs_gaochengmin; GEdit meta stays on afs_caiqi.
#
# Default ckpt: checkpoints/20260817_110510/latest.pth
#
# Usage:
#   bash evaluation/run_gmkontext_data_gedit_infer.sh
#   CKPT=/path/to/iter_20000.pth RUN_TAG=... bash evaluation/run_gmkontext_data_gedit_infer.sh
#   MAX_SAMPLES=8 bash evaluation/run_gmkontext_data_gedit_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17/EditFlow"

export WORKSPACE_ROOT="/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17"
export EDITFLOW_DIR
export CONDA_ROOT="/mnt/afs_gaochengmin/anaconda3"
export CONDA_ENV="arcflow"
export HF_HOME="/mnt/afs_gaochengmin/.cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export RUN_NAME="${RUN_NAME:-gmkontext_k16_2nfe_oss30_pico70_data}"
export CONFIG="${CONFIG:-${EDITFLOW_DIR}/configs/kontext/editflux_2nfe_k16_data_oss_pico.py}"
export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/20260817_110510/latest.pth}"
export RUN_TAG="${RUN_TAG:-gmkontext_k16_2nfe_oss30_pico70_data_20260817_110510}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export STUDENT_NFE="${STUDENT_NFE:-2}"
export STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-3.5}"
export META_JSON="${META_JSON:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev}"
export REQUIRE_ALPHA=0

mkdir -p "${HF_HOME}/hub"
mkdir -p /mnt/afs_gaochengmin/.cache/torch/hub/checkpoints

echo "[gmkontext_data / GEdit-v2] CONFIG=${CONFIG}"
echo "[gmkontext_data / GEdit-v2] CKPT=${CKPT}"
echo "[gmkontext_data / GEdit-v2] RUN_TAG=${RUN_TAG}"
echo "[gmkontext_data / GEdit-v2] NFE=${STUDENT_NFE}  guidance=${STUDENT_GUIDANCE}  resize=${STUDENT_RESIZE_MODE}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_gedit_infer.sh" "$@"
