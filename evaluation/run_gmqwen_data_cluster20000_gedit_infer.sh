#!/usr/bin/env bash
# GEdit-v2 pi-stats inference for the VANILLA ArcFlow Qwen student
# (train_qwen_data.sh, no uedit / fixedeps / alpha / GAN), iter_20000.
#
# Mirrors the alpha-model cluster runs: dumps per-NFE-step mixture pi
# (mixture_stats/{stem}.json) and auto-computes dominant_k-vs-edit_type ARI.
# Soft metrics (kNN k=20 / k-means on mean_weights):
#   python evaluation/gedit_v2/compute_pi_ari.py --soft --output_dir <student dir>
#
# Usage:
#   bash evaluation/run_gmqwen_data_cluster20000_gedit_infer.sh
#   NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash evaluation/run_gmqwen_data_cluster20000_gedit_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITFLOW_DIR="/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17/EditFlow"

export CKPT="${CKPT:-${EDITFLOW_DIR}/checkpoints/gmqwen_k16_2nfe_oss30_pico70_data/20260817_111001/iter_20000.pth}"
export RUN_TAG="${RUN_TAG:-gmqwen_k16_2nfe_oss30_pico70_data_cluster_20000}"
export DUMP_MIXTURE_STATS="${DUMP_MIXTURE_STATS:-1}"

exec bash "${SCRIPT_DIR}/run_gmqwen_data_gedit_infer.sh" "$@"
