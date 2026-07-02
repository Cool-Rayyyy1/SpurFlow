#!/usr/bin/env bash
# Alpha labels v3: 64x64 px per cell -> ~16x16 grid (256 values) on 1024 images.
# Output: .../alpha_vis_blank_v3/
#
# Usage:
#   bash evaluation/run_gmkontext_uedit_fixedeps_alpha_vis_v3.sh
#   SUITE=basic MAX_SAMPLES=8 bash evaluation/run_gmkontext_uedit_fixedeps_alpha_vis_v3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
EDITFLOW_DIR="${EDITFLOW_DIR:-${WORKSPACE_ROOT}/EditFlow}"
RUN_TAG="${RUN_TAG:-gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_iter_8500}"

export LABEL_PATCH_PX=64
export OUTPUT_DIR="${OUTPUT_DIR:-${EDITFLOW_DIR}/evaluation/imgedit_bench/outputs/runs/${RUN_TAG}/alpha_vis_blank_v3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"

exec bash "${SCRIPT_DIR}/run_gmkontext_uedit_fixedeps_alpha_vis.sh" "$@"
