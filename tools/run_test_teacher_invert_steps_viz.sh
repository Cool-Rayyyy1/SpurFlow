#!/usr/bin/env bash
# Visualize teacher inversion eps decoded at 5/10/15/20/28 steps (10 pico samples).
#
#   bash tools/run_test_teacher_invert_steps_viz.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/setup_env.sh"

GPU_ID="${GPU_ID:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
STEP_LIST="${STEP_LIST:-5,10,15,20,28}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/work_dirs/teacher_invert_steps_viz}"

python "${PROJECT_DIR}/tools/test_teacher_invert_steps_viz.py" \
    --device "cuda:${GPU_ID}" \
    --num-samples "${NUM_SAMPLES}" \
    --step-list "${STEP_LIST}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}"
