#!/usr/bin/env bash
# Kontext inversion: x_0=edited latent, cond=source image_latents + prompt + guidance.
# Compare 10-step vs 28-step teacher inversion on 10 random pico samples.
#
#   bash tools/run_test_teacher_invert_steps.sh
#   CUDA_VISIBLE_DEVICES=0 bash tools/run_test_teacher_invert_steps.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/setup_env.sh"

GPU_ID="${GPU_ID:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
STEPS_A="${STEPS_A:-10}"
STEPS_B="${STEPS_B:-28}"
SEED="${SEED:-42}"
OUTPUT_JSON="${OUTPUT_JSON:-${PROJECT_DIR}/work_dirs/teacher_invert_step_compare.json}"

python "${PROJECT_DIR}/tools/test_teacher_invert_steps.py" \
    --device "cuda:${GPU_ID}" \
    --num-samples "${NUM_SAMPLES}" \
    --steps-a "${STEPS_A}" \
    --steps-b "${STEPS_B}" \
    --seed "${SEED}" \
    --output-json "${OUTPUT_JSON}"
