#!/usr/bin/env bash
# Smoke: visualize step-2 alpha mask GAN crops on pico-banana val samples (1 GPU).
#
# Per sample outputs under evaluation/outputs/step2_alph_dino_gan_crop_vis/<tag>/val_XXXXX/:
#   crop_panel.png          — 2×2: src | edit-mass heatmap / crop boxes | mask zoom
#   step2_edit_mass_heatmap.png — low-α edit region heatmap (what drives mask crop)
#   step2_alpha_heatmap.png     — raw step2 α heatmap
#   crop_boxes_overlay.png      — global (green) / random local (blue) / mask local (red)
#   mask_crop_zoom.png          — cropped mask-local region
#   crop_specs.json             — normalized [top,left,h,w] specs
#
# Usage:
#   bash evaluation/smoke_step2_alph_dino_gan_crop_vis.sh
#   NUM_SAMPLES=4 GPU_ID=0 bash evaluation/smoke_step2_alph_dino_gan_crop_vis.sh
#   CKPT=checkpoints/model/.../iter_5000.pth bash evaluation/smoke_step2_alph_dino_gan_crop_vis.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/setup_env.sh"

GPU_ID="${GPU_ID:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
SEED="${SEED:-42}"
P_DISABLE_LOCAL="${P_DISABLE_LOCAL:-0.0}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
CKPT="${CKPT:-${PROJECT_DIR}/checkpoints/model/gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_20260713/iter_5000.pth}"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data.py}"
RUN_TAG="${RUN_TAG:-smoke_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/evaluation/outputs/step2_alph_dino_gan_crop_vis/${RUN_TAG}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/evaluation/imgedit_bench:${PYTHONPATH:-}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

echo "Smoke crop vis: GPU=${GPU_ID}  samples=${NUM_SAMPLES}  ckpt=${CKPT}"
echo "Output: ${OUTPUT_DIR}"

python "${PROJECT_DIR}/evaluation/run_step2_alph_dino_gan_crop_vis.py" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --device "cuda:0" \
    --p_disable_local "${P_DISABLE_LOCAL}" \
    --edit_is_low_alpha \
    --alpha_smooth_sigma "${ALPHA_SMOOTH_SIGMA:-2.0}" \
    --mass_threshold_percentile "${MASS_THRESHOLD_PERCENTILE:-30.0}" \
    --mass_coverage_min "${MASS_COVERAGE_MIN:-0.85}" \
    --mass_coverage_max "${MASS_COVERAGE_MAX:-0.90}" \
    --union_area_max_ratio "${UNION_AREA_MAX_RATIO:-0.55}" \
    --bbox_expand_factor "${BBOX_EXPAND_FACTOR:-1.4}" \
    --min_crop_area_ratio "${MIN_CROP_AREA_RATIO:-0.05}" \
    --max_crop_area_ratio "${MAX_CROP_AREA_RATIO:-0.55}" \
    --min_edit_mass_ratio "${MIN_EDIT_MASS_RATIO:-0.002}" \
    --min_component_pixels "${MIN_COMPONENT_PIXELS:-16}"

echo ""
echo "Done. Open panels:"
find "${OUTPUT_DIR}" -name 'crop_panel.png' | sort | head -20
echo ""
echo "Tip: step2_edit_mass_heatmap.png shows edit region (low α → warm colors)"
echo "     crop_boxes_overlay.png: green=global, blue=random local, red=mask local"
