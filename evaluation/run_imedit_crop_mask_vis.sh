#!/usr/bin/env bash
# ImgEdit-Bench crop-mask vis: 9 categories × 5 cases (1 GPU).
#
# Categories: action add adjust background compose extract remove replace style
#
# Per case:
#   <output>/<category>/<case_key>/
#     src.png            — preprocessed source
#     edit.png           — student 2-NFE output
#     crop_boxes.png     — 2-crop boxes on src (global + random/mask local, Case A/B)
#     alpha_overlay.png  — step2 α overlay on src
#     prompt.txt, crop_specs.json
#
# Per category summary:
#   <output>/<category>/overview.png   — 5 rows × 4 columns grid
#
# Usage:
#   bash evaluation/run_imedit_crop_mask_vis.sh
#   GPU_ID=0 SAMPLES_PER_CATEGORY=5 bash evaluation/run_imedit_crop_mask_vis.sh
#   CKPT=checkpoints/model/.../iter_10000.pth bash evaluation/run_imedit_crop_mask_vis.sh
#   RUN_TAG=my_run bash evaluation/run_imedit_crop_mask_vis.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${PROJECT_DIR}/evaluation/imgedit_bench"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/setup_env.sh"

GPU_ID="${GPU_ID:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/imgedit}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
CKPT="${CKPT:-${PROJECT_DIR}/checkpoints/model/gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_20260713/iter_10000.pth}"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data.py}"
RUN_TAG="${RUN_TAG:-imedit_crop_vis_iter10000_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/outputs/runs/${RUN_TAG}/crop_mask_vis}"
OVERLAY_BLEND="${OVERLAY_BLEND:-0.52}"
P_DISABLE_LOCAL="${P_DISABLE_LOCAL:-0.0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export IMGEDIT_BENCH_ROOT="${IMGEDIT_BENCH_ROOT:-${DATA_ROOT}/benchmark/Benchmark}"
export STUDENT_RESIZE_MODE="${STUDENT_RESIZE_MODE:-kontext}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/evaluation:${EVAL_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

echo "Downloading ImgEdit-Bench if needed..."
bash "${EVAL_DIR}/download_imgedit_bench.sh"

SKIP_FLAG=()
if [[ "${FORCE:-0}" != "1" ]]; then
    SKIP_FLAG=(--skip_existing)
fi

echo "Crop-mask vis: GPU=${GPU_ID}  categories=9  per_category=${SAMPLES_PER_CATEGORY}  total=$((9 * SAMPLES_PER_CATEGORY))"
echo "CKPT: ${CKPT}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Categories: action add adjust background compose extract remove replace style"
echo ""

python "${EVAL_DIR}/run_crop_mask_vis_bench.py" \
    --bench_root "${IMGEDIT_BENCH_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --samples_per_category "${SAMPLES_PER_CATEGORY}" \
    --seed "${SEED}" \
    --device "cuda:0" \
    --overlay_blend "${OVERLAY_BLEND}" \
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
    --min_component_pixels "${MIN_COMPONENT_PIXELS:-16}" \
    "${SKIP_FLAG[@]}"

echo ""
echo "Done. Browse category overviews:"
for cat in action add adjust background compose extract remove replace style; do
    f="${OUTPUT_DIR}/${cat}/overview.png"
    if [[ -f "${f}" ]]; then
        echo "  ${f}"
    fi
done
echo ""
echo "Per-case folders: ${OUTPUT_DIR}/<category>/<case_key>/"
