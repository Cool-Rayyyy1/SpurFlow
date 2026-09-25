#!/usr/bin/env bash
# Formal SpurFlow training on FLUX.1 Kontext.
#
# Load the warmup student, then train a shared 2-NFE split-stage rollout:
#   step 1 (t=1): teacher imitation
#   step 2 (step-1 endpoint -> t=0): teacher imitation + direct flow
#     + source-conditional DINO GAN on the decoded endpoint
# The discriminator sees the reference image with the dataset edit as real,
# and the same reference with the student decode as fake. Both views share a
# full-frame crop and an alpha-mask local crop. Frozen DINO features are
# channel-concatenated before the trainable head.
# This is one rollout. It does not run a second independent 2-NFE pass.
#
# Required:
#   KONTEXT_MODEL   local FLUX.1-Kontext-dev directory
#   DINOV3_MODEL    DINOv3 ViT-L/16 safetensors
#   DATA_A_ROOT     paired edit set A (metadata.jsonl, or built)
#   DATA_B_ROOT     paired edit set B (metadata.jsonl, or DATA_B_JSONL)
#   PRETRAIN_CKPT   warmup checkpoint (relative to the repo, or absolute)
# Optional:
#   DATA_A_PROB=0.3 DATA_B_PROB=0.7 NFE=2 TOTAL_ITERS=25000
#   GAN_WEIGHT=0.05 GAN_WARMUP_ITERS=0 GAN_RAMP_ITERS=0
#   TEACHER_LOSS_WEIGHT=0.5 DIFFUSION_LOSS_WEIGHT=0.5
#   GAN_GRAD_STEP2_ONLY=false
#   GPU_IDS=0,1,2,3,4,5,6,7 NUM_GPUS=8
#   FRESH=1
#   ALLOW_NO_PRETRAIN=1   # start without warmup weights
#
#   PRETRAIN_CKPT=checkpoints/spurflow_warmup/<run>/latest.pth \
#     bash train_flux_kontext.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/setup_env.sh"

if [[ $# -ge 1 && "${1}" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="${1}"
    shift
else
    NUM_GPUS="${NUM_GPUS:-${NPROC:-8}}"
fi

if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "Invalid NUM_GPUS=${NUM_GPUS}; expected a positive integer." >&2
    exit 1
fi

# ---------- user knobs ----------
NFE="${NFE:-2}"
CKPT_INTERVAL="${CKPT_INTERVAL:-500}"
CKPT_MUST_SAVE_INTERVAL="${CKPT_MUST_SAVE_INTERVAL:-1000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-500}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
GEDIT_SAMPLES_PER_CATEGORY="${GEDIT_SAMPLES_PER_CATEGORY:-3}"
TOTAL_ITERS="${TOTAL_ITERS:-25000}"
EVAL="${EVAL:-1}"
DATA_B_ROOT="${DATA_B_ROOT:-}"
DATA_A_ROOT="${DATA_A_ROOT:-}"
DATA_A_JSONL="${DATA_A_JSONL:-${DATA_A_ROOT}/metadata.jsonl}"
DATA_B_JSONL="${DATA_B_JSONL:-metadata.jsonl}"
DATA_A_PROB="${DATA_A_PROB:-0.3}"
DATA_B_PROB="${DATA_B_PROB:-0.7}"
KONTEXT_MODEL="${KONTEXT_MODEL:-}"
DINOV3_MODEL="${DINOV3_MODEL:-}"
IMGEDIT_ANN="${IMGEDIT_ANN:-${PROJECT_DIR}/evaluation/imgedit_bench/annotations/basic_edit.json}"
IMGEDIT_ROOT="${IMGEDIT_ROOT:-}"
GEDIT_META="${GEDIT_META:-}"
GEDIT_ROOT="${GEDIT_ROOT:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
GAN_WARMUP_ITERS="${GAN_WARMUP_ITERS:-0}"
GAN_RAMP_ITERS="${GAN_RAMP_ITERS:-0}"
GAN_WEIGHT="${GAN_WEIGHT:-0.05}"
TEACHER_LOSS_WEIGHT="${TEACHER_LOSS_WEIGHT:-0.5}"
DIFFUSION_LOSS_WEIGHT="${DIFFUSION_LOSS_WEIGHT:-0.5}"
STEP2_X_REF_SCALE="${STEP2_X_REF_SCALE:-1.0}"
GAN_GRAD_STEP2_ONLY="${GAN_GRAD_STEP2_ONLY:-false}"
NUM_DECAY_ITERS="${NUM_DECAY_ITERS:-0}"
DINO_GLOBAL_SIZE="${DINO_GLOBAL_SIZE:-224}"
DINO_LOCAL_SIZE="${DINO_LOCAL_SIZE:-224}"
DINO_FEATURE_LAYERS="${DINO_FEATURE_LAYERS:-23}"
P_DISABLE_LOCAL="${P_DISABLE_LOCAL:-0.3}"
EDIT_IS_LOW_ALPHA="${EDIT_IS_LOW_ALPHA:-true}"
ALPHA_SMOOTH_SIGMA="${ALPHA_SMOOTH_SIGMA:-2.0}"
MASS_THRESHOLD_PERCENTILE="${MASS_THRESHOLD_PERCENTILE:-30.0}"
MASS_COVERAGE_MIN="${MASS_COVERAGE_MIN:-0.85}"
MASS_COVERAGE_MAX="${MASS_COVERAGE_MAX:-0.90}"
HOT_MASS_FRAC="${HOT_MASS_FRAC:-0.40}"
UNION_AREA_MAX_RATIO="${UNION_AREA_MAX_RATIO:-0.35}"
BBOX_EXPAND_FACTOR="${BBOX_EXPAND_FACTOR:-1.1}"
MIN_CROP_AREA_RATIO="${MIN_CROP_AREA_RATIO:-0.01}"
MAX_CROP_AREA_RATIO="${MAX_CROP_AREA_RATIO:-0.35}"
MIN_EDIT_MASS_RATIO="${MIN_EDIT_MASS_RATIO:-0.002}"
MIN_COMPONENT_PIXELS="${MIN_COMPONENT_PIXELS:-16}"
GAN_GLOBAL_WEIGHT="${GAN_GLOBAL_WEIGHT:-1.0}"
GAN_RANDOM_LOCAL_WEIGHT="${GAN_RANDOM_LOCAL_WEIGHT:-1.0}"
GAN_MASK_LOCAL_WEIGHT="${GAN_MASK_LOCAL_WEIGHT:-1.0}"
CONDITION_ON_SOURCE="${CONDITION_ON_SOURCE:-true}"
USE_MASK_LOCAL_CROP="${USE_MASK_LOCAL_CROP:-true}"
NUM_LOCAL_CROPS="${NUM_LOCAL_CROPS:-1}"
# --------------------------------

if [[ -z "${KONTEXT_MODEL}" || -z "${DINOV3_MODEL}" || -z "${DATA_B_ROOT}" || -z "${DATA_A_ROOT}" ]]; then
    echo "Set KONTEXT_MODEL, DINOV3_MODEL, DATA_B_ROOT, and DATA_A_ROOT." >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-spurflow}"
CONFIG="${PROJECT_DIR}/configs/kontext/editflux_kontext_split_stage_alpha_dino_gan.py"

if [[ ! -f "${DATA_A_JSONL}" ]]; then
    if [[ "${RANK:-0}" == "0" ]]; then
        echo "[data] building ${DATA_A_JSONL}"
        python "${PROJECT_DIR}/tools/build_pair_jsonl.py" --root "${DATA_A_ROOT}" --out "${DATA_A_JSONL}"
    else
        echo "[data] waiting for ${DATA_A_JSONL}"
        for _ in $(seq 1 120); do
            [[ -f "${DATA_A_JSONL}" ]] && break
            sleep 5
        done
        [[ -f "${DATA_A_JSONL}" ]] || { echo "[data] timeout waiting for ${DATA_A_JSONL}" >&2; exit 1; }
    fi
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
CKPT_BASE="checkpoints/${RUN_NAME}"
LOAD_FROM=""
RESUME_FROM=""
if [[ -n "${PRETRAIN_CKPT}" ]]; then
    if [[ "${PRETRAIN_CKPT}" = /* ]]; then
        _pretrain_path="${PRETRAIN_CKPT}"
    else
        _pretrain_path="${PROJECT_DIR}/${PRETRAIN_CKPT}"
    fi
    if [[ -e "${_pretrain_path}" ]]; then
        LOAD_FROM="${PRETRAIN_CKPT}"
        if [[ "${PRETRAIN_CKPT}" = /* ]]; then
            LOAD_FROM="${_pretrain_path}"
        fi
        echo "[pretrain] loading warmup student from ${LOAD_FROM}"
    else
        echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT})." >&2
        exit 1
    fi
elif [[ "${ALLOW_NO_PRETRAIN:-0}" != "1" ]]; then
    echo "[pretrain] Set PRETRAIN_CKPT to the warmup checkpoint. ALLOW_NO_PRETRAIN=1 overrides this." >&2
    exit 1
fi

if [[ "${FRESH}" != "1" ]]; then
    RESUME_RUN_ID="${RESUME_RUN_DIR:-${RUN_ID:-}}"
    if [[ -z "${RESUME_RUN_ID}" && -d "${PROJECT_DIR}/${CKPT_BASE}" ]]; then
        for _cand in $(ls -1dt "${PROJECT_DIR}/${CKPT_BASE}"/*/ 2>/dev/null); do
            if [[ -e "${_cand}latest.pth" ]]; then
                RESUME_RUN_ID="$(basename "${_cand}")"
                break
            fi
        done
    fi
    if [[ -n "${RESUME_RUN_ID}" && -e "${PROJECT_DIR}/${CKPT_BASE}/${RESUME_RUN_ID}/latest.pth" ]]; then
        RESUME_FROM="${CKPT_BASE}/${RESUME_RUN_ID}/latest.pth"
        RESUME_RUN_DIR="${RESUME_RUN_ID}"
        LOAD_FROM=""
        echo "[resume] run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
    fi
fi

export RUN_ID
export RESUME_RUN_DIR

CFG_OPTS=(
    "name=${RUN_NAME}"
    "work_dir=work_dirs/${RUN_NAME}"
    "load_from=${LOAD_FROM}"
    "resume_from=${RESUME_FROM}"
    "checkpoint_config.out_dir=checkpoints/${RUN_NAME}"
    "train_cfg.nfe=${NFE}"
    "test_cfg.nfe=${NFE}"
    "train_cfg.split_stage_gan_warmup_iters=${GAN_WARMUP_ITERS}"
    "train_cfg.split_stage_gan_ramp_iters=${GAN_RAMP_ITERS}"
    "train_cfg.split_stage_gan_loss_weight=${GAN_WEIGHT}"
    "train_cfg.split_stage_teacher_loss_weight=${TEACHER_LOSS_WEIGHT}"
    "train_cfg.split_stage_diffusion_loss_weight=${DIFFUSION_LOSS_WEIGHT}"
    "train_cfg.split_stage_step2_x_ref_scale=${STEP2_X_REF_SCALE}"
    "train_cfg.gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY}"
    "train_cfg.num_decay_iters=${NUM_DECAY_ITERS}"
    "model.discriminator.checkpoint_path=${DINOV3_MODEL}"
    "model.discriminator.num_steps=${NFE}"
    "model.discriminator.global_input_size=${DINO_GLOBAL_SIZE}"
    "model.discriminator.local_input_size=${DINO_LOCAL_SIZE}"
    "model.discriminator.feature_layers=(${DINO_FEATURE_LAYERS},)"
    "model.discriminator.p_disable_local=${P_DISABLE_LOCAL}"
    "model.discriminator.edit_is_low_alpha=${EDIT_IS_LOW_ALPHA}"
    "model.discriminator.alpha_smooth_sigma=${ALPHA_SMOOTH_SIGMA}"
    "model.discriminator.mass_threshold_percentile=${MASS_THRESHOLD_PERCENTILE}"
    "model.discriminator.mass_coverage_min=${MASS_COVERAGE_MIN}"
    "model.discriminator.mass_coverage_max=${MASS_COVERAGE_MAX}"
    "model.discriminator.hot_mass_frac=${HOT_MASS_FRAC}"
    "model.discriminator.union_area_max_ratio=${UNION_AREA_MAX_RATIO}"
    "model.discriminator.bbox_expand_factor=${BBOX_EXPAND_FACTOR}"
    "model.discriminator.min_crop_area_ratio=${MIN_CROP_AREA_RATIO}"
    "model.discriminator.max_crop_area_ratio=${MAX_CROP_AREA_RATIO}"
    "model.discriminator.min_edit_mass_ratio=${MIN_EDIT_MASS_RATIO}"
    "model.discriminator.min_component_pixels=${MIN_COMPONENT_PIXELS}"
    "model.discriminator.gan_global_weight=${GAN_GLOBAL_WEIGHT}"
    "model.discriminator.gan_random_local_weight=${GAN_RANDOM_LOCAL_WEIGHT}"
    "model.discriminator.gan_mask_local_weight=${GAN_MASK_LOCAL_WEIGHT}"
    "model.discriminator.condition_on_source=${CONDITION_ON_SOURCE}"
    "model.discriminator.use_mask_local_crop=${USE_MASK_LOCAL_CROP}"
    "model.discriminator.num_local_crops=${NUM_LOCAL_CROPS}"
    "checkpoint_config.interval=${CKPT_INTERVAL}"
    "checkpoint_config.must_save_interval=${CKPT_MUST_SAVE_INTERVAL}"
    "sample_eval.interval=${SAMPLE_INTERVAL}"
    "sample_eval.must_save_interval=0"
    "sample_eval.dataset.datasets.0.samples_per_category=${SAMPLES_PER_CATEGORY}"
    "sample_eval.dataset.datasets.0.annotations_path=${IMGEDIT_ANN}"
    "sample_eval.dataset.datasets.1.samples_per_category=${GEDIT_SAMPLES_PER_CATEGORY}"
    "total_iters=${TOTAL_ITERS}"
    "data.train.probs=[${DATA_A_PROB},${DATA_B_PROB}]"
    "data.train.datasets.0.data_root=${DATA_A_ROOT}"
    "data.train.datasets.0.jsonl_path=${DATA_A_JSONL}"
    "data.train.datasets.1.data_root=${DATA_B_ROOT}"
    "data.train.datasets.1.jsonl_path=${DATA_B_JSONL}"
    "data.val.data_root=${DATA_B_ROOT}"
    "data.val.jsonl_path=${DATA_B_JSONL}"
    "model.vae.from_pretrained=${KONTEXT_MODEL}"
    "model.text_encoder.from_pretrained=${KONTEXT_MODEL}"
    "model.diffusion.denoising.pretrained=${KONTEXT_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
    "model.teacher.denoising.pretrained=${KONTEXT_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
)

if [[ -n "${IMGEDIT_ROOT}" ]]; then
    CFG_OPTS+=("sample_eval.dataset.datasets.0.bench_root=${IMGEDIT_ROOT}")
fi
if [[ -n "${GEDIT_META}" ]]; then
    CFG_OPTS+=("sample_eval.dataset.datasets.1.annotations_path=${GEDIT_META}")
fi
if [[ -n "${GEDIT_ROOT}" ]]; then
    CFG_OPTS+=("sample_eval.dataset.datasets.1.bench_root=${GEDIT_ROOT}")
fi

if [[ "${EVAL}" == "1" ]]; then
    CFG_OPTS+=("sample_eval.enabled=true")
else
    CFG_OPTS+=("sample_eval.enabled=false")
fi

echo "Formal FLUX.1 Kontext split-stage GAN (mix ${DATA_A_PROB}/${DATA_B_PROB}): nproc=${NUM_GPUS} total_iters=${TOTAL_ITERS} gan_weight=${GAN_WEIGHT} teacher_w=${TEACHER_LOSS_WEIGHT} diffusion_w=${DIFFUSION_LOSS_WEIGHT} gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY} load_from=${LOAD_FROM:-none} resume_from=${RESUME_FROM:-none} run=${RUN_NAME}"

torchrun \
    --nnodes="${WORLD_SIZE:-1}" \
    --nproc_per_node="${NUM_GPUS}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29500}" \
    "${PROJECT_DIR}/train.py" \
    "${CONFIG}" \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
