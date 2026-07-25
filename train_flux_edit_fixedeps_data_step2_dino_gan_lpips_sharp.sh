#!/usr/bin/env bash
# Source-conditional local-refinement step2 DINO-GAN:
#   load student from step2_dino_gan iter_5500 (D head re-inits: 2x channels)
#   D(x_src, x_gt_edit) -> 1
#   D(x_src, x_student) -> 0
#   L = L_distill + λ_gan L_gan
#   local/patch DINO crops only; source is conditioner, not the real target
#
# Single-GPU overnight:
#   bash train_flux_edit_fixedeps_data_step2_dino_gan_lpips_sharp.sh
#   GPU_IDS=0 bash train_flux_edit_fixedeps_data_step2_dino_gan_lpips_sharp.sh 1
#
# Knobs:
#   STEP2_GAN_WEIGHT=0.05 STEP2_GAN_RAMP_ITERS=500
#   DINO_CONDITION_ON_SOURCE=true
#   DINO_NUM_GLOBAL_CROPS=0 DINO_NUM_LOCAL_CROPS=4
#   GAN_GLOBAL_WEIGHT=0.0 GAN_LOCAL_WEIGHT=1.0
#   DINO_DENSE_OUTPUT=true LPIPS_LOSS_WEIGHT=0.0
#   PRETRAIN_CKPT=.../iter_5500.pth

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/setup_env.sh"

if [[ $# -ge 1 && "${1}" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="${1}"
    shift
else
    NUM_GPUS="${NUM_GPUS:-${NPROC:-1}}"
fi

if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "Invalid NUM_GPUS=${NUM_GPUS}; expected a positive integer." >&2
    exit 1
fi

# ---------- user knobs ----------
NFE="${NFE:-2}"
CKPT_INTERVAL="${CKPT_INTERVAL:-500}"
CKPT_MUST_SAVE_INTERVAL="${CKPT_MUST_SAVE_INTERVAL:-1000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-100}"
TOTAL_ITERS="${TOTAL_ITERS:-15000}"
EVAL="${EVAL:-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
DINOV3_MODEL="${DINOV3_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
GPU_IDS="${GPU_IDS:-0}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
# Start from the 3.62 ImgEdit ckpt (weights only; new run name / optimizer).
PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/model/gmkontext_uedit_fixedeps_k16_${NFE}nfe_pico400k_step2_dino_gan/iter_5500.pth}"
STEP2_GAN_WARMUP_ITERS="${STEP2_GAN_WARMUP_ITERS:-0}"
STEP2_GAN_RAMP_ITERS="${STEP2_GAN_RAMP_ITERS:-500}"
# Keep λ_gan modest so distill owns "what to edit".
STEP2_GAN_WEIGHT="${STEP2_GAN_WEIGHT:-0.05}"
# Full-image LPIPS off by default (can pull content); optional HF-only later.
LPIPS_LOSS_WEIGHT="${LPIPS_LOSS_WEIGHT:-0.0}"
PERCEPTUAL_IMAGE_SIZE="${PERCEPTUAL_IMAGE_SIZE:-256}"
GAN_GRAD_STEP2_ONLY="${GAN_GRAD_STEP2_ONLY:-true}"
GAN_REAL_KEY="${GAN_REAL_KEY:-edited_images}"
NUM_DECAY_ITERS="${NUM_DECAY_ITERS:-0}"
DINO_GLOBAL_SIZE="${DINO_GLOBAL_SIZE:-224}"
DINO_LOCAL_SIZE="${DINO_LOCAL_SIZE:-224}"
DINO_FEATURE_LAYERS="${DINO_FEATURE_LAYERS:-23}"
# Local-only refinement GAN (no full-frame semantic view).
DINO_NUM_GLOBAL_CROPS="${DINO_NUM_GLOBAL_CROPS:-0}"
DINO_NUM_LOCAL_CROPS="${DINO_NUM_LOCAL_CROPS:-4}"
DINO_LOCAL_CROP_SCALE="${DINO_LOCAL_CROP_SCALE:-(0.08, 0.35)}"
DINO_DENSE_OUTPUT="${DINO_DENSE_OUTPUT:-true}"
DINO_CONDITION_ON_SOURCE="${DINO_CONDITION_ON_SOURCE:-true}"
GAN_GLOBAL_WEIGHT="${GAN_GLOBAL_WEIGHT:-0.0}"
GAN_LOCAL_WEIGHT="${GAN_LOCAL_WEIGHT:-1.0}"
# --------------------------------

if [[ "${GAN_REAL_KEY}" == "source_images" ]]; then
    echo "GAN_REAL_KEY=source_images is forbidden; source is only the D conditioner." >&2
    exit 1
fi

# New run name: src-cond head is incompatible with old unconditional D weights.
RUN_NAME="gmkontext_uedit_fixedeps_k16_${NFE}nfe_pico400k_step2_dino_gan_srccond_local"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"

CKPT_BASE="checkpoints/${RUN_NAME}"
LOAD_FROM=""
RESUME_FROM=""
if [[ "${FRESH}" != "1" ]]; then
    if [[ -n "${PRETRAIN_CKPT}" && -e "${PROJECT_DIR}/${PRETRAIN_CKPT}" ]]; then
        LOAD_FROM="${PRETRAIN_CKPT}"
        echo "[pretrain] loading student+D from ${LOAD_FROM}"
    elif [[ -n "${PRETRAIN_CKPT}" && -e "${PRETRAIN_CKPT}" ]]; then
        LOAD_FROM="${PRETRAIN_CKPT}"
        echo "[pretrain] loading student+D from ${LOAD_FROM}"
    else
        echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT}); starting from scratch" >&2
    fi

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
        echo "[resume] resuming local-refine run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
    fi
fi
if [[ -z "${LOAD_FROM}" && -z "${RESUME_FROM}" ]]; then
    echo "[init] no checkpoint to load (FRESH=${FRESH}); starting a new run"
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
    "train_cfg.split_stage_gan_warmup_iters=${STEP2_GAN_WARMUP_ITERS}"
    "train_cfg.split_stage_gan_ramp_iters=${STEP2_GAN_RAMP_ITERS}"
    "train_cfg.split_stage_gan_loss_weight=${STEP2_GAN_WEIGHT}"
    "train_cfg.gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY}"
    "train_cfg.gan_real_key=${GAN_REAL_KEY}"
    "train_cfg.lpips_loss_weight=${LPIPS_LOSS_WEIGHT}"
    "train_cfg.perceptual_image_size=${PERCEPTUAL_IMAGE_SIZE}"
    "train_cfg.num_decay_iters=${NUM_DECAY_ITERS}"
    "model.discriminator.checkpoint_path=${DINOV3_MODEL}"
    "model.discriminator.num_steps=${NFE}"
    "model.discriminator.global_input_size=${DINO_GLOBAL_SIZE}"
    "model.discriminator.local_input_size=${DINO_LOCAL_SIZE}"
    "model.discriminator.feature_layers=(${DINO_FEATURE_LAYERS},)"
    "model.discriminator.num_global_crops=${DINO_NUM_GLOBAL_CROPS}"
    "model.discriminator.num_local_crops=${DINO_NUM_LOCAL_CROPS}"
    "model.discriminator.local_crop_scale=${DINO_LOCAL_CROP_SCALE}"
    "model.discriminator.dense_output=${DINO_DENSE_OUTPUT}"
    "model.discriminator.condition_on_source=${DINO_CONDITION_ON_SOURCE}"
    "model.discriminator.gan_global_weight=${GAN_GLOBAL_WEIGHT}"
    "model.discriminator.gan_local_weight=${GAN_LOCAL_WEIGHT}"
    "checkpoint_config.interval=${CKPT_INTERVAL}"
    "checkpoint_config.must_save_interval=${CKPT_MUST_SAVE_INTERVAL}"
    "sample_eval.interval=${SAMPLE_INTERVAL}"
    "sample_eval.must_save_interval=0"
    "total_iters=${TOTAL_ITERS}"
    "data.train.data_root=${DATA_ROOT}"
    "data.val.data_root=${DATA_ROOT}"
    "model.vae.from_pretrained=${KONTEXT_MODEL}"
    "model.text_encoder.from_pretrained=${KONTEXT_MODEL}"
    "model.diffusion.denoising.pretrained=${KONTEXT_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
    "model.teacher.denoising.pretrained=${KONTEXT_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
)

if [[ "${EVAL}" == "1" ]]; then
    CFG_OPTS+=("sample_eval.enabled=true")
else
    CFG_OPTS+=("sample_eval.enabled=false")
fi

echo "Launching src-cond local DINO-GAN: nproc=${NUM_GPUS}  iters=${TOTAL_ITERS}  gan_w=${STEP2_GAN_WEIGHT}  ramp=${STEP2_GAN_RAMP_ITERS}  lpips_w=${LPIPS_LOSS_WEIGHT}  cond_src=${DINO_CONDITION_ON_SOURCE}  global_crops=${DINO_NUM_GLOBAL_CROPS}  local_crops=${DINO_NUM_LOCAL_CROPS}  gan_global_w=${GAN_GLOBAL_WEIGHT}  gan_local_w=${GAN_LOCAL_WEIGHT}  dense=${DINO_DENSE_OUTPUT}  real=${GAN_REAL_KEY}  load=${LOAD_FROM:-none}  resume=${RESUME_FROM:-none}  run=${RUN_NAME}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_step2_dino_gan_lpips_sharp.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
