#!/usr/bin/env bash
# EditFlow fixed-eps alpha PIID + step-2 TDM-style DINO feature GAN:
#   Standard random-segment PIID with ArcFlowEditNewAlpha (low alpha ~ edit region).
#   Step-2 endpoint: Kontext VAE decode -> shared crop specs:
#   Case A (local_enabled=False): global + random local; real = random edited
#       image sampled from the dataset (load_unpaired_edited; works at bs=1).
#   Case B (local_enabled=True): global + alpha-mask local, paired ref/edit.
#   Mask local crop uses step-2 alpha.detach() (edit mass = ref - alpha); fallback to global full.
#   GAN grads update the student only through the final NFE step (gan_grad_step2_only=true by default).
#   Sample eval: ImgEdit-Bench 9 categories x 5 (same as train_flux_edit_fixedeps_alpha_data.sh).
#
#   Uses FSDP (configs/kontext/_fsdp_train.py) to shard diffusion/teacher.
#   VAE + DINO discriminator stay replicated per GPU.
#
#   bash train_flux_edit_fixedeps_data_step2_alph_dino_gan.sh              # default: 8 GPUs
#   bash train_flux_edit_fixedeps_data_step2_alph_dino_gan.sh 4
#   GPU_IDS=0,1 bash train_flux_edit_fixedeps_data_step2_alph_dino_gan.sh 2
#
# Pretrain / fresh (expects alpha ckpt with proj_out_alpha):
#   PRETRAIN_CKPT=checkpoints/.../iter_20000.pth bash train_flux_edit_fixedeps_data_step2_alph_dino_gan.sh
#   FRESH=1 bash train_flux_edit_fixedeps_data_step2_alph_dino_gan.sh   # skip auto-resume; still loads PRETRAIN_CKPT

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
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-100}"
TOTAL_ITERS="${TOTAL_ITERS:-25000}"
EVAL="${EVAL:-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
DINOV3_MODEL="${DINOV3_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/model/gmkontext_uedit_fixedeps_alpha_k16_${NFE}nfe_pico400k_20260713/iter_10000.pth}"
STEP2_GAN_WARMUP_ITERS="${STEP2_GAN_WARMUP_ITERS:-0}"
STEP2_GAN_RAMP_ITERS="${STEP2_GAN_RAMP_ITERS:-0}"
STEP2_GAN_WEIGHT="${STEP2_GAN_WEIGHT:-0.05}"
DIRECT_DELTA_WEIGHT="${DIRECT_DELTA_WEIGHT:-0.5}"
GAN_GRAD_STEP2_ONLY="${GAN_GRAD_STEP2_ONLY:-true}"
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
UNION_AREA_MAX_RATIO="${UNION_AREA_MAX_RATIO:-0.55}"
BBOX_EXPAND_FACTOR="${BBOX_EXPAND_FACTOR:-1.4}"
MIN_CROP_AREA_RATIO="${MIN_CROP_AREA_RATIO:-0.05}"
MAX_CROP_AREA_RATIO="${MAX_CROP_AREA_RATIO:-0.55}"
MIN_EDIT_MASS_RATIO="${MIN_EDIT_MASS_RATIO:-0.002}"
MIN_COMPONENT_PIXELS="${MIN_COMPONENT_PIXELS:-16}"
GAN_GLOBAL_WEIGHT="${GAN_GLOBAL_WEIGHT:-1.0}"
GAN_RANDOM_LOCAL_WEIGHT="${GAN_RANDOM_LOCAL_WEIGHT:-1.0}"
GAN_MASK_LOCAL_WEIGHT="${GAN_MASK_LOCAL_WEIGHT:-1.0}"
# --------------------------------

RUN_NAME="gmkontext_uedit_fixedeps_alpha_k16_${NFE}nfe_pico400k_step2_alph_dino_gan"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"

CKPT_BASE="checkpoints/${RUN_NAME}"
LOAD_FROM=""
RESUME_FROM=""
if [[ -n "${PRETRAIN_CKPT}" && -e "${PROJECT_DIR}/${PRETRAIN_CKPT}" ]]; then
    LOAD_FROM="${PRETRAIN_CKPT}"
    echo "[pretrain] loading alpha student weights from ${LOAD_FROM}"
elif [[ -n "${PRETRAIN_CKPT}" ]]; then
    echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT}); starting from scratch" >&2
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
        echo "[resume] resuming step2-alph-dino-gan run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
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
    "train_cfg.direct_delta_loss_weight=${DIRECT_DELTA_WEIGHT}"
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
    "model.discriminator.union_area_max_ratio=${UNION_AREA_MAX_RATIO}"
    "model.discriminator.bbox_expand_factor=${BBOX_EXPAND_FACTOR}"
    "model.discriminator.min_crop_area_ratio=${MIN_CROP_AREA_RATIO}"
    "model.discriminator.max_crop_area_ratio=${MAX_CROP_AREA_RATIO}"
    "model.discriminator.min_edit_mass_ratio=${MIN_EDIT_MASS_RATIO}"
    "model.discriminator.min_component_pixels=${MIN_COMPONENT_PIXELS}"
    "model.discriminator.gan_global_weight=${GAN_GLOBAL_WEIGHT}"
    "model.discriminator.gan_random_local_weight=${GAN_RANDOM_LOCAL_WEIGHT}"
    "model.discriminator.gan_mask_local_weight=${GAN_MASK_LOCAL_WEIGHT}"
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

echo "Launching EditFlow step2 alpha-mask DINO GAN FSDP: nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  sample_interval=${SAMPLE_INTERVAL}  gan_weight=${STEP2_GAN_WEIGHT}  direct_delta_weight=${DIRECT_DELTA_WEIGHT}  gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY}  p_disable_local=${P_DISABLE_LOCAL}  edit_is_low_alpha=${EDIT_IS_LOW_ALPHA}  gan_weights=(${GAN_GLOBAL_WEIGHT},${GAN_RANDOM_LOCAL_WEIGHT},${GAN_MASK_LOCAL_WEIGHT})  load_from=${LOAD_FROM:-none}  resume_from=${RESUME_FROM:-none}  run=${RUN_NAME}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_step2_alph_dino_gan.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"