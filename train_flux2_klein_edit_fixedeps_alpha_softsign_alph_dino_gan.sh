#!/usr/bin/env bash
# EditFlow FLUX.2-klein-base Softsign-01 alpha + split-stage PIID + source-cond DINO GAN.
# GAN recipe matches train_flux_edit_fixedeps_alpha_data_qwen_alph_dino_gan.sh:
#   DinoAlphaMaskFeatureDiscriminator, condition_on_source,
#   cat([DINO(ref), DINO(target)]) -> D(ref, edit)=1, D(ref, student)=0
#   alpha-mask local crop from step-2 alpha (p_disable_local random local)
#   gan_weight=0.05, gan_grad_step2_only=false (grads through both NFE steps)
# Distillation is split-stage so fake endpoint reuses the same 2-NFE unroll
# (no extra independent rollout).
#
# Softsign-01:
#   alpha = 0.5 * (raw / (1 + abs(raw)) + 1)
#   student_u = path_epsilon - alpha * x_ref - pred_delta
#
# Guidance:
#   Teacher (klein-base): official CFG=4.0
#   Student: CFG=1 / cond-only
#
#   bash train_flux2_klein_edit_fixedeps_alpha_softsign_alph_dino_gan.sh
#   bash train_flux2_klein_edit_fixedeps_alpha_softsign_alph_dino_gan.sh 8
#   TEACHER_CFG=4.0 GPU_IDS=0,1,2,3 bash train_flux2_klein_edit_fixedeps_alpha_softsign_alph_dino_gan.sh 4
#
# Pretrain from Softsign-01 alpha (no-GAN) ckpt with proj_out_alpha:
#   PRETRAIN_CKPT=checkpoints/.../iter_XXXX.pth \
#     bash train_flux2_klein_edit_fixedeps_alpha_softsign_alph_dino_gan.sh
#   FRESH=1 bash train_flux2_klein_edit_fixedeps_alpha_softsign_alph_dino_gan.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${PROJECT_DIR}/setup_env.sh"

if [[ $# -ge 1 && "${1}" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="${1}"
    shift
else
    NUM_GPUS="${NUM_GPUS:-${NPROC:-2}}"
fi

if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "Invalid NUM_GPUS=${NUM_GPUS}; expected a positive integer." >&2
    exit 1
fi

# ---------- user knobs ----------
NFE="${NFE:-2}"
SHIFT="${SHIFT:-3.2}"
TEACHER_CFG="${TEACHER_CFG:-4.0}"
CKPT_INTERVAL="${CKPT_INTERVAL:-500}"
CKPT_MUST_SAVE_INTERVAL="${CKPT_MUST_SAVE_INTERVAL:-1000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-500}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
TOTAL_ITERS="${TOTAL_ITERS:-50000}"
EVAL="${EVAL:-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_gaochengmin/data/pico-banana-400k}"
KLEIN_MODEL="${KLEIN_MODEL:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"
DINOV3_MODEL="${DINOV3_MODEL:-/mnt/afs_gaochengmin/checkpoints/dinov3/dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
# Default: softsign no-GAN iter_20000 (override / unset as needed).
PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/gmklein_base_uedit_fixedeps_alpha_softsign01_k16_2nfe_shift3.2_teachercfg4.0_pico400k/20260806_155327/iter_20000.pth}"
STEP2_GAN_WARMUP_ITERS="${STEP2_GAN_WARMUP_ITERS:-0}"
STEP2_GAN_RAMP_ITERS="${STEP2_GAN_RAMP_ITERS:-0}"
STEP2_GAN_WEIGHT="${STEP2_GAN_WEIGHT:-0.05}"
DIRECT_DELTA_WEIGHT="${DIRECT_DELTA_WEIGHT:-0.0}"
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
# --------------------------------

RUN_NAME="gmklein_base_uedit_fixedeps_alpha_softsign01_k16_${NFE}nfe_shift${SHIFT}_teachercfg${TEACHER_CFG}_pico400k_split_stage_alph_dino_gan"
CONFIG="${PROJECT_DIR}/configs/flux2_klein/editflux2_klein_uedit_fixedeps_2nfe_k16_alpha_softsign_split_stage_alph_dino_gan.py"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export KLEIN_MODEL_PATH="${KLEIN_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
        LOAD_FROM="${_pretrain_path}"
        echo "[pretrain] loading Softsign-01 alpha student weights from ${LOAD_FROM}"
    else
        echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT})." >&2
        if [[ "${ALLOW_NO_PRETRAIN:-0}" == "1" ]]; then
            echo "[pretrain] ALLOW_NO_PRETRAIN=1; starting without softsign pretrain (mask crop may fallback)." >&2
        else
            echo "[pretrain] Refuse to start without softsign alpha weights. Set ALLOW_NO_PRETRAIN=1 to override." >&2
            exit 1
        fi
    fi
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
        echo "[resume] resuming klein-softsign-alph-dino-gan run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
    fi
fi
if [[ -z "${LOAD_FROM}" && -z "${RESUME_FROM}" ]]; then
    echo "[init] no checkpoint to load (FRESH=${FRESH}); starting a new Softsign-01 alph-dino-gan run"
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
    "train_cfg.teacher_guidance_scale=${TEACHER_CFG}"
    "test_cfg.guidance_scale=1.0"
    "model.diffusion.timestep_sampler.shift=${SHIFT}"
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
    "checkpoint_config.interval=${CKPT_INTERVAL}"
    "checkpoint_config.must_save_interval=${CKPT_MUST_SAVE_INTERVAL}"
    "sample_eval.interval=${SAMPLE_INTERVAL}"
    "sample_eval.must_save_interval=0"
    "sample_eval.dataset.samples_per_category=${SAMPLES_PER_CATEGORY}"
    "total_iters=${TOTAL_ITERS}"
    "data.train.data_root=${DATA_ROOT}"
    "data.val.data_root=${DATA_ROOT}"
    "model.vae.from_pretrained=${KLEIN_MODEL}"
    "model.text_encoder.from_pretrained=${KLEIN_MODEL}"
    "model.diffusion.denoising.pretrained=${KLEIN_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
    "model.teacher.denoising.pretrained=${KLEIN_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
)

if [[ "${EVAL}" == "1" ]]; then
    CFG_OPTS+=("sample_eval.enabled=true")
else
    CFG_OPTS+=("sample_eval.enabled=false")
fi

echo "Launching EditFlow FLUX.2-klein Softsign-01 alpha split-stage DINO GAN FSDP: nproc=${NUM_GPUS} nfe=${NFE} shift=${SHIFT} teacher_cfg=${TEACHER_CFG} student_cfg=1.0 total_iters=${TOTAL_ITERS} gan_weight=${STEP2_GAN_WEIGHT} gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY} condition_on_source=${CONDITION_ON_SOURCE} p_disable_local=${P_DISABLE_LOCAL} load_from=${LOAD_FROM:-none} resume_from=${RESUME_FROM:-none} run=${RUN_NAME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Single-node example:
# torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
#     "${CONFIG}" \
#     --launcher pytorch --diff_seed \
#     --cfg-options "${CFG_OPTS[@]}"

torchrun \
    --nnodes="${WORLD_SIZE}" \
    --nproc_per_node=8 \
    --node_rank="${RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train.py \
    "${CONFIG}" \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
