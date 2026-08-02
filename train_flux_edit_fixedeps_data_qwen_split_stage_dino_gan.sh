#!/usr/bin/env bash
# EditFlow Qwen-Image-Edit fixed-eps split-stage PIID + DINO feature GAN (no alpha).
# Same unroll sharing as train_flux_edit_fixedeps_alpha_data_qwen_alph_dino_gan.sh:
#   ArcFlowEditImitationSplitStageGAN
#     step-1 pred + PIID -> integrate mid -> step-2 pred + PIID/direct
#     GAN fake = integrate same step-2 policy to t=0 (NO extra independent 2-NFE rollout)
#   DinoFeatureDiscriminator: random global/local crops (not alpha-mask)
#   condition_on_source=true: cat([DINO(ref), DINO(target)]) -> D(ref, edit)=1 / D(ref, student)=0
#   gan_grad_step2_only=false (grads through both NFE steps, like alph)
#
#   bash train_flux_edit_fixedeps_data_qwen_split_stage_dino_gan.sh              # default: 2 GPUs
#   bash train_flux_edit_fixedeps_data_qwen_split_stage_dino_gan.sh 8
#   GPU_IDS=0,1,2,3,4,5,6,7 bash train_flux_edit_fixedeps_data_qwen_split_stage_dino_gan.sh 8
#
# Pretrain / fresh (optional base fixedeps student weights):
#   PRETRAIN_CKPT=checkpoints/gmqwen_uedit_fixedeps_k16_2nfe_pico400k/<run>/iter_XXXX.pth \
#     bash train_flux_edit_fixedeps_data_qwen_split_stage_dino_gan.sh
#   FRESH=1 bash train_flux_edit_fixedeps_data_qwen_split_stage_dino_gan.sh

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
CKPT_INTERVAL="${CKPT_INTERVAL:-500}"
CKPT_MUST_SAVE_INTERVAL="${CKPT_MUST_SAVE_INTERVAL:-1000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-100}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
TOTAL_ITERS="${TOTAL_ITERS:-50000}"
EVAL="${EVAL:-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
QWEN_MODEL="${QWEN_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/Qwen-Image-Edit-2511}"
DINOV3_MODEL="${DINOV3_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
GPU_IDS="${GPU_IDS:-0,1}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
STEP2_GAN_WARMUP_ITERS="${STEP2_GAN_WARMUP_ITERS:-0}"
STEP2_GAN_RAMP_ITERS="${STEP2_GAN_RAMP_ITERS:-0}"
STEP2_GAN_WEIGHT="${STEP2_GAN_WEIGHT:-0.05}"
SPLIT_STAGE_TEACHER_WEIGHT="${SPLIT_STAGE_TEACHER_WEIGHT:-0.5}"
SPLIT_STAGE_DIFFUSION_WEIGHT="${SPLIT_STAGE_DIFFUSION_WEIGHT:-0.5}"
SPLIT_STAGE_STEP2_X_REF_SCALE="${SPLIT_STAGE_STEP2_X_REF_SCALE:-1.0}"
GAN_GRAD_STEP2_ONLY="${GAN_GRAD_STEP2_ONLY:-false}"
CONDITION_ON_SOURCE="${CONDITION_ON_SOURCE:-true}"
NUM_DECAY_ITERS="${NUM_DECAY_ITERS:-0}"
DINO_GLOBAL_SIZE="${DINO_GLOBAL_SIZE:-224}"
DINO_LOCAL_SIZE="${DINO_LOCAL_SIZE:-224}"
DINO_FEATURE_LAYERS="${DINO_FEATURE_LAYERS:-23}"
# --------------------------------

RUN_NAME="gmqwen_uedit_fixedeps_k16_${NFE}nfe_pico400k_split_stage_dino_gan"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export TOKENIZERS_PARALLELISM=false
export QWEN_MODEL_PATH="${QWEN_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"

CKPT_BASE="checkpoints/${RUN_NAME}"
LOAD_FROM=""
RESUME_FROM=""
if [[ -n "${PRETRAIN_CKPT}" && -e "${PROJECT_DIR}/${PRETRAIN_CKPT}" ]]; then
    LOAD_FROM="${PRETRAIN_CKPT}"
    echo "[pretrain] loading Qwen fixedeps student weights from ${LOAD_FROM}"
elif [[ -n "${PRETRAIN_CKPT}" ]]; then
    echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT}); continuing without load_from" >&2
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
        echo "[resume] resuming qwen-split-stage-dino-gan run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
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
    "train_cfg.split_stage_teacher_loss_weight=${SPLIT_STAGE_TEACHER_WEIGHT}"
    "train_cfg.split_stage_diffusion_loss_weight=${SPLIT_STAGE_DIFFUSION_WEIGHT}"
    "train_cfg.split_stage_step2_x_ref_scale=${SPLIT_STAGE_STEP2_X_REF_SCALE}"
    "train_cfg.gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY}"
    "train_cfg.num_decay_iters=${NUM_DECAY_ITERS}"
    "model.discriminator.checkpoint_path=${DINOV3_MODEL}"
    "model.discriminator.num_steps=${NFE}"
    "model.discriminator.global_input_size=${DINO_GLOBAL_SIZE}"
    "model.discriminator.local_input_size=${DINO_LOCAL_SIZE}"
    "model.discriminator.feature_layers=(${DINO_FEATURE_LAYERS},)"
    "model.discriminator.condition_on_source=${CONDITION_ON_SOURCE}"
    "checkpoint_config.interval=${CKPT_INTERVAL}"
    "checkpoint_config.must_save_interval=${CKPT_MUST_SAVE_INTERVAL}"
    "sample_eval.interval=${SAMPLE_INTERVAL}"
    "sample_eval.must_save_interval=0"
    "sample_eval.dataset.samples_per_category=${SAMPLES_PER_CATEGORY}"
    "total_iters=${TOTAL_ITERS}"
    "data.train.data_root=${DATA_ROOT}"
    "data.val.data_root=${DATA_ROOT}"
    "data.train.load_unpaired_edited=False"
    "model.vae.from_pretrained=${QWEN_MODEL}"
    "model.text_encoder.from_pretrained=${QWEN_MODEL}"
    "model.diffusion.denoising.pretrained=${QWEN_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
    "model.teacher.denoising.pretrained=${QWEN_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
)

if [[ "${EVAL}" == "1" ]]; then
    CFG_OPTS+=("sample_eval.enabled=true")
else
    CFG_OPTS+=("sample_eval.enabled=false")
fi

echo "Launching EditFlow Qwen split-stage DINO GAN FSDP: nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  sample_interval=${SAMPLE_INTERVAL}  gan_weight=${STEP2_GAN_WEIGHT}  gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY}  condition_on_source=${CONDITION_ON_SOURCE}  w_teacher=${SPLIT_STAGE_TEACHER_WEIGHT}  w_direct=${SPLIT_STAGE_DIFFUSION_WEIGHT}  load_from=${LOAD_FROM:-none}  resume_from=${RESUME_FROM:-none}  run=${RUN_NAME}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/qwen/editqwen_uedit_fixedeps_2nfe_k16_data_split_stage_dino_gan.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
