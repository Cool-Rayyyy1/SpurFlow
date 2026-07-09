#!/usr/bin/env bash
# EditFlow split-stage dual-LoRA + step-2 DINO feature GAN:
#   Loads fixed-eps pretrain (default iter_20000); single LoRA + output heads are
#   copied to both step1 and step2 modules at load time.
#   step1 LoRA: first rollout segment PIID
#   step2 LoRA: second segment PIID + direct flow MSE + GAN (GAN grad -> step2 LoRA only)
#   teacher_ratio stays 0 (num_decay_iters=0). GAN scale ramps 0->1 over 0-1000 iters.
#
#   bash train_flux_edit_fixedeps_data_split_stage_dual_lora_dino_gan.sh
#   GPU_IDS=0,1 bash train_flux_edit_fixedeps_data_split_stage_dual_lora_dino_gan.sh 2
#
# Pretrain / resume:
#   PRETRAIN_CKPT=checkpoints/.../iter_20000.pth bash train_flux_edit_fixedeps_data_split_stage_dual_lora_dino_gan.sh
#   FRESH=0 bash train_flux_edit_fixedeps_data_split_stage_dual_lora_dino_gan.sh   # resume this run
#   FRESH=1 bash train_flux_edit_fixedeps_data_split_stage_dual_lora_dino_gan.sh   # skip pretrain load

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
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"
TOTAL_ITERS="${TOTAL_ITERS:-25000}"
EVAL="${EVAL:-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
DINOV3_MODEL="${DINOV3_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
FIXEDEPS_RUN_NAME="gmkontext_uedit_fixedeps_k16_${NFE}nfe_pico400k"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/${FIXEDEPS_RUN_NAME}/20260618_055623/iter_20000.pth}"
SPLIT_STAGE_DIFFUSION_WEIGHT="${SPLIT_STAGE_DIFFUSION_WEIGHT:-0.5}"
SPLIT_STAGE_TEACHER_WEIGHT="${SPLIT_STAGE_TEACHER_WEIGHT:-0.5}"
SPLIT_STAGE_STEP2_X_REF_SCALE="${SPLIT_STAGE_STEP2_X_REF_SCALE:-1.0}"
SPLIT_STAGE_GAN_WARMUP_ITERS="${SPLIT_STAGE_GAN_WARMUP_ITERS:-0}"
SPLIT_STAGE_GAN_RAMP_ITERS="${SPLIT_STAGE_GAN_RAMP_ITERS:-1000}"
SPLIT_STAGE_GAN_WEIGHT="${SPLIT_STAGE_GAN_WEIGHT:-0.01}"
DINO_GLOBAL_SIZE="${DINO_GLOBAL_SIZE:-224}"
DINO_LOCAL_SIZE="${DINO_LOCAL_SIZE:-224}"
DINO_FEATURE_LAYERS="${DINO_FEATURE_LAYERS:-17,23}"  # ViT-L has blocks 0-23 only
# --------------------------------

RUN_NAME="gmkontext_uedit_fixedeps_k16_${NFE}nfe_pico400k_split_stage_dual_lora_dino_gan"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"

_resolve_pretrain_ckpt() {
    if [[ -n "${PRETRAIN_CKPT}" && -e "${PROJECT_DIR}/${PRETRAIN_CKPT}" ]]; then
        echo "${PRETRAIN_CKPT}"
        return
    fi
    if [[ -n "${PRETRAIN_CKPT}" ]]; then
        echo "${PRETRAIN_CKPT}"
    fi
}

CKPT_BASE="checkpoints/${RUN_NAME}"
LOAD_FROM=""
RESUME_FROM=""
if [[ "${FRESH}" != "1" ]]; then
    _PRETRAIN_CKPT="$(_resolve_pretrain_ckpt)"
    if [[ -n "${_PRETRAIN_CKPT}" && -e "${PROJECT_DIR}/${_PRETRAIN_CKPT}" ]]; then
        PRETRAIN_CKPT="${_PRETRAIN_CKPT}"
        LOAD_FROM="${PRETRAIN_CKPT}"
        echo "[pretrain] loading fixed-eps student weights from ${LOAD_FROM}"
        echo "[pretrain] single LoRA + heads will be copied to dual step1/step2 at load"
    else
        echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT}); starting without pretrain" >&2
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
        echo "[resume] resuming dual-lora split-stage-dino-gan run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
    fi
fi
if [[ -z "${LOAD_FROM}" && -z "${RESUME_FROM}" ]]; then
    echo "[init] no checkpoint to load (FRESH=${FRESH}); Kontext init + dual LoRA from scratch"
fi

export RUN_ID
export RESUME_RUN_DIR

IFS=',' read -r -a _DINO_LAYERS <<< "${DINO_FEATURE_LAYERS}"
DINO_FEATURE_LAYERS_CFG="("
for _i in "${!_DINO_LAYERS[@]}"; do
    [[ -n "${_DINO_LAYERS[_i]}" ]] || continue
    if [[ "${_i}" -gt 0 ]]; then
        DINO_FEATURE_LAYERS_CFG+=", "
    fi
    DINO_FEATURE_LAYERS_CFG+="${_DINO_LAYERS[_i]}"
done
DINO_FEATURE_LAYERS_CFG+=")"

CFG_OPTS=(
    "name=${RUN_NAME}"
    "work_dir=work_dirs/${RUN_NAME}"
    "load_from=${LOAD_FROM}"
    "resume_from=${RESUME_FROM}"
    "checkpoint_config.out_dir=checkpoints/${RUN_NAME}"
    "train_cfg.nfe=${NFE}"
    "test_cfg.nfe=${NFE}"
    "train_cfg.split_stage_diffusion_loss_weight=${SPLIT_STAGE_DIFFUSION_WEIGHT}"
    "train_cfg.split_stage_teacher_loss_weight=${SPLIT_STAGE_TEACHER_WEIGHT}"
    "train_cfg.split_stage_step2_x_ref_scale=${SPLIT_STAGE_STEP2_X_REF_SCALE}"
    "test_cfg.split_stage_step2_x_ref_scale=${SPLIT_STAGE_STEP2_X_REF_SCALE}"
    "train_cfg.split_stage_gan_warmup_iters=${SPLIT_STAGE_GAN_WARMUP_ITERS}"
    "train_cfg.split_stage_gan_ramp_iters=${SPLIT_STAGE_GAN_RAMP_ITERS}"
    "train_cfg.split_stage_gan_loss_weight=${SPLIT_STAGE_GAN_WEIGHT}"
    "train_cfg.gan_grad_step2_only=True"
    "train_cfg.num_decay_iters=0"
    "model.discriminator.checkpoint_path=${DINOV3_MODEL}"
    "model.discriminator.num_steps=${NFE}"
    "model.discriminator.global_input_size=${DINO_GLOBAL_SIZE}"
    "model.discriminator.local_input_size=${DINO_LOCAL_SIZE}"
    "model.discriminator.feature_layers=${DINO_FEATURE_LAYERS_CFG}"
    "model.diffusion.denoising.dual_stage_lora=True"
    "model.diffusion.denoising.checkpointing=True"
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

echo "Launching EditFlow split-stage dual-LoRA DINO GAN FSDP: nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  diffusion_w=${SPLIT_STAGE_DIFFUSION_WEIGHT}  teacher_w=${SPLIT_STAGE_TEACHER_WEIGHT}  gan_warmup=${SPLIT_STAGE_GAN_WARMUP_ITERS}  gan_ramp=${SPLIT_STAGE_GAN_RAMP_ITERS}  gan_weight=${SPLIT_STAGE_GAN_WEIGHT}  teacher_ratio=0  sample_interval=${SAMPLE_INTERVAL}  dino_layers=${DINO_FEATURE_LAYERS}  load_from=${LOAD_FROM:-none}  resume_from=${RESUME_FROM:-none}  fresh=${FRESH}  run=${RUN_NAME}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_split_stage_dual_lora_dino_gan.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
