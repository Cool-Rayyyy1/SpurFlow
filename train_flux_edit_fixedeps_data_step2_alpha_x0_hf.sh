#!/usr/bin/env bash
# Alpha PIID + latent edit-HF on alpha local crop (NO GAN, NO RGB x0):
#   L = L_PIID + λ_hf ||H(Δ_S) - H(Δ_GT)||_1
# Δ_S = ẑ0_S - z_src, Δ_GT = z_edit - z_src (unpatchified latents)
# H(z) = z - GaussianBlur(z); crop from step-2 alpha mask local.
#
#   bash train_flux_edit_fixedeps_data_step2_alpha_x0_hf.sh
#   GPU_IDS=0 bash train_flux_edit_fixedeps_data_step2_alpha_x0_hf.sh 1
#
# Knobs:
#   HF_LOSS_WEIGHT=0.3 HF_BLUR_SIGMA=1.0 LATENT_CROP_SIZE=64
#   PRETRAIN_CKPT=checkpoints/model/..._step2_alph_dino_gan/iter_7000.pth  (default)

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
GPU_IDS="${GPU_IDS:-0}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
# Default: latest alph_dino_gan student under checkpoints/model/...
PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/model/gmkontext_uedit_fixedeps_alpha_k16_${NFE}nfe_pico400k_step2_alph_dino_gan/iter_7000.pth}"
# RGB x0 removed; keep knob at 0.
X0_LOSS_WEIGHT="${X0_LOSS_WEIGHT:-0.0}"
HF_LOSS_WEIGHT="${HF_LOSS_WEIGHT:-0.3}"
HF_BLUR_SIGMA="${HF_BLUR_SIGMA:-1.0}"
LATENT_CROP_SIZE="${LATENT_CROP_SIZE:-64}"
GAN_GRAD_STEP2_ONLY="${GAN_GRAD_STEP2_ONLY:-true}"
NUM_DECAY_ITERS="${NUM_DECAY_ITERS:-0}"
# --------------------------------

# New run name: do not auto-resume the old RGB-x0 run.
RUN_NAME="gmkontext_uedit_fixedeps_alpha_k16_${NFE}nfe_pico400k_step2_latent_hf"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"

CKPT_BASE="checkpoints/${RUN_NAME}"
LOAD_FROM=""
RESUME_FROM=""
if [[ "${FRESH}" != "1" ]]; then
    if [[ -n "${PRETRAIN_CKPT}" && -e "${PROJECT_DIR}/${PRETRAIN_CKPT}" ]]; then
        LOAD_FROM="${PRETRAIN_CKPT}"
        echo "[pretrain] loading alpha student from ${LOAD_FROM}"
    elif [[ -n "${PRETRAIN_CKPT}" && -e "${PRETRAIN_CKPT}" ]]; then
        LOAD_FROM="${PRETRAIN_CKPT}"
        echo "[pretrain] loading alpha student from ${LOAD_FROM}"
    elif [[ -n "${PRETRAIN_CKPT}" ]]; then
        echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT}); starting without load_from" >&2
    else
        echo "[pretrain] PRETRAIN_CKPT empty; starting without load_from"
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
        echo "[resume] resuming latent-hf run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
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
    "train_cfg.x0_loss_weight=${X0_LOSS_WEIGHT}"
    "train_cfg.hf_loss_weight=${HF_LOSS_WEIGHT}"
    "train_cfg.hf_blur_sigma=${HF_BLUR_SIGMA}"
    "train_cfg.latent_crop_size=${LATENT_CROP_SIZE}"
    "train_cfg.gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY}"
    "train_cfg.num_decay_iters=${NUM_DECAY_ITERS}"
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

echo "Launching alpha latent-HF (no GAN/no RGB x0): nproc=${NUM_GPUS}  iters=${TOTAL_ITERS}  hf_w=${HF_LOSS_WEIGHT}  hf_sigma=${HF_BLUR_SIGMA}  latent_crop=${LATENT_CROP_SIZE}  load=${LOAD_FROM:-none}  resume=${RESUME_FROM:-none}  run=${RUN_NAME}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_step2_alpha_x0_hf.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
