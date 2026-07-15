#!/usr/bin/env bash
# DMD2 (distribution matching + fake score) for FLUX Kontext image editing.
# Data-dependent pico-banana-400k benchmark, 2-NFE student.
#
# Mirrors the official DMD2 multi-step recipe: backward simulation
# (random-step no-grad bootstrap + single-step gradient), shared generator
# sample for DM and fake-score turns, and the DMD2-native GAN (cls head on
# fake-score features over latents; no external DINO discriminator).
#
#   bash train_flux_edit_dmd2_fixedeps_data.sh              # default: 2 GPUs
#   bash train_flux_edit_dmd2_fixedeps_data.sh 8
#   GPU_IDS=0,1,2,3 bash train_flux_edit_dmd2_fixedeps_data.sh 4
#   TOTAL_ITERS=30000 bash train_flux_edit_dmd2_fixedeps_data.sh
#   GAN_WEIGHT=0 bash train_flux_edit_dmd2_fixedeps_data.sh   # DM + fake score only
#
# Resume / fresh:
#   RESUME_RUN_DIR=<run_id> bash train_flux_edit_dmd2_fixedeps_data.sh
#   FRESH=1 bash train_flux_edit_dmd2_fixedeps_data.sh
#
# Optional warm-start from fixed-eps PIID pretrain:
#   PRETRAIN_CKPT=checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k/.../iter_20000.pth \
#     bash train_flux_edit_dmd2_fixedeps_data.sh

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
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"
TOTAL_ITERS="${TOTAL_ITERS:-20000}"
EVAL="${EVAL:-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
GPU_IDS="${GPU_IDS:-0,1}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
if [[ "${FRESH}" == "1" ]]; then
    RESUME_RUN_DIR=""
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
fi
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
# Official DMD2: dfake_gen_update_ratio=5, gen_cls_loss_weight=5e-3,
# guidance_cls_loss_weight=1e-2
GEN_UPDATE_RATIO="${GEN_UPDATE_RATIO:-5}"
GAN_WEIGHT="${GAN_WEIGHT:-5e-3}"
GUIDANCE_CLS_WEIGHT="${GUIDANCE_CLS_WEIGHT:-1e-2}"
DM_WEIGHT="${DM_WEIGHT:-1.0}"
FAKE_WEIGHT="${FAKE_WEIGHT:-1.0}"
REG_WEIGHT="${REG_WEIGHT:-0.0}"
GEN_LR="${GEN_LR:-5e-7}"
FAKE_LR="${FAKE_LR:-5e-7}"
# --------------------------------

RUN_NAME="gmkontext_dmd2_uedit_fixedeps_k16_${NFE}nfe_pico400k"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_BASE="checkpoints/${RUN_NAME}"
LOAD_FROM=""
RESUME_FROM=""
if [[ "${FRESH}" != "1" ]]; then
    if [[ -n "${PRETRAIN_CKPT}" && -e "${PROJECT_DIR}/${PRETRAIN_CKPT}" ]]; then
        LOAD_FROM="${PRETRAIN_CKPT}"
        echo "[pretrain] loading student weights from ${LOAD_FROM}"
    elif [[ -n "${PRETRAIN_CKPT}" ]]; then
        echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT}); continuing without it" >&2
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
        echo "[resume] resuming dmd2 run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
    fi
fi
if [[ -z "${LOAD_FROM}" && -z "${RESUME_FROM}" ]]; then
    echo "[init] no checkpoint to load (FRESH=${FRESH}); starting a new DMD2 run"
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
    "train_cfg.dmd2_gen_update_ratio=${GEN_UPDATE_RATIO}"
    "train_cfg.dmd2_dm_loss_weight=${DM_WEIGHT}"
    "train_cfg.dmd2_fake_loss_weight=${FAKE_WEIGHT}"
    "train_cfg.dmd2_reg_loss_weight=${REG_WEIGHT}"
    "train_cfg.dmd2_gan_loss_weight=${GAN_WEIGHT}"
    "train_cfg.dmd2_guidance_cls_loss_weight=${GUIDANCE_CLS_WEIGHT}"
    "optimizer.diffusion.lr=${GEN_LR}"
    "optimizer.fake_score.lr=${FAKE_LR}"
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
    "model.fake_score.denoising.pretrained=${KONTEXT_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
)

if [[ "${EVAL}" == "1" ]]; then
    CFG_OPTS+=("sample_eval.enabled=true")
else
    CFG_OPTS+=("sample_eval.enabled=false")
fi

ACTIVE_RUN_ID="${RESUME_RUN_DIR:-${RUN_ID:-auto}}"
echo "Launching DMD2 edit DDP/FSDP: nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  run=${RUN_NAME}/${ACTIVE_RUN_ID}  work_dir=work_dirs/${RUN_NAME}/${ACTIVE_RUN_ID}  ckpt_dir=checkpoints/${RUN_NAME}/${ACTIVE_RUN_ID}  nfe=${NFE}  gen_ratio=${GEN_UPDATE_RATIO}  gan_w=${GAN_WEIGHT}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_dmd2_uedit_fixedeps_2nfe_k16_data.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
