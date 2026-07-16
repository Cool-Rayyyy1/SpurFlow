#!/usr/bin/env bash
# EditFlow fixed-eps alpha + LPIPS + DINOv3 feature losses:
#   PIID (random segment) as in train_flux_edit_fixedeps_alpha_data.sh
#   + 2-NFE rollout: step1/step2 analytic x0_hat = alpha*x_ref + pred_delta
#   + VAE decode each x0_hat vs edited image:
#       LPIPS  weight = 0.2  (20% of PIID scale)
#       DINO   weight = 0.1  (frozen dinov3-vitl16 feature cosine)
#
# Weights:
#   LPIPS: /mnt/afs_zhangyunzhe/pretrained_models/lpips/vgg.pth
#   DINO:  /mnt/afs_zhangyunzhe/pretrained_models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
#
#   bash train_flux_edit_fixedeps_alpha_data_lpips.sh              # default: 2 GPUs
#   bash train_flux_edit_fixedeps_alpha_data_lpips.sh 8
#   GPU_IDS=0 bash train_flux_edit_fixedeps_alpha_data_lpips.sh 1
#   LPIPS_WEIGHT=0.2 DINO_WEIGHT=0.1 bash train_flux_edit_fixedeps_alpha_data_lpips.sh
#
# Pretrain / resume:
#   PRETRAIN_CKPT=checkpoints/model/.../iter_18000.pth bash ...   # default
#   FRESH=1 bash ...                                              # skip auto-resume; still loads PRETRAIN_CKPT
#   RESUME_RUN_DIR=20260716_120000 bash ...

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
TOTAL_ITERS="${TOTAL_ITERS:-20000}"
EVAL="${EVAL:-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
LPIPS_WEIGHTS="${LPIPS_WEIGHTS:-/mnt/afs_zhangyunzhe/pretrained_models/lpips/vgg.pth}"
DINOV3_MODEL="${DINOV3_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.2}"
DINO_WEIGHT="${DINO_WEIGHT:-0.1}"
GPU_IDS="${GPU_IDS:-0,1}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/model/gmkontext_uedit_fixedeps_alpha_k16_${NFE}nfe_pico400k_20260713/iter_20000.pth}"
# --------------------------------

RUN_NAME="gmkontext_uedit_fixedeps_alpha_k16_${NFE}nfe_pico400k_lpips"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"

LPIPS_VGG16="${LPIPS_VGG16:-/mnt/afs_zhangyunzhe/pretrained_models/lpips/vgg16-397923af.pth}"

ensure_lpips_weights() {
    mkdir -p "$(dirname "${LPIPS_WEIGHTS}")"
    if [[ ! -f "${LPIPS_WEIGHTS}" ]]; then
        local url="https://download.openmmlab.com/mmgen/evaluation/lpips/weights/v0.1/vgg.pth"
        echo "[lpips] downloading lin layers ${url} -> ${LPIPS_WEIGHTS}"
        if command -v wget >/dev/null 2>&1; then
            wget -q -O "${LPIPS_WEIGHTS}" "${url}"
        else
            curl -fsSL -o "${LPIPS_WEIGHTS}" "${url}"
        fi
    fi
    echo "[lpips] lin weights: ${LPIPS_WEIGHTS}"

    if [[ ! -f "${LPIPS_VGG16}" ]]; then
        local url="https://download.pytorch.org/models/vgg16-397923af.pth"
        echo "[lpips] downloading VGG16 backbone ${url} -> ${LPIPS_VGG16}"
        if command -v aria2c >/dev/null 2>&1; then
            aria2c -x 16 -s 16 -k 1M -c \
                -d "$(dirname "${LPIPS_VGG16}")" \
                -o "$(basename "${LPIPS_VGG16}")" \
                "${url}"
        elif command -v wget >/dev/null 2>&1; then
            wget -c -O "${LPIPS_VGG16}" "${url}"
        else
            curl -L -C - -o "${LPIPS_VGG16}" "${url}"
        fi
    fi
    local sz
    sz="$(stat -c%s "${LPIPS_VGG16}" 2>/dev/null || echo 0)"
    if [[ "${sz}" -lt 500000000 ]]; then
        echo "ERROR: VGG16 backbone incomplete/missing: ${LPIPS_VGG16} (size=${sz})" >&2
        exit 1
    fi
    echo "[lpips] vgg16 backbone: ${LPIPS_VGG16} (${sz} bytes)"
    mkdir -p /root/.cache/torch/hub/checkpoints
    cp -f "${LPIPS_VGG16}" /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
}

ensure_lpips_weights

if [[ ! -f "${DINOV3_MODEL}" ]]; then
    echo "ERROR: DINOv3 weights not found: ${DINOV3_MODEL}" >&2
    exit 1
fi
echo "[dino] weights: ${DINOV3_MODEL}"

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
        echo "[resume] resuming lpips run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
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
    "train_cfg.lpips_loss_weight=${LPIPS_WEIGHT}"
    "train_cfg.dino_loss_weight=${DINO_WEIGHT}"
    "model.lpips.weights_path=${LPIPS_WEIGHTS}"
    "model.lpips.vgg_weights_path=${LPIPS_VGG16}"
    "model.dino_loss.checkpoint_path=${DINOV3_MODEL}"
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

echo "Launching EditFlow alpha+LPIPS+DINO DDP: nproc=${NUM_GPUS}  iters=${TOTAL_ITERS}  lpips_w=${LPIPS_WEIGHT}  dino_w=${DINO_WEIGHT}  load_from=${LOAD_FROM:-none}  resume_from=${RESUME_FROM:-none}  run=${RUN_NAME}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data_lpips.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
