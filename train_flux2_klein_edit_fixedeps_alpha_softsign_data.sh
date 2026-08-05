#!/usr/bin/env bash
# EditFlow: distill FLUX.2-klein-base-9B -> 2-NFE Softsign-01 alpha student (no GAN).
# Goal: beat official step-distilled FLUX.2-klein-9B with our 2-step student.
#
# Softsign-01:
#   alpha = 0.5 * (raw / (1 + abs(raw)) + 1)
#   student_u = path_epsilon - alpha * x_ref - pred_delta
#
# Guidance:
#   Teacher (klein-base): official CFG=4.0
#     Flux2KleinPipeline runs cond DiT forward + uncond DiT forward, then combines.
#     (guidance_embeds=false — NOT Flux2-dev / Kontext single-pass embeds)
#   Student: CFG=1 / cond-only — one DiT forward, no uncond.
#
# Data: resize_mode=flux2 matches Flux2KleinPipeline
#   (area-cap ~1MP + multiple-of-16; NOT Kontext buckets).
#
# Uses FSDP (configs/flux2_klein/_fsdp_train.py). Prefer >=2 GPUs; 1x80GB
# often OOMs because teacher CFG=4 doubles the teacher forward.
#
# SHIFT: flow-matching timestep sampler shift (train/test schedule), NOT related
# to validation images. Validation sample_eval dumps ImgEdit-Bench 9 cats x 5 imgs.
#
#   bash train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh
#   bash train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh 8
#   TEACHER_CFG=4.0 GPU_IDS=0,1,2,3 bash train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh 4

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
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-100}"
TOTAL_ITERS="${TOTAL_ITERS:-20000}"
EVAL="${EVAL:-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
KLEIN_MODEL="${KLEIN_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.2-klein-base-9B}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
# --------------------------------

RUN_NAME="gmklein_base_uedit_fixedeps_alpha_softsign01_k16_${NFE}nfe_shift${SHIFT}_teachercfg${TEACHER_CFG}_pico400k"
CONFIG="${PROJECT_DIR}/configs/flux2_klein/editflux2_klein_uedit_fixedeps_2nfe_k16_alpha_softsign_data.py"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export KLEIN_MODEL_PATH="${KLEIN_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_BASE="checkpoints/${RUN_NAME}"
RESUME_FROM=""
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
        echo "[resume] resuming run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
    fi
fi
if [[ -z "${RESUME_FROM}" ]]; then
    echo "[resume] no checkpoint to resume (FRESH=${FRESH}); starting a new run"
fi

export RUN_ID
export RESUME_RUN_DIR

CFG_OPTS=(
    "name=${RUN_NAME}"
    "work_dir=work_dirs/${RUN_NAME}"
    "resume_from=${RESUME_FROM}"
    "checkpoint_config.out_dir=checkpoints/${RUN_NAME}"
    "train_cfg.nfe=${NFE}"
    "test_cfg.nfe=${NFE}"
    "train_cfg.teacher_guidance_scale=${TEACHER_CFG}"
    "test_cfg.guidance_scale=1.0"
    "model.diffusion.timestep_sampler.shift=${SHIFT}"
    "checkpoint_config.interval=${CKPT_INTERVAL}"
    "checkpoint_config.must_save_interval=${CKPT_MUST_SAVE_INTERVAL}"
    "sample_eval.interval=${SAMPLE_INTERVAL}"
    "sample_eval.must_save_interval=0"
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

echo "Launching EditFlow FLUX.2-klein-base Softsign-01 alpha DDP: nproc=${NUM_GPUS} nfe=${NFE} shift=${SHIFT} teacher_cfg=${TEACHER_CFG} student_cfg=1.0 total_iters=${TOTAL_ITERS} run=${RUN_NAME} model=${KLEIN_MODEL} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    "${CONFIG}" \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
