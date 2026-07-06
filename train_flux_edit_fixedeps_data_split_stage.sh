#!/usr/bin/env bash
# EditFlow fixed-eps data-dependent split-stage distillation (3 proj heads, no epsilon head):
#   Same student / teacher / data / LoRA as train_flux_edit_fixedeps_data.sh, but:
#   - data-dependent: each batch samples path_epsilon and uses real edited x0_tgt
#   - student does a full 2-step rollout from the sampled noise (nfe=2)
#   - step-1 (t=1): PIID teacher-alignment loss only
#   - step-2 (student step-1 endpoint):
#       split_stage_teacher_loss_weight * PIID loss
#       + split_stage_diffusion_loss_weight * direct flow MSE (path_epsilon - x0_tgt)
#       with x_ref scaled by split_stage_step2_x_ref_scale (default 1.0)
#   Step-2 gradients backpropagate into step-1. Defaults: 0.5 / 0.5 teacher/direct.
#   SPLIT_STAGE_STEP2_WARMUP_ITERS (default 2000): ramp step-2 loss from 0; step-1 only first.
#
#   bash train_flux_edit_fixedeps_data_split_stage.sh              # default: 2 GPUs
#   bash train_flux_edit_fixedeps_data_split_stage.sh 8
#   GPU_IDS=0,1,2,3,4,5,6,7 bash train_flux_edit_fixedeps_data_split_stage.sh 8
#   TOTAL_ITERS=50000 bash train_flux_edit_fixedeps_data_split_stage.sh
#
# Resume:
#   RESUME_RUN_DIR=<run_id> bash train_flux_edit_fixedeps_data_split_stage.sh
#   FRESH=1 bash train_flux_edit_fixedeps_data_split_stage.sh

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
SPLIT_STAGE_DIFFUSION_WEIGHT="${SPLIT_STAGE_DIFFUSION_WEIGHT:-0.5}"
SPLIT_STAGE_TEACHER_WEIGHT="${SPLIT_STAGE_TEACHER_WEIGHT:-0.5}"
SPLIT_STAGE_STEP2_X_REF_SCALE="${SPLIT_STAGE_STEP2_X_REF_SCALE:-1.0}"
SPLIT_STAGE_STEP2_WARMUP_ITERS="${SPLIT_STAGE_STEP2_WARMUP_ITERS:-2000}"
# --------------------------------

RUN_NAME="gmkontext_uedit_fixedeps_k16_${NFE}nfe_pico400k_split_stage"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"

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
    "train_cfg.split_stage_diffusion_loss_weight=${SPLIT_STAGE_DIFFUSION_WEIGHT}"
    "train_cfg.split_stage_teacher_loss_weight=${SPLIT_STAGE_TEACHER_WEIGHT}"
    "train_cfg.split_stage_step2_x_ref_scale=${SPLIT_STAGE_STEP2_X_REF_SCALE}"
    "test_cfg.split_stage_step2_x_ref_scale=${SPLIT_STAGE_STEP2_X_REF_SCALE}"
    "train_cfg.split_stage_step2_warmup_iters=${SPLIT_STAGE_STEP2_WARMUP_ITERS}"
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

echo "Launching EditFlow fixed-eps data-dependent split-stage DDP: nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  split_stage=${SPLIT_STAGE_DIFFUSION_WEIGHT}/${SPLIT_STAGE_TEACHER_WEIGHT}  step2_x_ref_scale=${SPLIT_STAGE_STEP2_X_REF_SCALE}  step2_warmup_iters=${SPLIT_STAGE_STEP2_WARMUP_ITERS}  run=${RUN_NAME}  ckpts=checkpoints/${RUN_NAME}/  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_split_stage.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
