#!/usr/bin/env bash
# EditFlow fixed-eps split-stage + dual LoRA + step-2 teacher-x0 alignment:
#   Step-1 (step1 LoRA): same as train_flux_edit_fixedeps_data_split_stage.sh
#     — PIID from sampled path_epsilon to produce an intermediate x_t.
#   Step-2 (step2 LoRA):
#     teacher_u = teacher velocity at x_t
#     teacher_x0 = path_epsilon - teacher_u
#     student_x0 = x_ref + pred_delta
#     loss = MSE(student_x0, teacher_x0)
#   Validation: step-1 integrates to x_t; step-2 returns x_ref + delta directly.
#   Does not modify the original split_stage / dual_lora_dino_gan scripts.
#
#   bash train_flux_edit_fixedeps_data_split_stage_dual_lora_teacher_x0.sh
#   bash train_flux_edit_fixedeps_data_split_stage_dual_lora_teacher_x0.sh 8
#   GPU_IDS=0 bash train_flux_edit_fixedeps_data_split_stage_dual_lora_teacher_x0.sh 1
#   FRESH=1 bash train_flux_edit_fixedeps_data_split_stage_dual_lora_teacher_x0.sh
#
# Optional pretrain (single LoRA is auto-copied to step1/step2 at load):
#   PRETRAIN_CKPT=checkpoints/.../iter_20000.pth bash train_flux_edit_fixedeps_data_split_stage_dual_lora_teacher_x0.sh

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
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
SPLIT_STAGE_STEP2_X0_WEIGHT="${SPLIT_STAGE_STEP2_X0_WEIGHT:-1.0}"
SPLIT_STAGE_STEP2_WARMUP_ITERS="${SPLIT_STAGE_STEP2_WARMUP_ITERS:-2000}"
NUM_DECAY_ITERS="${NUM_DECAY_ITERS:-2000}"
# --------------------------------

RUN_NAME="gmkontext_uedit_fixedeps_k16_${NFE}nfe_pico400k_split_stage_dual_lora_teacher_x0"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"

CKPT_BASE="checkpoints/${RUN_NAME}"
LOAD_FROM=""
RESUME_FROM=""
if [[ -n "${PRETRAIN_CKPT}" && -e "${PROJECT_DIR}/${PRETRAIN_CKPT}" ]]; then
    LOAD_FROM="${PRETRAIN_CKPT}"
    echo "[pretrain] loading student weights from ${LOAD_FROM}"
    echo "[pretrain] single LoRA + heads will be copied to dual step1/step2 at load"
elif [[ -n "${PRETRAIN_CKPT}" ]]; then
    echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT}); starting without pretrain" >&2
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
        echo "[resume] resuming dual-lora teacher-x0 run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
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
    "train_cfg.split_stage_step2_x0_loss_weight=${SPLIT_STAGE_STEP2_X0_WEIGHT}"
    "train_cfg.split_stage_step2_warmup_iters=${SPLIT_STAGE_STEP2_WARMUP_ITERS}"
    "train_cfg.num_decay_iters=${NUM_DECAY_ITERS}"
    "model.diffusion.denoising.dual_stage_lora=True"
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

echo "Launching EditFlow split-stage dual-LoRA teacher-x0 DDP: nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  step2_x0_w=${SPLIT_STAGE_STEP2_X0_WEIGHT}  step2_warmup=${SPLIT_STAGE_STEP2_WARMUP_ITERS}  load_from=${LOAD_FROM:-none}  resume_from=${RESUME_FROM:-none}  run=${RUN_NAME}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_split_stage_dual_lora_teacher_x0.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
