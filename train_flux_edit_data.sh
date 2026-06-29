#!/usr/bin/env bash
# EditFlow uedit variant: data-dependent distillation with 4 proj heads
# (deltax, epsilon, logweights, loggamma).
#   uedit = mixture_k(deltax_k - epsilon_k; weights, gammas)
#   u = uedit + x_ref   matched to teacher u = epsilon - x_0
#   integration: x_new = x_old - u * dt
#
#   bash train_flux_edit_data.sh              # default: 2 GPUs (dev machine)
#   bash train_flux_edit_data.sh 8            # override to 8 GPUs on server
#   GPU_IDS=0,1,2,3,4,5,6,7 bash train_flux_edit_data.sh 8
#   TOTAL_ITERS=50000 bash train_flux_edit_data.sh

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
TOTAL_ITERS="${TOTAL_ITERS:-10000}"
EVAL="${EVAL:-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
GPU_IDS="${GPU_IDS:-0,1}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
# --------------------------------

RUN_NAME="gmkontext_uedit_k16_${NFE}nfe_pico400k"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"
export RUN_ID
export RESUME_RUN_DIR

CFG_OPTS=(
    "name=${RUN_NAME}"
    "work_dir=work_dirs/${RUN_NAME}"
    "resume_from=checkpoints/${RUN_NAME}/latest.pth"
    "checkpoint_config.out_dir=checkpoints/${RUN_NAME}"
    "train_cfg.nfe=${NFE}"
    "test_cfg.nfe=${NFE}"
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

echo "Launching EditFlow uedit data DDP: nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  run=${RUN_NAME}  ckpts=checkpoints/${RUN_NAME}/  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_uedit_2nfe_k16_data.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
