#!/usr/bin/env bash
# EditFlow teacher-inverted path_epsilon (3 proj heads, no epsilon head):
#   Train: teacher 10-step invert x0_tgt -> path_epsilon; x_t = (1-s)*x0 + s*path_epsilon
#   Policy: pred_delta = mixture(deltax); student_u = path_epsilon - x_ref - pred_delta
#   Loss: MSE(student_u, teacher_u) via PIID
#   Infer: path_epsilon = sampled noise (fixed for trajectory)
#   deltax_init=kaiming (random weights; not zero / not inherit proj_out)
#   inherit_proj_out_deltax=false|true in user knobs below
#
#   bash train_flux_edit_teacherinv_nopol_data.sh              # default: 2 GPUs
#   bash train_flux_edit_teacherinv_nopol_data.sh 8
#   GPU_IDS=0,1,2,3,4,5,6,7 bash train_flux_edit_teacherinv_nopol_data.sh 8
#   TOTAL_ITERS=50000 bash train_flux_edit_teacherinv_nopol_data.sh

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
TEACHER_INVERT_STEPS=10
INHERIT_PROJ_OUT_DELTAX=false  # true: inherit FLUX proj_out -> deltax; false: kaiming random weights
# --------------------------------

RUN_NAME="gmkontext_uedit_teacherinv_nopol_k16_${NFE}nfe_pico400k"

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
    "train_cfg.teacher_invert_steps=${TEACHER_INVERT_STEPS}"
    "model.diffusion.denoising.inherit_proj_out_deltax=${INHERIT_PROJ_OUT_DELTAX}"
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

echo "Launching EditFlow teacher-inv nopol DDP: nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  teacher_invert_steps=${TEACHER_INVERT_STEPS}  inherit_proj_out_deltax=${INHERIT_PROJ_OUT_DELTAX}  run=${RUN_NAME}  ckpts=checkpoints/${RUN_NAME}/  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_uedit_teacherinv_nopol_2nfe_k16_data.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
