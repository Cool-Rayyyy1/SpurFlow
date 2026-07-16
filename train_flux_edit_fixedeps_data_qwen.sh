#!/usr/bin/env bash
# EditFlow fixed path epsilon on Qwen-Image-Edit-2511 (3 proj heads, no epsilon head):
#   pred_delta = mixture(deltax_k; weights, gammas) ~ x0_tgt - x_ref
#   student_u = path_epsilon - x_ref - pred_delta
#   teacher_u = path_epsilon - x0_tgt  =>  loss aligns pred_delta to edit residual
#   DiT backbone from Qwen-Image-Edit; deltax_init=kaiming on proj_out_deltax (not Qwen proj_out).
#   inherit_proj_out_deltax=False — deltax does NOT copy proj_out.
#
#   Sample eval: ImgEdit-Bench 9 categories x 5 (same as train_flux_edit_fixedeps_data_step2_alph_dino_gan.sh).
#
#   bash train_flux_edit_fixedeps_data_qwen.sh              # default: 2 GPUs
#   bash train_flux_edit_fixedeps_data_qwen.sh 8
#   GPU_IDS=0,1,2,3,4,5,6,7 bash train_flux_edit_fixedeps_data_qwen.sh 8
#   TOTAL_ITERS=80000 bash train_flux_edit_fixedeps_data_qwen.sh
#
# Resume:
#   Auto-detects the most recent run folder under checkpoints/gmqwen_uedit_fixedeps_k16_*.
#   RESUME_RUN_DIR=<run_id> bash train_flux_edit_fixedeps_data_qwen.sh   # resume another run
#   FRESH=1 bash train_flux_edit_fixedeps_data_qwen.sh                   # ignore checkpoints, start new run

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
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
TOTAL_ITERS="${TOTAL_ITERS:-50000}"
EVAL="${EVAL:-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
QWEN_MODEL="${QWEN_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/Qwen-Image-Edit-2511}"
GPU_IDS="${GPU_IDS:-0,1}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
# --------------------------------

RUN_NAME="gmqwen_uedit_fixedeps_k16_${NFE}nfe_pico400k"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export TOKENIZERS_PARALLELISM=false
export QWEN_MODEL_PATH="${QWEN_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"

# ---------- resume resolution ----------
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
# ----------------------------------------

export RUN_ID
export RESUME_RUN_DIR

CFG_OPTS=(
    "name=${RUN_NAME}"
    "work_dir=work_dirs/${RUN_NAME}"
    "resume_from=${RESUME_FROM}"
    "checkpoint_config.out_dir=checkpoints/${RUN_NAME}"
    "train_cfg.nfe=${NFE}"
    "test_cfg.nfe=${NFE}"
    "checkpoint_config.interval=${CKPT_INTERVAL}"
    "checkpoint_config.must_save_interval=${CKPT_MUST_SAVE_INTERVAL}"
    "sample_eval.interval=${SAMPLE_INTERVAL}"
    "sample_eval.must_save_interval=0"
    "sample_eval.dataset.samples_per_category=${SAMPLES_PER_CATEGORY}"
    "total_iters=${TOTAL_ITERS}"
    "data.train.data_root=${DATA_ROOT}"
    "data.val.data_root=${DATA_ROOT}"
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

echo "Launching EditFlow Qwen uedit fixed-eps FSDP: nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  sample_interval=${SAMPLE_INTERVAL}  samples_per_category=${SAMPLES_PER_CATEGORY}  run=${RUN_NAME}  ckpts=checkpoints/${RUN_NAME}/  model=${QWEN_MODEL}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/qwen/editqwen_uedit_fixedeps_2nfe_k16_data.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"