#!/usr/bin/env bash
# EditFlow: FLUX Kontext data-based distillation on pico-banana-400k.
# x0 = edited target image latent; forward-diffuse + noise for ArcFlowImitation.
#
#   bash train_flux_data.sh              # default: 8 GPUs, CUDA 0-7
#   bash train_flux_data.sh 2            # override to 2 GPUs
#   TOTAL_ITERS=20000 bash train_flux_data.sh
#
# Resume the existing 10k run to 20k (same work_dir + ckpt dir):
#   TOTAL_ITERS=20000 RESUME_RUN_DIR=20260728_220103 bash train_flux_data.sh
# Or auto-pick newest run with latest.pth:
#   TOTAL_ITERS=20000 bash train_flux_data.sh
# Fresh run (ignore existing ckpts):
#   FRESH=1 TOTAL_ITERS=10000 bash train_flux_data.sh

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
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-100}"
TOTAL_ITERS="${TOTAL_ITERS:-10000}"
EVAL="${EVAL:-1}"
FRESH="${FRESH:-0}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
# --------------------------------

RUN_NAME="gmkontext_k16_${NFE}nfe_pico400k_data"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${DATA_ROOT}"

# ---------- resume resolution ----------
# CheckpointHook with out_dir='checkpoints/' writes:
#   checkpoints/<run_id>/iter_XXXX.pth   (+ latest.pth symlink)
# (basename(work_dir) == run_id). Older/wrong layout also tried:
#   checkpoints/<RUN_NAME>/<run_id>/
has_latest() { [[ -e "${1}/latest.pth" ]]; }

RESUME_FROM=""
if [[ "${FRESH}" != "1" ]]; then
    RESUME_RUN_ID="${RESUME_RUN_DIR:-${RUN_ID:-}}"
    if [[ -z "${RESUME_RUN_ID}" ]]; then
        # Prefer newest run_id under checkpoints/ that has latest.pth
        # (skip named experiment dirs that are not timestamp-like run folders).
        for _cand in $(ls -1dt "${PROJECT_DIR}/checkpoints"/*/ 2>/dev/null); do
            _base="$(basename "${_cand}")"
            if [[ "${_base}" == "${RUN_NAME}" ]]; then
                for _sub in $(ls -1dt "${_cand}"*/ 2>/dev/null); do
                    if has_latest "${_sub}"; then
                        RESUME_RUN_ID="$(basename "${_sub}")"
                        RESUME_FROM="checkpoints/${RUN_NAME}/${RESUME_RUN_ID}/latest.pth"
                        break 2
                    fi
                done
                continue
            fi
            if has_latest "${_cand}"; then
                RESUME_RUN_ID="${_base}"
                RESUME_FROM="checkpoints/${RESUME_RUN_ID}/latest.pth"
                break
            fi
        done
    else
        if has_latest "${PROJECT_DIR}/checkpoints/${RESUME_RUN_ID}"; then
            RESUME_FROM="checkpoints/${RESUME_RUN_ID}/latest.pth"
        elif has_latest "${PROJECT_DIR}/checkpoints/${RUN_NAME}/${RESUME_RUN_ID}"; then
            RESUME_FROM="checkpoints/${RUN_NAME}/${RESUME_RUN_ID}/latest.pth"
        else
            echo "[resume] RESUME_RUN_DIR=${RESUME_RUN_ID} set but missing latest.pth under" >&2
            echo "[resume]   checkpoints/${RESUME_RUN_ID}/  or  checkpoints/${RUN_NAME}/${RESUME_RUN_ID}/" >&2
            echo "[resume] Refuse to start. Unset RESUME_RUN_DIR or set FRESH=1." >&2
            exit 1
        fi
    fi
    if [[ -n "${RESUME_FROM}" ]]; then
        RESUME_RUN_DIR="${RESUME_RUN_ID}"
        echo "[resume] resuming run_id=${RESUME_RUN_DIR} from ${RESUME_FROM}"
        echo "[resume] resolved -> $(readlink -f "${PROJECT_DIR}/${RESUME_FROM}" 2>/dev/null || realpath "${PROJECT_DIR}/${RESUME_FROM}")"
    fi
fi
if [[ -z "${RESUME_FROM}" ]]; then
    echo "[resume] no checkpoint to resume (FRESH=${FRESH}); starting a new run"
    unset RESUME_RUN_DIR
fi
# ----------------------------------------

export RUN_ID
if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
    export RESUME_RUN_DIR
fi

CFG_OPTS=(
    "name=${RUN_NAME}"
    "work_dir=work_dirs/${RUN_NAME}"
    "load_from="
    "resume_from=${RESUME_FROM}"
    "checkpoint_config.out_dir=checkpoints/"
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

echo "Launching EditFlow data DDP: nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  run=${RUN_NAME}  resume_from=${RESUME_FROM:-none}  resume_run_dir=${RESUME_RUN_DIR:-new}  ckpt_out=checkpoints/<run_id>/  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
    configs/kontext/editflux_2nfe_k16_data.py \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
