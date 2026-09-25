#!/usr/bin/env bash
# Warmup for FLUX.1 Kontext EditFlow. No GAN.
#
# Student is a 2-NFE conditional editor. A sigmoid alpha head mixes the
# reference latent into the predicted velocity:
#   alpha = sigmoid(raw head), one channel per 2x2 patch
#   student_u = path_epsilon - alpha * x_ref - pred_delta
# The teacher is the frozen FLUX.1 Kontext backbone with distilled CFG.
# Weights start from that backbone plus randomly initialized alpha and delta
# heads. The checkpoint from this run is the init for train_flux_kontext.sh.
#
# Required:
#   KONTEXT_MODEL   local FLUX.1-Kontext-dev directory
#   PICO_ROOT       paired edit dataset root
#   OSS_ROOT        second paired edit dataset root (metadata.jsonl, or built)
# Optional:
#   OSS_PROB=0.3 PICO_PROB=0.7 NFE=2 SHIFT=3.2 TOTAL_ITERS=50000
#   GPU_IDS=0,1,2,3,4,5,6,7 NUM_GPUS=8
#   PRETRAIN_CKPT=path/to/student.pth   # continue from an existing student
#   FRESH=1                             # ignore checkpoints on disk
#   RESUME_RUN_DIR=<id> RESUME_ITER=<n>
#
#   bash train_flux_kontext_warmup.sh
#   bash train_flux_kontext_warmup.sh 8

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
SHIFT="${SHIFT:-3.2}"
CKPT_INTERVAL="${CKPT_INTERVAL:-500}"
CKPT_MUST_SAVE_INTERVAL="${CKPT_MUST_SAVE_INTERVAL:-1000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-500}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-2}"
TOTAL_ITERS="${TOTAL_ITERS:-50000}"
EVAL="${EVAL:-1}"
PICO_ROOT="${PICO_ROOT:-${DATA_ROOT:-}}"
OSS_ROOT="${OSS_ROOT:-}"
OSS_JSONL="${OSS_JSONL:-${OSS_ROOT}/metadata.jsonl}"
OSS_PROB="${OSS_PROB:-0.3}"
PICO_PROB="${PICO_PROB:-0.7}"
KONTEXT_MODEL="${KONTEXT_MODEL:-}"
GEDIT_META="${GEDIT_META:-}"
GEDIT_ROOT="${GEDIT_ROOT:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
RESUME_ITER="${RESUME_ITER:-}"
FRESH="${FRESH:-0}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
# --------------------------------

if [[ -z "${KONTEXT_MODEL}" || -z "${PICO_ROOT}" || -z "${OSS_ROOT}" ]]; then
    echo "Set KONTEXT_MODEL, PICO_ROOT, and OSS_ROOT." >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-flux_kontext_warmup}"
CONFIG="${PROJECT_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data_oss_pico.py"

if [[ ! -f "${OSS_JSONL}" ]]; then
    if [[ "${RANK:-0}" == "0" ]]; then
        echo "[data] building ${OSS_JSONL}"
        python "${PROJECT_DIR}/tools/build_oss_edit_jsonl.py" --root "${OSS_ROOT}" --out "${OSS_JSONL}"
    else
        echo "[data] waiting for ${OSS_JSONL}"
        for _ in $(seq 1 120); do
            [[ -f "${OSS_JSONL}" ]] && break
            sleep 5
        done
        [[ -f "${OSS_JSONL}" ]] || { echo "[data] timeout waiting for ${OSS_JSONL}" >&2; exit 1; }
    fi
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL}"
export PICO_BANANA_PATH="${PICO_ROOT}"

CKPT_BASE="checkpoints/${RUN_NAME}"
LOAD_FROM=""
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
    if [[ -n "${RESUME_RUN_ID}" ]]; then
        if [[ -n "${RESUME_ITER}" ]]; then
            _resume_ckpt="${CKPT_BASE}/${RESUME_RUN_ID}/iter_${RESUME_ITER}.pth"
        else
            _resume_ckpt="${CKPT_BASE}/${RESUME_RUN_ID}/latest.pth"
        fi
        if [[ -e "${PROJECT_DIR}/${_resume_ckpt}" ]]; then
            RESUME_FROM="${_resume_ckpt}"
            RESUME_RUN_DIR="${RESUME_RUN_ID}"
            echo "[resume] run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
        else
            echo "[resume] checkpoint not found: ${PROJECT_DIR}/${_resume_ckpt}" >&2
            exit 1
        fi
    fi
fi
if [[ -z "${RESUME_FROM}" && -n "${PRETRAIN_CKPT}" ]]; then
    if [[ "${PRETRAIN_CKPT}" = /* ]]; then
        _pretrain_path="${PRETRAIN_CKPT}"
    else
        _pretrain_path="${PROJECT_DIR}/${PRETRAIN_CKPT}"
    fi
    if [[ -e "${_pretrain_path}" ]]; then
        LOAD_FROM="${_pretrain_path}"
        echo "[pretrain] loading warmup student from ${LOAD_FROM}"
    else
        echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT})" >&2
        exit 1
    fi
elif [[ -z "${RESUME_FROM}" ]]; then
    echo "[init] FLUX.1 Kontext backbone, random alpha and delta heads"
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
    "model.diffusion.timestep_sampler.shift=${SHIFT}"
    "checkpoint_config.interval=${CKPT_INTERVAL}"
    "checkpoint_config.must_save_interval=${CKPT_MUST_SAVE_INTERVAL}"
    "sample_eval.interval=${SAMPLE_INTERVAL}"
    "sample_eval.must_save_interval=0"
    "sample_eval.dataset.samples_per_category=${SAMPLES_PER_CATEGORY}"
    "total_iters=${TOTAL_ITERS}"
    "data.train.probs=[${OSS_PROB},${PICO_PROB}]"
    "data.train.datasets.0.data_root=${OSS_ROOT}"
    "data.train.datasets.0.jsonl_path=${OSS_JSONL}"
    "data.train.datasets.1.data_root=${PICO_ROOT}"
    "data.val.data_root=${PICO_ROOT}"
    "model.vae.from_pretrained=${KONTEXT_MODEL}"
    "model.text_encoder.from_pretrained=${KONTEXT_MODEL}"
    "model.diffusion.denoising.pretrained=${KONTEXT_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
    "model.teacher.denoising.pretrained=${KONTEXT_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
)

if [[ -n "${GEDIT_META}" ]]; then
    CFG_OPTS+=("sample_eval.dataset.annotations_path=${GEDIT_META}")
fi
if [[ -n "${GEDIT_ROOT}" ]]; then
    CFG_OPTS+=("sample_eval.dataset.bench_root=${GEDIT_ROOT}")
fi

if [[ "${EVAL}" == "1" ]]; then
    CFG_OPTS+=("sample_eval.enabled=true")
else
    CFG_OPTS+=("sample_eval.enabled=false")
fi

echo "Warmup FLUX.1 Kontext (no GAN, mix ${OSS_PROB}/${PICO_PROB}): nproc=${NUM_GPUS} nfe=${NFE} shift=${SHIFT} load_from=${LOAD_FROM:-none} resume_from=${RESUME_FROM:-none} total_iters=${TOTAL_ITERS} run=${RUN_NAME}"

torchrun \
    --nnodes="${WORLD_SIZE:-1}" \
    --nproc_per_node="${NUM_GPUS}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29500}" \
    "${PROJECT_DIR}/train.py" \
    "${CONFIG}" \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"
