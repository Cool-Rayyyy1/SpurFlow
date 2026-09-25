#!/usr/bin/env bash
# EditFlow Qwen-Image-Edit Softsign-01 alpha, non-split PIID (no GAN).
#   alpha = 0.5 * (raw / (1 + abs(raw)) + 1)
#   student_u = path_epsilon - alpha * x_ref - pred_delta
#   student: cond-only 2-NFE
#   teacher: CFG=4.0 + Qwen norm rescale
#
# Init: Qwen-Image-Edit-2511 transformer + randomly initialized alpha/delta heads.
# Optional EditFlow student: PRETRAIN_CKPT=pretrain/qwen/gan/iter_2800.pth
# Data: 30% oss / 70% pico.
#
#   bash train_flux_edit_fixedeps_alpha_data_qwen.sh
#   bash train_flux_edit_fixedeps_alpha_data_qwen.sh 8
#   GPU_IDS=0,1,2,3,4,5,6,7 bash train_flux_edit_fixedeps_alpha_data_qwen.sh 8
#   FRESH=1 bash train_flux_edit_fixedeps_alpha_data_qwen.sh
#   RESUME_RUN_DIR=20260819_150749 RESUME_ITER=7000 bash train_flux_edit_fixedeps_alpha_data_qwen.sh

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
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-500}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-2}"
TOTAL_ITERS="${TOTAL_ITERS:-50000}"
EVAL="${EVAL:-1}"
PICO_ROOT="${PICO_ROOT:-/mnt/afs_gaochengmin/data/pico-banana-400k}"
OSS_ROOT="${OSS_ROOT:-/mnt/afs_gaochengmin/data/oss_edit}"
OSS_JSONL="${OSS_JSONL:-${OSS_ROOT}/metadata.jsonl}"
OSS_PROB="${OSS_PROB:-0.3}"
PICO_PROB="${PICO_PROB:-0.7}"
QWEN_MODEL="${QWEN_MODEL:-/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
RESUME_ITER="${RESUME_ITER:-}"
FRESH="${FRESH:-0}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
# --------------------------------

RUN_NAME="gmqwen_uedit_fixedeps_alpha_softsign01_k16_${NFE}nfe_oss30_pico70"
CONFIG="${PROJECT_DIR}/configs/qwen/editqwen_uedit_fixedeps_2nfe_k16_alpha_softsign_data_oss_pico.py"

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
export TOKENIZERS_PARALLELISM=false
export QWEN_MODEL_PATH="${QWEN_MODEL}"
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
            echo "[resume] resuming run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
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
        echo "[pretrain] loading Softsign-01 student from ${LOAD_FROM}"
    else
        echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT})" >&2
        exit 1
    fi
elif [[ -z "${RESUME_FROM}" ]]; then
    echo "[pretrain] no PRETRAIN_CKPT; init from Qwen transformer + random alpha heads"
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
    "checkpoint_config.interval=${CKPT_INTERVAL}"
    "checkpoint_config.must_save_interval=${CKPT_MUST_SAVE_INTERVAL}"
    "sample_eval.interval=${SAMPLE_INTERVAL}"
    "sample_eval.must_save_interval=0"
    "sample_eval.dataset.samples_per_category=${SAMPLES_PER_CATEGORY}"
    "sample_eval.dataset.annotations_path=/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json"
    "sample_eval.dataset.bench_root=/mnt/afs_caiqi/data/benchmark/GEdit_v2"
    "total_iters=${TOTAL_ITERS}"
    "data.train.probs=[${OSS_PROB},${PICO_PROB}]"
    "data.train.datasets.0.data_root=${OSS_ROOT}"
    "data.train.datasets.0.jsonl_path=${OSS_JSONL}"
    "data.train.datasets.1.data_root=${PICO_ROOT}"
    "data.val.data_root=${PICO_ROOT}"
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

echo "Launching EditFlow Qwen Softsign-01 alpha (no GAN, oss ${OSS_PROB} / pico ${PICO_PROB}): nproc=${NUM_GPUS} nfe=${NFE} load_from=${LOAD_FROM:-none} resume_from=${RESUME_FROM:-none} total_iters=${TOTAL_ITERS} run=${RUN_NAME} model=${QWEN_MODEL} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
#     "${CONFIG}" \
#     --launcher pytorch --diff_seed \
#     --cfg-options "${CFG_OPTS[@]}"


torchrun \
    --nnodes="${WORLD_SIZE}" \
    --nproc_per_node=8 \
    --node_rank="${RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${PROJECT_DIR}/train.py" \
    "${CONFIG}" \
    --launcher pytorch --diff_seed \
    --cfg-options "${CFG_OPTS[@]}"

