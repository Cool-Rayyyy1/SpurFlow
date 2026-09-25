gei#!/usr/bin/env bash
# Pure ArcFlow data-based distillation on Qwen-Image-Edit.
# Counterpart of train_flux_data.sh (Kontext ArcFlowImitation + ArcFlow policy):
#   x0 = edited target latent; source conditioning via image_latents only.
#   No uedit / fixedeps / alpha / GAN.
#
# Data mix (same knobs as train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh):
#   default 30% oss_edit + 70% pico-banana-400k.
#   OSS_PROB=0.5 PICO_PROB=0.5 bash train_qwen_data.sh
#
#   bash train_qwen_data.sh              # default: 8 GPUs, CUDA 0-7
#   bash train_qwen_data.sh 2            # override to 2 GPUs
#   GPU_IDS=0,1,2,3 bash train_qwen_data.sh 4
#   TOTAL_ITERS=20000 bash train_qwen_data.sh
#
# Resume (EMA included):
#   Ckpts live at checkpoints/<run_name>/<run_id>/{iter_*.pth,latest.pth}.
#   Auto-detects newest run_id with latest.pth; sets RESUME_RUN_DIR so work_dir matches.
#   Saved ckpts mirror diffusion_ema.* (trainable heads/LoRA); resume loads them.
#   RESUME_RUN_DIR=<run_id> bash train_qwen_data.sh
#   FRESH=1 bash train_qwen_data.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
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
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
TOTAL_ITERS="${TOTAL_ITERS:-10000}"
EVAL="${EVAL:-1}"
PICO_ROOT="${PICO_ROOT:-${DATA_ROOT:-/mnt/afs_gaochengmin/data/pico-banana-400k}}"
OSS_ROOT="${OSS_ROOT:-/mnt/afs_gaochengmin/data/oss_edit}"
OSS_JSONL="${OSS_JSONL:-${OSS_ROOT}/metadata.jsonl}"
OSS_PROB="${OSS_PROB:-0.3}"
PICO_PROB="${PICO_PROB:-0.7}"
QWEN_MODEL="${QWEN_MODEL:-/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
# --------------------------------

oss_pct="$(awk -v p="${OSS_PROB}" 'BEGIN{printf "%d", p*100+0.5}')"
pico_pct="$(awk -v p="${PICO_PROB}" 'BEGIN{printf "%d", p*100+0.5}')"
RUN_NAME="gmqwen_k16_${NFE}nfe_oss${oss_pct}_pico${pico_pct}_data"
CONFIG="${PROJECT_DIR}/configs/qwen/editqwen_2nfe_k16_data_oss_pico.py"

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

# ---------- resume resolution ----------
# mmcv CheckpointHook saves to: checkpoints/<RUN_NAME>/<run_id>/
# (out_dir + basename(work_dir)). latest.pth is a symlink to the newest iter_*.pth.
# ckpt_trainable_only=True still mirrors diffusion_ema.* via append_ema_mirrors_to_state_dict;
# resume loads those keys (falls back to promoting online weights only if EMA keys are missing).
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
        echo "[resume] resolved -> $(readlink -f "${PROJECT_DIR}/${RESUME_FROM}" 2>/dev/null || realpath "${PROJECT_DIR}/${RESUME_FROM}")"
    elif [[ -n "${RESUME_RUN_ID}" ]]; then
        echo "[resume] RESUME_RUN_DIR=${RESUME_RUN_ID} set but missing ${CKPT_BASE}/${RESUME_RUN_ID}/latest.pth" >&2
        echo "[resume] Refuse to start a new timestamped run under that run_id without a ckpt. Unset RESUME_RUN_DIR or set FRESH=1." >&2
        exit 1
    fi
fi
if [[ -z "${RESUME_FROM}" ]]; then
    echo "[resume] no checkpoint to resume (FRESH=${FRESH}); starting a new run"
    # Avoid exporting an empty RESUME_RUN_DIR that could confuse tooling; only pass when set.
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
    "checkpoint_config.out_dir=checkpoints/${RUN_NAME}"
    "train_cfg.nfe=${NFE}"
    "test_cfg.nfe=${NFE}"
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

echo "Launching EditFlow Qwen vanilla-ArcFlow data FSDP (oss ${OSS_PROB} / pico ${PICO_PROB}): nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  oss_root=${OSS_ROOT}  pico_root=${PICO_ROOT}  run=${RUN_NAME}  resume_from=${RESUME_FROM:-none}  resume_run_dir=${RESUME_RUN_DIR:-new}  ckpt_out=checkpoints/${RUN_NAME}/<run_id>/  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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
