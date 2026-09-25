#!/usr/bin/env bash
# EditFlow fixed-eps PIID + step-2 TDM-style DINO feature GAN:
#   70% oss_edit banana-task pairs + 30% pico-banana-400k.
#   OssEdit uses the same Kontext resolution buckets as ImageEdit.
#
#   bash train_flux_edit_fixedeps_data_step2_dino_gan_plus_oss.sh
#   PRETRAIN_CKPT=pretrain/flux_kontext/nogan/iter_20000.pth \
#     bash train_flux_edit_fixedeps_data_step2_dino_gan_plus_oss.sh
#   FRESH=1 bash train_flux_edit_fixedeps_data_step2_dino_gan_plus_oss.sh

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
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
TOTAL_ITERS="${TOTAL_ITERS:-25000}"
EVAL="${EVAL:-1}"
PICO_ROOT="${PICO_ROOT:-/mnt/afs_gaochengmin/data/pico-banana-400k}"
OSS_ROOT="${OSS_ROOT:-/mnt/afs_gaochengmin/data/oss_edit}"
OSS_JSONL="${OSS_JSONL:-${OSS_ROOT}/metadata.jsonl}"
OSS_PROB="${OSS_PROB:-0.7}"
PICO_PROB="${PICO_PROB:-0.3}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev}"
DINOV3_MODEL="${DINOV3_MODEL:-/mnt/afs_gaochengmin/checkpoints/dinov3/dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/model/final/gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_step2_alph_dino_gan_new/model/final/gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_step2_alph_dino_gan_new/iter_7000.pth}"
STEP2_GAN_WARMUP_ITERS="${STEP2_GAN_WARMUP_ITERS:-0}"
STEP2_GAN_RAMP_ITERS="${STEP2_GAN_RAMP_ITERS:-0}"
STEP2_GAN_WEIGHT="${STEP2_GAN_WEIGHT:-0.05}"
GAN_GRAD_STEP2_ONLY="${GAN_GRAD_STEP2_ONLY:-true}"
NUM_DECAY_ITERS="${NUM_DECAY_ITERS:-0}"
DINO_GLOBAL_SIZE="${DINO_GLOBAL_SIZE:-224}"
DINO_LOCAL_SIZE="${DINO_LOCAL_SIZE:-224}"
DINO_FEATURE_LAYERS="${DINO_FEATURE_LAYERS:-23}"
# --------------------------------

RUN_NAME="gmkontext_uedit_fixedeps_k16_${NFE}nfe_oss70_pico30_step2_dino_gan"
CONFIG="${PROJECT_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_step2_dino_gan_oss_pico.py"

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
    if [[ -n "${PRETRAIN_CKPT}" && -e "${PROJECT_DIR}/${PRETRAIN_CKPT}" ]]; then
        LOAD_FROM="${PRETRAIN_CKPT}"
        echo "[pretrain] loading student weights from ${LOAD_FROM}"
    else
        echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT}); starting from scratch" >&2
    fi

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
        echo "[resume] resuming step2-dino-gan run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
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
    "train_cfg.split_stage_gan_warmup_iters=${STEP2_GAN_WARMUP_ITERS}"
    "train_cfg.split_stage_gan_ramp_iters=${STEP2_GAN_RAMP_ITERS}"
    "train_cfg.split_stage_gan_loss_weight=${STEP2_GAN_WEIGHT}"
    "train_cfg.gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY}"
    "train_cfg.num_decay_iters=${NUM_DECAY_ITERS}"
    "model.discriminator.checkpoint_path=${DINOV3_MODEL}"
    "model.discriminator.num_steps=${NFE}"
    "model.discriminator.global_input_size=${DINO_GLOBAL_SIZE}"
    "model.discriminator.local_input_size=${DINO_LOCAL_SIZE}"
    "model.discriminator.feature_layers=(${DINO_FEATURE_LAYERS},)"
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

if [[ "${EVAL}" == "1" ]]; then
    CFG_OPTS+=("sample_eval.enabled=true")
else
    CFG_OPTS+=("sample_eval.enabled=false")
fi

echo "Launching EditFlow step2 TDM-DINO GAN FSDP (oss ${OSS_PROB} / pico ${PICO_PROB}): nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  num_decay_iters=${NUM_DECAY_ITERS}  gan_weight=${STEP2_GAN_WEIGHT}  gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY}  global=${DINO_GLOBAL_SIZE}  local=${DINO_LOCAL_SIZE}  layer=${DINO_FEATURE_LAYERS}  oss_root=${OSS_ROOT}  pico_root=${PICO_ROOT}  load_from=${LOAD_FROM:-none}  resume_from=${RESUME_FROM:-none}  run=${RUN_NAME}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"



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