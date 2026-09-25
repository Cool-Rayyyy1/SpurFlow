#!/usr/bin/env bash
# EditFlow: distill FLUX.2-klein-base-9B -> 2-NFE Softsign-01 alpha student (no GAN).
# Goal: beat official step-distilled FLUX.2-klein-9B with our 2-step student.
#
# Training mode: NON-split-stage random-segment PIID (ArcFlowEditAlphaImitation),
# same as train_flux_edit_fixedeps_alpha_data_qwen.sh. NO GAN / discriminator.
#
# Softsign-01:
#   alpha = 0.5 * (raw / (1 + abs(raw)) + 1)
#   student_u = path_epsilon - alpha * x_ref - pred_delta
#
# Guidance:
#   Teacher (klein-base): official CFG=4.0
#     Flux2KleinPipeline runs cond DiT forward + uncond DiT forward, then combines.
#     (guidance_embeds=false — NOT Flux2-dev / Kontext single-pass embeds)
#     No CFG norm rescale (that is Qwen-only).
#   Student: CFG=1 / cond-only — one DiT forward, no uncond.
#
# Data: 70% oss_edit + 30% pico-banana-400k (same mix as
#   train_flux2_klein_edit_fixedeps_alpha_softsign_alph_dino_gan_plus_oss.sh).
#   resize_mode=flux2 matches Flux2KleinPipeline
#   (area-cap ~1MP + multiple-of-16; NOT Kontext buckets).
#
# Uses FSDP (configs/flux2_klein/_fsdp_train.py). Prefer >=2 GPUs; 1x80GB
# often OOMs because teacher CFG=4 doubles the teacher forward.
#
# Training-time vis: ImgEdit 9 cats × 5 + GEdit-v2 23 types × 2, two folders.
#
#   bash train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh
#   bash train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh 8
#   TEACHER_CFG=4.0 GPU_IDS=0,1,2,3 bash train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh 4
#   RESUME_RUN_DIR=20260818_033917 RESUME_ITER=10000 bash train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh

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
TEACHER_CFG="${TEACHER_CFG:-4.0}"
CKPT_INTERVAL="${CKPT_INTERVAL:-500}"
CKPT_MUST_SAVE_INTERVAL="${CKPT_MUST_SAVE_INTERVAL:-1000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-500}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
GEDIT_SAMPLES_PER_CATEGORY="${GEDIT_SAMPLES_PER_CATEGORY:-2}"
TOTAL_ITERS="${TOTAL_ITERS:-20000}"
EVAL="${EVAL:-1}"
PICO_ROOT="${PICO_ROOT:-/mnt/afs_gaochengmin/data/pico-banana-400k}"
OSS_ROOT="${OSS_ROOT:-/mnt/afs_gaochengmin/data/oss_edit}"
OSS_JSONL="${OSS_JSONL:-${OSS_ROOT}/metadata.jsonl}"
OSS_PROB="${OSS_PROB:-0.3}"
PICO_PROB="${PICO_PROB:-0.7}"
KLEIN_MODEL="${KLEIN_MODEL:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
RESUME_ITER="${RESUME_ITER:-}"
FRESH="${FRESH:-0}"
# --------------------------------

RUN_NAME="gmklein_base_uedit_fixedeps_alpha_softsign01_k16_${NFE}nfe_shift${SHIFT}_teachercfg${TEACHER_CFG}_oss70_pico30"
CONFIG="${PROJECT_DIR}/configs/flux2_klein/editflux2_klein_uedit_fixedeps_2nfe_k16_alpha_softsign_data_oss_pico.py"

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
export KLEIN_MODEL_PATH="${KLEIN_MODEL}"
export PICO_BANANA_PATH="${PICO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    "train_cfg.teacher_guidance_scale=${TEACHER_CFG}"
    "test_cfg.guidance_scale=1.0"
    "model.diffusion.timestep_sampler.shift=${SHIFT}"
    "checkpoint_config.interval=${CKPT_INTERVAL}"
    "checkpoint_config.must_save_interval=${CKPT_MUST_SAVE_INTERVAL}"
    "sample_eval.interval=${SAMPLE_INTERVAL}"
    "sample_eval.must_save_interval=0"
    "sample_eval.dataset.datasets.0.samples_per_category=${SAMPLES_PER_CATEGORY}"
    "sample_eval.dataset.datasets.0.annotations_path=${PROJECT_DIR}/evaluation/imgedit_bench/annotations/basic_edit.json"
    "sample_eval.dataset.datasets.0.bench_root=/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark"
    "sample_eval.dataset.datasets.1.samples_per_category=${GEDIT_SAMPLES_PER_CATEGORY}"
    "sample_eval.dataset.datasets.1.annotations_path=/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json"
    "sample_eval.dataset.datasets.1.bench_root=/mnt/afs_caiqi/data/benchmark/GEdit_v2"
    "total_iters=${TOTAL_ITERS}"
    "data.train.probs=[${OSS_PROB},${PICO_PROB}]"
    "data.train.datasets.0.data_root=${OSS_ROOT}"
    "data.train.datasets.0.jsonl_path=${OSS_JSONL}"
    "data.train.datasets.1.data_root=${PICO_ROOT}"
    "data.val.data_root=${PICO_ROOT}"
    "model.vae.from_pretrained=${KLEIN_MODEL}"
    "model.text_encoder.from_pretrained=${KLEIN_MODEL}"
    "model.diffusion.denoising.pretrained=${KLEIN_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
    "model.teacher.denoising.pretrained=${KLEIN_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
)

if [[ "${EVAL}" == "1" ]]; then
    CFG_OPTS+=("sample_eval.enabled=true")
else
    CFG_OPTS+=("sample_eval.enabled=false")
fi

echo "Launching EditFlow FLUX.2-klein-base Softsign-01 alpha (no GAN, oss ${OSS_PROB} / pico ${PICO_PROB}): nproc=${NUM_GPUS} nfe=${NFE} shift=${SHIFT} teacher_cfg=${TEACHER_CFG} student_cfg=1.0 total_iters=${TOTAL_ITERS} oss_root=${OSS_ROOT} pico_root=${PICO_ROOT} run=${RUN_NAME} model=${KLEIN_MODEL} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Single-node:
# torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
#     "${CONFIG}" \
#     --launcher pytorch --diff_seed \
#     --cfg-options "${CFG_OPTS[@]}"

# Teacher matches Flux2KleinPipeline (klein-base, is_distilled=false):
#   guidance_scale=4.0, cond then uncond, guidance=None, no norm rescale.
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

