#!/usr/bin/env bash
# EditFlow fixed-eps alpha PIID + step-2 TDM-style DINO feature GAN:
#   70% oss_edit banana-task pairs + 30% pico-banana-400k.
#   Student: ArcFluxEditNewAlphaTransformer2DModel (proj_out_alpha).
#   OssEdit uses the same Kontext resolution buckets as ImageEdit.
#
#   PRETRAIN_CKPT=pretrain/flux_kontext/nogan/iter_20000.pth \
#     bash train_flux_edit_fixedeps_data_step2_alph_dino_gan_plus_oss.sh
#   FRESH=1 bash train_flux_edit_fixedeps_data_step2_alph_dino_gan_plus_oss.sh

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
GEDIT_SAMPLES_PER_CATEGORY="${GEDIT_SAMPLES_PER_CATEGORY:-3}"
TOTAL_ITERS="${TOTAL_ITERS:-25000}"
EVAL="${EVAL:-1}"
PICO_ROOT="${PICO_ROOT:-/mnt/afs_gaochengmin/data/pico-banana-400k}"
OSS_ROOT="${OSS_ROOT:-/mnt/afs_gaochengmin/data/oss_edit}"
OSS_JSONL="${OSS_JSONL:-${OSS_ROOT}/metadata.jsonl}"
OSS_PROB="${OSS_PROB:-0.3}"
PICO_PROB="${PICO_PROB:-0.7}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev}"
DINOV3_MODEL="${DINOV3_MODEL:-/mnt/afs_gaochengmin/checkpoints/dinov3/dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-pretrain/flux_kontext/nogan/iter_20000.pth}"
STEP2_GAN_WARMUP_ITERS="${STEP2_GAN_WARMUP_ITERS:-0}"
STEP2_GAN_RAMP_ITERS="${STEP2_GAN_RAMP_ITERS:-0}"
STEP2_GAN_WEIGHT="${STEP2_GAN_WEIGHT:-0.05}"
DIRECT_DELTA_WEIGHT="${DIRECT_DELTA_WEIGHT:-0.0}"
GAN_GRAD_STEP2_ONLY="${GAN_GRAD_STEP2_ONLY:-false}"
NUM_DECAY_ITERS="${NUM_DECAY_ITERS:-0}"
DINO_GLOBAL_SIZE="${DINO_GLOBAL_SIZE:-224}"
DINO_LOCAL_SIZE="${DINO_LOCAL_SIZE:-224}"
DINO_FEATURE_LAYERS="${DINO_FEATURE_LAYERS:-23}"
P_DISABLE_LOCAL="${P_DISABLE_LOCAL:-0.3}"
EDIT_IS_LOW_ALPHA="${EDIT_IS_LOW_ALPHA:-true}"
ALPHA_SMOOTH_SIGMA="${ALPHA_SMOOTH_SIGMA:-2.0}"
MASS_THRESHOLD_PERCENTILE="${MASS_THRESHOLD_PERCENTILE:-30.0}"
MASS_COVERAGE_MIN="${MASS_COVERAGE_MIN:-0.85}"
MASS_COVERAGE_MAX="${MASS_COVERAGE_MAX:-0.90}"
HOT_MASS_FRAC="${HOT_MASS_FRAC:-0.40}"
UNION_AREA_MAX_RATIO="${UNION_AREA_MAX_RATIO:-0.35}"
BBOX_EXPAND_FACTOR="${BBOX_EXPAND_FACTOR:-1.1}"
MIN_CROP_AREA_RATIO="${MIN_CROP_AREA_RATIO:-0.01}"
MAX_CROP_AREA_RATIO="${MAX_CROP_AREA_RATIO:-0.35}"
MIN_EDIT_MASS_RATIO="${MIN_EDIT_MASS_RATIO:-0.002}"
MIN_COMPONENT_PIXELS="${MIN_COMPONENT_PIXELS:-16}"
GAN_GLOBAL_WEIGHT="${GAN_GLOBAL_WEIGHT:-1.0}"
GAN_RANDOM_LOCAL_WEIGHT="${GAN_RANDOM_LOCAL_WEIGHT:-1.0}"
GAN_MASK_LOCAL_WEIGHT="${GAN_MASK_LOCAL_WEIGHT:-1.0}"
CONDITION_ON_SOURCE="${CONDITION_ON_SOURCE:-true}"
USE_MASK_LOCAL_CROP="${USE_MASK_LOCAL_CROP:-true}"
NUM_LOCAL_CROPS="${NUM_LOCAL_CROPS:-1}"
# --------------------------------

RUN_NAME="${RUN_NAME:-gmkontext_uedit_fixedeps_alpha_k16_${NFE}nfe_oss${OSS_PROB}_pico${PICO_PROB}_step2_alph_dino_gan}"
CONFIG="${PROJECT_DIR}/configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_step2_alph_dino_gan_oss_pico.py"

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
if [[ -n "${PRETRAIN_CKPT}" && -e "${PROJECT_DIR}/${PRETRAIN_CKPT}" ]]; then
    LOAD_FROM="${PRETRAIN_CKPT}"
    echo "[pretrain] loading alpha student weights from ${LOAD_FROM}"
elif [[ -n "${PRETRAIN_CKPT}" ]]; then
    echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT})." >&2
    if [[ "${ALLOW_NO_PRETRAIN:-0}" == "1" ]]; then
        echo "[pretrain] ALLOW_NO_PRETRAIN=1; starting from scratch (alpha mask crop will mostly fallback)." >&2
    else
        echo "[pretrain] Refuse to start without alpha weights (flat alpha => mask_fallback=1). Set ALLOW_NO_PRETRAIN=1 to override." >&2
        exit 1
    fi
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
        echo "[resume] resuming step2-alph-dino-gan run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
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
    "train_cfg.direct_delta_loss_weight=${DIRECT_DELTA_WEIGHT}"
    "train_cfg.gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY}"
    "train_cfg.num_decay_iters=${NUM_DECAY_ITERS}"
    "model.discriminator.checkpoint_path=${DINOV3_MODEL}"
    "model.discriminator.num_steps=${NFE}"
    "model.discriminator.global_input_size=${DINO_GLOBAL_SIZE}"
    "model.discriminator.local_input_size=${DINO_LOCAL_SIZE}"
    "model.discriminator.feature_layers=(${DINO_FEATURE_LAYERS},)"
    "model.discriminator.p_disable_local=${P_DISABLE_LOCAL}"
    "model.discriminator.edit_is_low_alpha=${EDIT_IS_LOW_ALPHA}"
    "model.discriminator.alpha_smooth_sigma=${ALPHA_SMOOTH_SIGMA}"
    "model.discriminator.mass_threshold_percentile=${MASS_THRESHOLD_PERCENTILE}"
    "model.discriminator.mass_coverage_min=${MASS_COVERAGE_MIN}"
    "model.discriminator.mass_coverage_max=${MASS_COVERAGE_MAX}"
    "model.discriminator.hot_mass_frac=${HOT_MASS_FRAC}"
    "model.discriminator.union_area_max_ratio=${UNION_AREA_MAX_RATIO}"
    "model.discriminator.bbox_expand_factor=${BBOX_EXPAND_FACTOR}"
    "model.discriminator.min_crop_area_ratio=${MIN_CROP_AREA_RATIO}"
    "model.discriminator.max_crop_area_ratio=${MAX_CROP_AREA_RATIO}"
    "model.discriminator.min_edit_mass_ratio=${MIN_EDIT_MASS_RATIO}"
    "model.discriminator.min_component_pixels=${MIN_COMPONENT_PIXELS}"
    "model.discriminator.gan_global_weight=${GAN_GLOBAL_WEIGHT}"
    "model.discriminator.gan_random_local_weight=${GAN_RANDOM_LOCAL_WEIGHT}"
    "model.discriminator.gan_mask_local_weight=${GAN_MASK_LOCAL_WEIGHT}"
    "model.discriminator.condition_on_source=${CONDITION_ON_SOURCE}"
    "model.discriminator.use_mask_local_crop=${USE_MASK_LOCAL_CROP}"
    "model.discriminator.num_local_crops=${NUM_LOCAL_CROPS}"
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

echo "Launching EditFlow step2 alpha-mask DINO GAN FSDP (oss ${OSS_PROB} / pico ${PICO_PROB}): nproc_per_node=${NUM_GPUS}  total_iters=${TOTAL_ITERS}  num_decay_iters=${NUM_DECAY_ITERS}  gan_weight=${STEP2_GAN_WEIGHT}  gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY}  use_mask_local=${USE_MASK_LOCAL_CROP}  num_local_crops=${NUM_LOCAL_CROPS}  global=${DINO_GLOBAL_SIZE}  local=${DINO_LOCAL_SIZE}  layer=${DINO_FEATURE_LAYERS}  oss_root=${OSS_ROOT}  pico_root=${PICO_ROOT}  load_from=${LOAD_FROM:-none}  resume_from=${RESUME_FROM:-none}  run=${RUN_NAME}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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
