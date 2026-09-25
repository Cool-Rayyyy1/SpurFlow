#!/usr/bin/env bash
# EditFlow FLUX.2-klein-base Softsign-01 alpha + standard PIID + independent
# step-2 2-NFE rollout DINO GAN (Kontext step2_alph_dino_gan recipe):
#   30% oss_edit + 70% pico-banana-400k (override with OSS_PROB / PICO_PROB).
#   Student: ArcFlux2EditAlphaSoftsign01Transformer2DModel (proj_out_alpha).
#   Distillation: ArcFlowEditImitationStep2GAN
#     - random-segment PIID (same as non-GAN softsign)
#     - EXTRA independent 2-NFE rollout for GAN fake + step-2 alpha
#       (NOT split-stage shared unroll)
#   GAN: DinoAlphaMaskFeatureDiscriminator, condition_on_source,
#     alpha-mask local crop from rolled-out step-2 alpha.
#
# Guidance:
#   Teacher (klein-base): official CFG=4.0
#   Student: CFG=1 / cond-only
#
#   bash train_flux2_klein_edit_fixedeps_alpha_softsign_step2_alph_dino_gan_plus_oss.sh
#   bash train_flux2_klein_edit_fixedeps_alpha_softsign_step2_alph_dino_gan_plus_oss.sh 8
#   PRETRAIN_CKPT=checkpoints/.../iter_XXXX.pth \
#     bash train_flux2_klein_edit_fixedeps_alpha_softsign_step2_alph_dino_gan_plus_oss.sh
#   FRESH=1 bash train_flux2_klein_edit_fixedeps_alpha_softsign_step2_alph_dino_gan_plus_oss.sh

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
GEDIT_SAMPLES_PER_CATEGORY="${GEDIT_SAMPLES_PER_CATEGORY:-3}"
TOTAL_ITERS="${TOTAL_ITERS:-25000}"
EVAL="${EVAL:-1}"
PICO_ROOT="${PICO_ROOT:-/mnt/afs_gaochengmin/data/pico-banana-400k}"
OSS_ROOT="${OSS_ROOT:-/mnt/afs_gaochengmin/data/oss_edit}"
OSS_JSONL="${OSS_JSONL:-${OSS_ROOT}/metadata.jsonl}"
OSS_PROB="${OSS_PROB:-0.3}"
PICO_PROB="${PICO_PROB:-0.7}"
KLEIN_MODEL="${KLEIN_MODEL:-/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B}"
DINOV3_MODEL="${DINOV3_MODEL:-/mnt/afs_gaochengmin/checkpoints/dinov3/dinov3-vitl16-pretrain-lvd1689m/model.safetensors}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
FRESH="${FRESH:-0}"
# Default: same no-GAN Softsign-01 alpha ckpt as the pico-only / split-stage GAN recipes.
PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/gmklein_base_uedit_fixedeps_alpha_softsign01_k16_2nfe_shift3.2_teachercfg4.0_pico400k/20260806_155327/iter_20000.pth}"
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
# --------------------------------

oss_pct="$(awk -v p="${OSS_PROB}" 'BEGIN{printf "%d", p*100+0.5}')"
pico_pct="$(awk -v p="${PICO_PROB}" 'BEGIN{printf "%d", p*100+0.5}')"
RUN_NAME="gmklein_base_uedit_fixedeps_alpha_softsign01_k16_${NFE}nfe_shift${SHIFT}_teachercfg${TEACHER_CFG}_oss${oss_pct}_pico${pico_pct}_step2_alph_dino_gan"
CONFIG="${PROJECT_DIR}/configs/flux2_klein/editflux2_klein_uedit_fixedeps_2nfe_k16_alpha_softsign_step2_alph_dino_gan_oss_pico.py"

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
LOAD_FROM=""
RESUME_FROM=""
if [[ -n "${PRETRAIN_CKPT}" ]]; then
    if [[ "${PRETRAIN_CKPT}" = /* ]]; then
        _pretrain_path="${PRETRAIN_CKPT}"
    else
        _pretrain_path="${PROJECT_DIR}/${PRETRAIN_CKPT}"
    fi
    if [[ -e "${_pretrain_path}" ]]; then
        LOAD_FROM="${_pretrain_path}"
        echo "[pretrain] loading Softsign-01 alpha student weights from ${LOAD_FROM}"
    else
        echo "[pretrain] PRETRAIN_CKPT not found (${PRETRAIN_CKPT})." >&2
        if [[ "${ALLOW_NO_PRETRAIN:-0}" == "1" ]]; then
            echo "[pretrain] ALLOW_NO_PRETRAIN=1; starting from scratch (alpha mask crop will mostly fallback)." >&2
        else
            echo "[pretrain] Refuse to start without alpha weights (flat alpha => mask_fallback=1). Set ALLOW_NO_PRETRAIN=1 to override." >&2
            exit 1
        fi
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
        echo "[resume] resuming klein-softsign-step2-alph-dino-gan plus_oss run_id=${RESUME_RUN_ID} from ${RESUME_FROM}"
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
    "train_cfg.teacher_guidance_scale=${TEACHER_CFG}"
    "test_cfg.guidance_scale=1.0"
    "model.diffusion.timestep_sampler.shift=${SHIFT}"
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

echo "Launching EditFlow FLUX.2-klein Softsign-01 alpha STEP2 DINO GAN FSDP (oss ${OSS_PROB} / pico ${PICO_PROB}): nproc=${NUM_GPUS} nfe=${NFE} shift=${SHIFT} teacher_cfg=${TEACHER_CFG} student_cfg=1.0 total_iters=${TOTAL_ITERS} gan_weight=${STEP2_GAN_WEIGHT} gan_grad_step2_only=${GAN_GRAD_STEP2_ONLY} condition_on_source=${CONDITION_ON_SOURCE} p_disable_local=${P_DISABLE_LOCAL} oss_root=${OSS_ROOT} pico_root=${PICO_ROOT} load_from=${LOAD_FROM:-none} resume_from=${RESUME_FROM:-none} run=${RUN_NAME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [[ -n "${WORLD_SIZE:-}" ]]; then
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
else
    torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${PROJECT_DIR}/train.py" \
        "${CONFIG}" \
        --launcher pytorch --diff_seed \
        --cfg-options "${CFG_OPTS[@]}"
fi
