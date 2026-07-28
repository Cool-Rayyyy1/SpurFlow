#!/usr/bin/env bash
# Flux.1-Kontext SFT on pico-banana via SimpleTuner (bghira/SimpleTuner).
#
# This is intentionally separate from EditFlow distillation launchers
# (train_flux.sh / train_flux_edit_*.sh). Do not source setup_env.sh here.
#
# Prereqs:
#   bash setup_simpletuner_env.sh --install
#   bash prepare_pico_banana_simpletuner.sh          # or MAX_SAMPLES=2000 ...
#
# Launch:
#   bash train_flux_kontext_sft_simpletuner.sh              # 8 GPUs default
#   bash train_flux_kontext_sft_simpletuner.sh 2            # 2 GPUs
#   GPU_IDS=0,1 NUM_GPUS=2 bash train_flux_kontext_sft_simpletuner.sh
#   MAX_TRAIN_STEPS=2000 LORA_RANK=16 bash train_flux_kontext_sft_simpletuner.sh
#   LORA_TYPE=lycoris bash train_flux_kontext_sft_simpletuner.sh
#
# Single-GPU smoke:
#   GPU_IDS=0 PAIR_ROOT=.../simpletuner_pairs_smoke MAX_TRAIN_STEPS=50 \
#     bash train_flux_kontext_sft_simpletuner.sh 1
#
# Full-rank (needs DeepSpeed/FSDP; not the recommended Kontext path):
#   MODEL_TYPE=full bash train_flux_kontext_sft_simpletuner.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${SIMPLETUNER_WORKDIR:-${PROJECT_DIR}/simpletuner_kontext_sft}"
CONFIG_DIR="${WORKDIR}/config"

# Activate dedicated SimpleTuner env (never arcflow).
# shellcheck source=/dev/null
source "${PROJECT_DIR}/setup_simpletuner_env.sh"

if [[ $# -ge 1 && "${1}" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="${1}"
    shift
else
    NUM_GPUS="${NUM_GPUS:-${NPROC:-8}}"
fi
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "Invalid NUM_GPUS=${NUM_GPUS}" >&2
    exit 1
fi

# ---------- user knobs ----------
DATA_ROOT="${DATA_ROOT:-/mnt/afs_zhangyunzhe/dataset/pico-banana-400k}"
PAIR_ROOT="${PAIR_ROOT:-${DATA_ROOT}/simpletuner_pairs}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev}"
RUN_NAME="${RUN_NAME:-flux_kontext_sft_pico400k}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/work_dirs/simpletuner_kontext_sft/output/${RUN_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_DIR}/work_dirs/simpletuner_kontext_sft/cache}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

MODEL_TYPE="${MODEL_TYPE:-lora}"          # lora | full
LORA_TYPE="${LORA_TYPE:-standard}"        # standard | lycoris
LORA_RANK="${LORA_RANK:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RESOLUTION="${RESOLUTION:-1024}"
CKPT_INTERVAL="${CKPT_INTERVAL:-500}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"
BASE_PRECISION="${BASE_PRECISION:-no_change}"  # no_change | int8-quanto | nf4-bnb ...
OPTIMIZER="${OPTIMIZER:-adamw_bf16}"
REPORT_TO="${REPORT_TO:-tensorboard}"
# on-demand encodes during training (faster startup, slower steps)
TEXT_CACHE_ONDEMAND="${TEXT_CACHE_ONDEMAND:-1}"
VAE_CACHE_ONDEMAND="${VAE_CACHE_ONDEMAND:-1}"
# --------------------------------

EDIT_TRAIN="${PAIR_ROOT}/train/edit"
REF_TRAIN="${PAIR_ROOT}/train/reference"
EDIT_VAL="${PAIR_ROOT}/val/edit"
REF_VAL="${PAIR_ROOT}/val/reference"

if [[ ! -d "${EDIT_TRAIN}" || ! -d "${REF_TRAIN}" ]]; then
    echo "[error] Paired data missing under ${PAIR_ROOT}" >&2
    echo "        Run: bash ${PROJECT_DIR}/prepare_pico_banana_simpletuner.sh" >&2
    exit 1
fi
if [[ ! -f "${KONTEXT_MODEL}/model_index.json" ]]; then
    echo "[error] Kontext model not found: ${KONTEXT_MODEL}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" \
    "${CACHE_ROOT}/vae/edit" "${CACHE_ROOT}/vae/reference" \
    "${CACHE_ROOT}/vae/edit_val" "${CACHE_ROOT}/vae/reference_val" \
    "${CACHE_ROOT}/text"

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
if [[ "${#GPU_ARR[@]}" -lt "${NUM_GPUS}" ]]; then
    echo "[warn] GPU_IDS has ${#GPU_ARR[@]} entries but NUM_GPUS=${NUM_GPUS}; truncating NUM_GPUS." >&2
    NUM_GPUS="${#GPU_ARR[@]}"
fi

# GPU selection via CUDA_VISIBLE_DEVICES only.
# Do NOT put accelerate_visible_devices / num_processes into config.json —
# those are launch-only knobs and break argparse inside train.py.
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TRAINING_NUM_PROCESSES="${NUM_GPUS}"

# Rewrite dataloader paths so overrides (PAIR_ROOT/CACHE_ROOT) never require
# hand-editing checked-in JSON.
python - <<PY
import json
from pathlib import Path

cfg_path = Path("${CONFIG_DIR}/multidatabackend.json")
data = json.loads(cfg_path.read_text())
mapping = {
    "pico-banana-edit": ("${EDIT_TRAIN}", "${CACHE_ROOT}/vae/edit"),
    "pico-banana-reference": ("${REF_TRAIN}", "${CACHE_ROOT}/vae/reference"),
    "pico-banana-edit-val": ("${EDIT_VAL}", "${CACHE_ROOT}/vae/edit_val"),
    "pico-banana-reference-val": ("${REF_VAL}", "${CACHE_ROOT}/vae/reference_val"),
    "pico-banana-text-embeds": (None, "${CACHE_ROOT}/text"),
}
vae_ondemand = "${VAE_CACHE_ONDEMAND}" == "1"
text_ondemand = "${TEXT_CACHE_ONDEMAND}" == "1"
for entry in data:
    eid = entry.get("id")
    if eid not in mapping:
        continue
    inst, cache = mapping[eid]
    if entry.get("dataset_type") == "text_embeds":
        entry["cache_dir"] = cache
        entry["text_cache_ondemand"] = text_ondemand
    else:
        if inst:
            entry["instance_data_dir"] = inst
        entry["cache_dir_vae"] = cache
        entry["vae_cache_ondemand"] = vae_ondemand
cfg_path.write_text(json.dumps(data, indent=2) + "\n")
print(f"[config] wrote {cfg_path}")
PY

python - <<PY
import json
from pathlib import Path

cfg_path = Path("${CONFIG_DIR}/config.json")
cfg = json.loads(cfg_path.read_text())
cfg["--output_dir"] = "${OUTPUT_DIR}"
cfg["--pretrained_model_name_or_path"] = "${KONTEXT_MODEL}"
cfg["--pretrained_vae_model_name_or_path"] = "${KONTEXT_MODEL}"
cfg["--model_type"] = "${MODEL_TYPE}"
cfg["--lora_type"] = "${LORA_TYPE}"
cfg["--lora_rank"] = int("${LORA_RANK}")
cfg["--learning_rate"] = "${LEARNING_RATE}"
cfg["--max_train_steps"] = int("${MAX_TRAIN_STEPS}")
cfg["--train_batch_size"] = int("${TRAIN_BATCH_SIZE}")
cfg["--gradient_accumulation_steps"] = int("${GRAD_ACCUM}")
cfg["--resolution"] = int("${RESOLUTION}")
cfg["--checkpoint_step_interval"] = int("${CKPT_INTERVAL}")
cfg["--validation_step_interval"] = int("${VAL_INTERVAL}")
cfg["--base_model_precision"] = "${BASE_PRECISION}"
cfg["--optimizer"] = "${OPTIMIZER}"
cfg["--report_to"] = "${REPORT_TO}"
cfg["--tracker_run_name"] = "${RUN_NAME}"
cfg["--text_cache_ondemand"] = "true" if "${TEXT_CACHE_ONDEMAND}" == "1" else "false"
cfg["--vae_cache_ondemand"] = "true" if "${VAE_CACHE_ONDEMAND}" == "1" else "false"
# Strip launch-only keys that are not valid train.py argparse flags.
cfg.pop("--num_processes", None)
cfg.pop("num_processes", None)
cfg.pop("--accelerate_visible_devices", None)
cfg.pop("accelerate_visible_devices", None)
if "${LORA_TYPE}" == "lycoris":
    cfg["--lycoris_config"] = "config/lycoris_config.json"
else:
    cfg.pop("--lycoris_config", None)
if "${MODEL_TYPE}" == "full":
    cfg["--model_type"] = "full"
    cfg.pop("--lora_type", None)
    cfg.pop("--lora_rank", None)
    cfg.pop("--lora_dropout", None)
    cfg.pop("--lycoris_config", None)
cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"[config] wrote {cfg_path}")
PY

cd "${WORKDIR}"

echo "[train] SimpleTuner Flux Kontext SFT"
echo "        nproc=${NUM_GPUS}  gpus=${CUDA_VISIBLE_DEVICES}"
echo "        model=${KONTEXT_MODEL}"
echo "        pairs=${PAIR_ROOT}"
echo "        out=${OUTPUT_DIR}"
echo "        type=${MODEL_TYPE}/${LORA_TYPE}  rank=${LORA_RANK}  steps=${MAX_TRAIN_STEPS}"
echo "        text_ondemand=${TEXT_CACHE_ONDEMAND}  vae_ondemand=${VAE_CACHE_ONDEMAND}"

simpletuner train config_backend=json
