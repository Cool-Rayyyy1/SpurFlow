#!/usr/bin/env bash
# Flux.1-Kontext SFT via SimpleTuner (bghira/SimpleTuner).
# Default mix: 70% pico-banana-400k + 30% oss_edit (same as SenseFlow/UniPic oss mix).
#
# This is intentionally separate from EditFlow distillation launchers
# (train_flux.sh / train_flux_edit_*.sh). Do not source setup_env.sh here.
#
# Prereqs:
#   bash setup_simpletuner_env.sh --install   # first time only
#
# Launch (pico/oss pair dirs are built automatically if missing):
#   bash train_flux_kontext_sft_simpletuner.sh              # all visible GPUs, pico70/oss30
#   bash train_flux_kontext_sft_simpletuner.sh 2            # 2 GPUs
#   OSS_PROB=0.5 PICO_PROB=0.5 bash train_flux_kontext_sft_simpletuner.sh
#   OSS_PROB=0 bash train_flux_kontext_sft_simpletuner.sh   # pico-only
#
# This pod currently has 1x H100; do NOT pass 8. Default is nvidia-smi count.
#   bash train_flux_kontext_sft_simpletuner.sh
#   GPU_IDS=0 bash train_flux_kontext_sft_simpletuner.sh 1
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

REQUESTED_NUM_GPUS=""
if [[ $# -ge 1 && "${1}" =~ ^[0-9]+$ ]]; then
    REQUESTED_NUM_GPUS="${1}"
    shift
elif [[ -n "${NUM_GPUS:-}" ]]; then
    REQUESTED_NUM_GPUS="${NUM_GPUS}"
elif [[ -n "${NPROC:-}" ]]; then
    REQUESTED_NUM_GPUS="${NPROC}"
fi
if [[ -n "${REQUESTED_NUM_GPUS}" ]] && { ! [[ "${REQUESTED_NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${REQUESTED_NUM_GPUS}" -lt 1 ]]; }; then
    echo "Invalid NUM_GPUS=${REQUESTED_NUM_GPUS}" >&2
    exit 1
fi

# ---------- user knobs ----------
DATA_ROOT="${DATA_ROOT:-/mnt/afs_gaochengmin/data/pico-banana-400k}"
PAIR_ROOT="${PAIR_ROOT:-${DATA_ROOT}/simpletuner_pairs}"
OSS_ROOT="${OSS_ROOT:-/mnt/afs_gaochengmin/data/oss_edit}"
OSS_PAIR_ROOT="${OSS_PAIR_ROOT:-${OSS_ROOT}/simpletuner_pairs}"
OSS_PROB="${OSS_PROB:-0.3}"
PICO_PROB="${PICO_PROB:-0.7}"
KONTEXT_MODEL="${KONTEXT_MODEL:-/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev}"
oss_pct="$(awk -v p="${OSS_PROB}" 'BEGIN{printf "%d", p*100+0.5}')"
pico_pct="$(awk -v p="${PICO_PROB}" 'BEGIN{printf "%d", p*100+0.5}')"
RUN_NAME="${RUN_NAME:-flux_kontext_sft_oss${oss_pct}_pico${pico_pct}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/work_dirs/simpletuner_kontext_sft/output/${RUN_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_DIR}/work_dirs/simpletuner_kontext_sft/cache}"
GPU_IDS="${GPU_IDS:-}"

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
# Bucket JSON already exists next to the pair dirs. Default skip so startup
# does not listdir 250k+ files or T5-encode every caption.
# Valid tokens: aspect,vae,text,metadata (comma-separated).
SKIP_FILE_DISCOVERY="${SKIP_FILE_DISCOVERY:-aspect,metadata}"
IGNORE_MISSING_FILES="${IGNORE_MISSING_FILES:-0}"
# --------------------------------

EDIT_TRAIN="${PAIR_ROOT}/train/edit"
REF_TRAIN="${PAIR_ROOT}/train/reference"
EDIT_VAL="${PAIR_ROOT}/val/edit"
REF_VAL="${PAIR_ROOT}/val/reference"
OSS_EDIT_TRAIN="${OSS_PAIR_ROOT}/train/edit"
OSS_REF_TRAIN="${OSS_PAIR_ROOT}/train/reference"

if [[ ! -d "${EDIT_TRAIN}" || ! -d "${REF_TRAIN}" ]]; then
    echo "[data] pico SimpleTuner pairs missing; building ${PAIR_ROOT}"
    DATA_ROOT="${DATA_ROOT}" OUT_ROOT="${PAIR_ROOT}" \
        bash "${PROJECT_DIR}/prepare_pico_banana_simpletuner.sh"
fi
if [[ ! -d "${EDIT_TRAIN}" || ! -d "${REF_TRAIN}" ]]; then
    echo "[error] pico pairs still missing under ${PAIR_ROOT}" >&2
    exit 1
fi
use_oss="$(python3 -c "print(1 if float('${OSS_PROB}') > 0 else 0)")"
if [[ "${use_oss}" == "1" ]]; then
    if [[ ! -d "${OSS_EDIT_TRAIN}" || ! -d "${OSS_REF_TRAIN}" ]]; then
        echo "[data] oss SimpleTuner pairs missing; building ${OSS_PAIR_ROOT}"
        OSS_ROOT="${OSS_ROOT}" OUT_ROOT="${OSS_PAIR_ROOT}" \
            bash "${PROJECT_DIR}/prepare_oss_edit_simpletuner.sh"
    fi
    if [[ ! -d "${OSS_EDIT_TRAIN}" || ! -d "${OSS_REF_TRAIN}" ]]; then
        echo "[error] oss pairs still missing under ${OSS_PAIR_ROOT}" >&2
        exit 1
    fi
fi
if [[ ! -f "${KONTEXT_MODEL}/model_index.json" ]]; then
    echo "[error] Kontext model not found: ${KONTEXT_MODEL}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" \
    "${CACHE_ROOT}/vae/edit" "${CACHE_ROOT}/vae/reference" \
    "${CACHE_ROOT}/vae/oss_edit" "${CACHE_ROOT}/vae/oss_reference" \
    "${CACHE_ROOT}/vae/edit_val" "${CACHE_ROOT}/vae/reference_val" \
    "${CACHE_ROOT}/text"

mapfile -t REAL_GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
if [[ "${#REAL_GPU_IDS[@]}" -eq 0 ]]; then
    echo "[error] nvidia-smi found no GPUs" >&2
    exit 1
fi
if [[ -z "${GPU_IDS}" ]]; then
    GPU_IDS="$(IFS=','; echo "${REAL_GPU_IDS[*]}")"
fi

declare -A REAL_GPU_SET=()
for gid in "${REAL_GPU_IDS[@]}"; do
    REAL_GPU_SET["${gid}"]=1
done
GPU_ARR=()
IFS=',' read -r -a _REQ_GPUS <<< "${GPU_IDS}"
for gid in "${_REQ_GPUS[@]}"; do
    gid="${gid// /}"
    [[ -z "${gid}" ]] && continue
    if [[ -z "${REAL_GPU_SET[${gid}]+x}" ]]; then
        echo "[warn] GPU ${gid} is not present on this machine; dropping it." >&2
        continue
    fi
    GPU_ARR+=("${gid}")
done
if [[ "${#GPU_ARR[@]}" -eq 0 ]]; then
    echo "[warn] none of GPU_IDS=${GPU_IDS} exist; using all visible GPUs: ${REAL_GPU_IDS[*]}" >&2
    GPU_ARR=("${REAL_GPU_IDS[@]}")
fi
GPU_IDS="$(IFS=','; echo "${GPU_ARR[*]}")"

if [[ -z "${REQUESTED_NUM_GPUS}" ]]; then
    NUM_GPUS="${#GPU_ARR[@]}"
else
    NUM_GPUS="${REQUESTED_NUM_GPUS}"
fi
if [[ "${NUM_GPUS}" -gt "${#GPU_ARR[@]}" ]]; then
    echo "[warn] requested NUM_GPUS=${NUM_GPUS} but only ${#GPU_ARR[@]} GPU(s) exist (${GPU_IDS}); clamping." >&2
    NUM_GPUS="${#GPU_ARR[@]}"
fi

# Cluster WORLD_SIZE/RANK makes accelerate wait for other nodes forever.
if [[ "${SIMPLETUNER_MULTI_NODE:-0}" != "1" ]]; then
    unset WORLD_SIZE RANK LOCAL_RANK GROUP_RANK MASTER_ADDR MASTER_PORT \
        PET_NODE_RANK PET_NNODES PET_NPROC_PER_NODE TORCHELASTIC_RUN_ID
    export TRAINING_NUM_MACHINES=1
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export SIMPLETUNER_OUTPUT_DIR="${OUTPUT_DIR}"
if [[ "${REPORT_TO}" != *wandb* ]]; then
    export WANDB_DISABLED=true
fi

# GPU selection via CUDA_VISIBLE_DEVICES only.
# Do NOT put accelerate_visible_devices / num_processes into config.json —
# those are launch-only knobs and break argparse inside train.py.
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TRAINING_NUM_PROCESSES="${NUM_GPUS}"

ACCEL_YAML="${CONFIG_DIR}/accelerate.local.yaml"
if [[ "${NUM_GPUS}" -eq 1 ]]; then
    _DIST_TYPE="NO"
else
    _DIST_TYPE="MULTI_GPU"
fi
cat > "${ACCEL_YAML}" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: ${_DIST_TYPE}
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${NUM_GPUS}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
export ACCELERATE_CONFIG_PATH="${ACCEL_YAML}"

# Rewrite dataloader paths so overrides (PAIR_ROOT/CACHE_ROOT) never require
# hand-editing checked-in JSON.
python - <<PY
import json
from pathlib import Path

cfg_path = Path("${CONFIG_DIR}/multidatabackend.json")
data = json.loads(cfg_path.read_text())
pico_prob = float("${PICO_PROB}")
oss_prob = float("${OSS_PROB}")
total = pico_prob + oss_prob
if total <= 0:
    raise SystemExit("PICO_PROB + OSS_PROB must be > 0")
pico_prob /= total
oss_prob /= total
use_oss = oss_prob > 0
mapping = {
    "pico-banana-edit": ("${EDIT_TRAIN}", "${CACHE_ROOT}/vae/edit"),
    "pico-banana-reference": ("${REF_TRAIN}", "${CACHE_ROOT}/vae/reference"),
    "oss-edit": ("${OSS_EDIT_TRAIN}", "${CACHE_ROOT}/vae/oss_edit"),
    "oss-reference": ("${OSS_REF_TRAIN}", "${CACHE_ROOT}/vae/oss_reference"),
    "pico-banana-edit-val": ("${EDIT_VAL}", "${CACHE_ROOT}/vae/edit_val"),
    "pico-banana-reference-val": ("${REF_VAL}", "${CACHE_ROOT}/vae/reference_val"),
    "pico-banana-text-embeds": (None, "${CACHE_ROOT}/text"),
}
probs = {
    "pico-banana-edit": pico_prob if use_oss else 1.0,
    "oss-edit": oss_prob,
}
vae_ondemand = "${VAE_CACHE_ONDEMAND}" == "1"
text_ondemand = "${TEXT_CACHE_ONDEMAND}" == "1"
kept = []
for entry in data:
    eid = entry.get("id")
    if eid in ("oss-edit", "oss-reference") and not use_oss:
        continue
    if eid not in mapping:
        kept.append(entry)
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
        entry["hash_filenames"] = False
    entry["preserve_data_backend_cache"] = True
    if eid in probs:
        entry["probability"] = float(probs[eid])
        entry["disabled"] = False
    kept.append(entry)
cfg_path.write_text(json.dumps(kept, indent=2) + "\n")
print(
    f"[config] wrote {cfg_path} (pico={pico_prob:.2f} oss={oss_prob:.2f} use_oss={use_oss})"
)
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
cfg["--tracker_project_name"] = "flux-kontext-oss-pico-sft"
cfg["--tracker_run_name"] = "${RUN_NAME}"
cfg["--vae_cache_ondemand"] = "true" if "${VAE_CACHE_ONDEMAND}" == "1" else "false"
# Skip the 28-step Kontext baseline sample at step 0; it looks like a hang.
cfg["--disable_benchmark"] = "true"
# Do not delete all_image_files_*.json on startup (that forces a 250k NFS listdir).
cfg["--preserve_data_backend_cache"] = "true"
# This SimpleTuner build has no CLI --text_cache_ondemand; keep it on the
# text_embeds backend in multidatabackend.json only.
cfg.pop("--text_cache_ondemand", None)
skip_tokens = [t.strip() for t in "${SKIP_FILE_DISCOVERY}".split(",") if t.strip()]
# This SimpleTuner still pre-reads every caption and T5-encodes them unless
# "text" is skipped. TEXT_CACHE_ONDEMAND only lives on the data backend.
if "${TEXT_CACHE_ONDEMAND}" == "1" and "text" not in skip_tokens:
    skip_tokens.append("text")
if "${VAE_CACHE_ONDEMAND}" == "1" and "vae" not in skip_tokens:
    skip_tokens.append("vae")
skip_disc = ",".join(skip_tokens)
if skip_disc:
    cfg["--skip_file_discovery"] = skip_disc
else:
    cfg.pop("--skip_file_discovery", None)
cfg["--ignore_missing_files"] = "true" if "${IGNORE_MISSING_FILES}" == "1" else "false"
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
print(f"[config] disable_benchmark=true skip_file_discovery={skip_disc or '<none>'} preserve_data_backend_cache=true")
PY

python - <<PY
"""Rebuild truncated bucket indices from metadata and seed all_image_files JSON.

SimpleTuner deletes all_image_files_<id>.json unless preserve_data_backend_cache
is set, then VAECache.list_files() rglob/iterdir 250k+ NFS entries (looks hung).
pico-banana-edit indices were also truncated to 54 paths; metadata still has 257k.
"""
import json
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
output_dir.mkdir(parents=True, exist_ok=True)

jobs = [
    ("${EDIT_TRAIN}", "pico-banana-edit"),
    ("${REF_TRAIN}", "pico-banana-reference"),
    ("${EDIT_VAL}", "pico-banana-edit-val"),
    ("${REF_VAL}", "pico-banana-reference-val"),
]
if "${use_oss}" == "1":
    jobs.extend(
        [
            ("${OSS_EDIT_TRAIN}", "oss-edit"),
            ("${OSS_REF_TRAIN}", "oss-reference"),
        ]
    )

for instance_dir, backend_id in jobs:
    folder = Path(instance_dir)
    meta_path = folder / f"aspect_ratio_bucket_metadata_{backend_id}.json"
    idx_path = folder / f"aspect_ratio_bucket_indices_{backend_id}.json"
    if not meta_path.is_file():
        print(f"[seed] skip {backend_id}: missing {meta_path.name}")
        continue
    meta = json.loads(meta_path.read_text())
    indices = {}
    for path, info in meta.items():
        ar = 1.0
        if isinstance(info, dict) and info.get("aspect_ratio") is not None:
            ar = float(info["aspect_ratio"])
        indices.setdefault(str(float(ar)), []).append(path)
    n_meta = len(meta)
    n_idx = 0
    cfg = {}
    if idx_path.is_file():
        try:
            old = json.loads(idx_path.read_text())
            cfg = old.get("config", {}) if isinstance(old, dict) else {}
            old_idx = old.get("aspect_ratio_bucket_indices", {}) if isinstance(old, dict) else {}
            n_idx = sum(len(v) for v in old_idx.values() if isinstance(v, list))
        except json.JSONDecodeError:
            n_idx = 0
    if n_idx < n_meta:
        idx_path.write_text(
            json.dumps({"config": cfg, "aspect_ratio_bucket_indices": indices}) + "\n",
            encoding="utf-8",
        )
        print(f"[repair] {backend_id} indices {n_idx} -> {n_meta}")
    else:
        print(f"[seed] {backend_id} indices ok n={n_idx}")
    out = output_dir / f"all_image_files_{backend_id}.json"
    out.write_text(json.dumps({p: False for p in meta}), encoding="utf-8")
    print(f"[seed] {out.name} n={n_meta}")
PY

cd "${WORKDIR}"

echo "[train] SimpleTuner Flux Kontext SFT"
echo "        nproc=${NUM_GPUS}  gpus=${CUDA_VISIBLE_DEVICES}  machines=${TRAINING_NUM_MACHINES:-1}"
echo "        accelerate=${ACCELERATE_CONFIG_PATH}"
echo "        model=${KONTEXT_MODEL}"
echo "        pico_pairs=${PAIR_ROOT}  p=${PICO_PROB}"
echo "        oss_pairs=${OSS_PAIR_ROOT}  p=${OSS_PROB}"
echo "        mix=oss ${oss_pct}% / pico ${pico_pct}%"
echo "        out=${OUTPUT_DIR}"
echo "        type=${MODEL_TYPE}/${LORA_TYPE}  rank=${LORA_RANK}  steps=${MAX_TRAIN_STEPS}"
echo "        text_ondemand=${TEXT_CACHE_ONDEMAND}  vae_ondemand=${VAE_CACHE_ONDEMAND}"
echo "        skip_file_discovery=${SKIP_FILE_DISCOVERY:-<none>}  ignore_missing_files=${IGNORE_MISSING_FILES}"
echo "        disable_benchmark=true preserve_cache=true"
echo "        PYTHONPATH patch: skip NFS rglob of pair dirs"

simpletuner train config_backend=json
