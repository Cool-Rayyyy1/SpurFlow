#!/usr/bin/env bash
# Cluster launch for official Qwen / Klein baselines.
# Four jobs run serially on each node:
#   1) qwen  ImgEdit
#   2) qwen  GEdit
#   3) klein ImgEdit
#   4) klein GEdit
# Inside each job, this node fans out GPUS_PER_NODE processes:
#   split_id = RANK * GPUS_PER_NODE .. RANK * GPUS_PER_NODE + GPUS_PER_NODE - 1
#   split_num = WORLD_SIZE * GPUS_PER_NODE
#   metadata[split_id::split_num]
#
# Usage (one process per node, RANK/WORLD_SIZE injected by the scheduler):
#   RANK=0 WORLD_SIZE=2 bash simple_run/run_official_baselines_cluster.sh
#   TASKS="qwen_imgedit,qwen_gedit" bash simple_run/run_official_baselines_cluster.sh

set -uo pipefail

EDITFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${EDITFLOW_DIR}"

if [[ -f "${EDITFLOW_DIR}/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${EDITFLOW_DIR}/env.sh"
fi

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_gaochengmin/anaconda3}"
CONDA_ENV="${CONDA_ENV:-arcflow}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
START_SPLIT_ID=$((RANK * GPUS_PER_NODE))
END_SPLIT_ID=$((START_SPLIT_ID + GPUS_PER_NODE - 1))
SPLIT_NUM=$((WORLD_SIZE * GPUS_PER_NODE))

GEDIT_META="${GEDIT_META:-/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json}"
IMGEDIT_BENCH_ROOT="${IMGEDIT_BENCH_ROOT:-/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark}"
EVAL_GEDIT_ROOT="${EVAL_GEDIT_ROOT:-${EDITFLOW_DIR}/evaluation/gedit_v2/outputs/runs}"
EVAL_IMGEDIT_ROOT="${EVAL_IMGEDIT_ROOT:-${EDITFLOW_DIR}/evaluation/imgedit_bench/outputs/runs}"
LOG_ROOT="${LOG_ROOT:-${EDITFLOW_DIR}/simple_run/outputs/logs}"
PYTHON="${PYTHON:-python}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TASKS="${TASKS:-qwen_imgedit,qwen_gedit,klein_imgedit,klein_gedit}"

export PYTHONPATH="${EDITFLOW_DIR}:${EDITFLOW_DIR}/simple_run:${PYTHONPATH:-}"

echo "[cluster] rank=${RANK}/${WORLD_SIZE}  gpus_per_node=${GPUS_PER_NODE}"
echo "[cluster] split_id=${START_SPLIT_ID}..${END_SPLIT_ID}  split_num=${SPLIT_NUM}"
echo "[cluster] gedit_meta=${GEDIT_META}"
echo "[cluster] gedit_out=${EVAL_GEDIT_ROOT}"
echo "[cluster] imgedit_out=${EVAL_IMGEDIT_ROOT}"
echo "[cluster] tasks=${TASKS}"

run_one_job() {
  local job_name="$1"
  shift
  local -a cmd=("$@")
  local log_dir="${LOG_ROOT}/${job_name}"
  mkdir -p "${log_dir}"
  echo "[cluster] ===== ${job_name} ====="
  echo "[cluster] ${cmd[*]}"

  local -a pids=()
  local split_id gpu_id log_file
  for ((split_id = START_SPLIT_ID; split_id <= END_SPLIT_ID; split_id++)); do
    gpu_id=$((split_id % GPUS_PER_NODE))
    log_file="${log_dir}/rank${RANK}_split${split_id}_gpu${gpu_id}.log"
    echo "[cluster] ${job_name} split_id=${split_id}/${SPLIT_NUM} gpu=${gpu_id} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" \
      --split_id "${split_id}" \
      --split_num "${SPLIT_NUM}" \
      > "${log_file}" 2>&1 &
    pids+=("$!")
  done

  local fail=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      echo "[cluster] ERROR: ${job_name} pid=${pid} failed" >&2
      fail=1
    fi
  done
  if [[ "${fail}" -ne 0 ]]; then
    echo "[cluster] ${job_name} failed; see ${log_dir}" >&2
    return 1
  fi
  echo "[cluster] ${job_name} done"
}

maybe_skip() {
  local extra=()
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    extra+=(--skip_existing)
  fi
  printf '%s\n' "${extra[@]}"
}

contains_task() {
  local needle="$1"
  [[ ",${TASKS}," == *",${needle},"* ]]
}

if contains_task "qwen_imgedit"; then
  run_one_job "qwen_imgedit" \
    ${PYTHON} simple_run/qwen_image_edit/run_imgedit.py \
    --output_dir "${EVAL_IMGEDIT_ROOT}/qwen_image_edit_2511_imgedit" \
    --bench_root "${IMGEDIT_BENCH_ROOT}" \
    --suite all \
    $(maybe_skip) || exit 1
fi

if contains_task "qwen_gedit"; then
  run_one_job "qwen_gedit" \
    ${PYTHON} simple_run/qwen_image_edit/run_gedit.py \
    --output_dir "${EVAL_GEDIT_ROOT}/qwen_image_edit_2511_gedit" \
    --meta_json "${GEDIT_META}" \
    $(maybe_skip) || exit 1
fi

if contains_task "klein_imgedit"; then
  run_one_job "klein_imgedit" \
    ${PYTHON} simple_run/flux2_klein/run_imgedit.py \
    --output_dir "${EVAL_IMGEDIT_ROOT}/flux2_klein_base_9b_imgedit" \
    --bench_root "${IMGEDIT_BENCH_ROOT}" \
    --suite all \
    $(maybe_skip) || exit 1
fi

if contains_task "klein_gedit"; then
  run_one_job "klein_gedit" \
    ${PYTHON} simple_run/flux2_klein/run_gedit.py \
    --output_dir "${EVAL_GEDIT_ROOT}/flux2_klein_base_9b_gedit" \
    --meta_json "${GEDIT_META}" \
    $(maybe_skip) || exit 1
fi

echo "[cluster] all requested tasks finished  rank=${RANK}/${WORLD_SIZE}"
