#!/usr/bin/env bash
# Download and extract ImgEdit-Bench (Benchmark.tar, ~48MB).
#
# Usage:
#   bash evaluation/imgedit_bench/download_imgedit_bench.sh
#   DATA_ROOT=/path/to/dataset/imgedit bash evaluation/imgedit_bench/download_imgedit_bench.sh

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-/path/to/data/imgedit}"
TAR_PATH="${DATA_ROOT}/Benchmark.tar"
EXTRACT_DIR="${DATA_ROOT}/benchmark"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export DIFFUSERS_OFFLINE=0

mkdir -p "${DATA_ROOT}"

if [[ ! -f "${TAR_PATH}" ]]; then
  echo "Downloading Benchmark.tar from sysuyy/ImgEdit ..."
  huggingface-cli download --repo-type dataset sysuyy/ImgEdit Benchmark.tar \
    --local-dir "${DATA_ROOT}"
else
  echo "Found existing ${TAR_PATH}"
fi

mkdir -p "${EXTRACT_DIR}"
if [[ ! -d "${EXTRACT_DIR}/Benchmark/singleturn" ]]; then
  echo "Extracting ${TAR_PATH} -> ${EXTRACT_DIR} ..."
  tar -xf "${TAR_PATH}" -C "${EXTRACT_DIR}"
else
  echo "Benchmark already extracted at ${EXTRACT_DIR}/Benchmark"
fi

# Official Benchmark.tar lists 50 UGE ids in hard/annotation.jsonl but only ships
# 48 jpgs under hard/. The remaining ids exist under multiturn/ in the same tar.
HARD_DIR="${EXTRACT_DIR}/Benchmark/hard"
ANNOTATION_JSONL="${HARD_DIR}/annotation.jsonl"
if [[ -f "${ANNOTATION_JSONL}" ]]; then
  echo "Checking UGE hard/ images against ${ANNOTATION_JSONL} ..."
  while IFS= read -r img_id; do
    [[ -z "${img_id}" ]] && continue
    if [[ -f "${HARD_DIR}/${img_id}" ]]; then
      continue
    fi
    extracted=0
    for sub in content_memory content_understand version_backtrace; do
      member="Benchmark/multiturn/${sub}/${img_id}"
      if tar -tf "${TAR_PATH}" "${member}" >/dev/null 2>&1; then
        echo "  extracting missing ${img_id} from ${member}"
        tar -xOf "${TAR_PATH}" "${member}" > "${HARD_DIR}/${img_id}"
        extracted=1
        break
      fi
    done
    if [[ "${extracted}" -eq 0 ]]; then
      echo "WARNING: ${img_id} listed in annotation.jsonl but not found in ${TAR_PATH}" >&2
    fi
  done < <(python3 - "${ANNOTATION_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
for line in jsonl.read_text().splitlines():
    if line.strip():
        print(json.loads(line)["id"])
PY
)
fi

echo "Done."
echo "  tar:      ${TAR_PATH}"
echo "  images:   ${EXTRACT_DIR}/Benchmark"
echo "  metadata: EditFlow/evaluation/imgedit_bench/annotations/"
