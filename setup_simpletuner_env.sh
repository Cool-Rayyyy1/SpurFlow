#!/usr/bin/env bash
# Isolated SimpleTuner env for Flux Kontext SFT.
# Does NOT touch the EditFlow `arcflow` conda env used by existing train_*.sh.
#
# First-time install:
#   bash /mnt/afs_zhangyunzhe/EditFlow/setup_simpletuner_env.sh --install
#
# Activate only (also safe to source from train script):
#   source /mnt/afs_zhangyunzhe/EditFlow/setup_simpletuner_env.sh

ENV_NAME="${SIMPLETUNER_ENV_NAME:-simpletuner}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_zhangyunzhe/miniconda3}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${SIMPLETUNER_WORKDIR:-${PROJECT_DIR}/simpletuner_kontext_sft}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/afs_zhangyunzhe/.cache/huggingface}"
export SIMPLETUNER_LOG_LEVEL="${SIMPLETUNER_LOG_LEVEL:-INFO}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

_do_install=0
if [[ "${1:-}" == "--install" ]]; then
    _do_install=1
fi

_pip_retry() {
    local tries="${PIP_RETRIES:-3}"
    local n=1
    until python -m pip "$@"; do
        if [[ "${n}" -ge "${tries}" ]]; then
            echo "[setup] pip failed after ${tries} attempts: pip $*" >&2
            return 1
        fi
        echo "[setup] pip retry ${n}/${tries}: pip $*" >&2
        n=$((n + 1))
        sleep 5
    done
}

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ "${_do_install}" != "1" ]]; then
        echo "[setup] conda env '${ENV_NAME}' missing. Run:" >&2
        echo "        bash ${PROJECT_DIR}/setup_simpletuner_env.sh --install" >&2
        return 1 2>/dev/null || exit 1
    fi
    echo "[setup] Creating conda env ${ENV_NAME} (python=3.12) ..."
    conda create -y -n "${ENV_NAME}" python=3.12
fi

conda activate "${ENV_NAME}"

if [[ "${_do_install}" == "1" ]]; then
    echo "[setup] Installing simpletuner[cuda] into env ${ENV_NAME} ..."
    _pip_retry install -U pip setuptools wheel
    if ! python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        _pip_retry install --index-url https://download.pytorch.org/whl/cu128 \
            torch torchvision torchaudio
    fi
    _pip_retry install 'simpletuner[cuda]'
fi

if ! command -v simpletuner >/dev/null 2>&1; then
    echo "[setup] 'simpletuner' not found in env '${ENV_NAME}'. Run:" >&2
    echo "        bash ${PROJECT_DIR}/setup_simpletuner_env.sh --install" >&2
    return 1 2>/dev/null || exit 1
fi

if ! python -c "import torch" >/dev/null 2>&1; then
    echo "[setup] torch missing in env '${ENV_NAME}'. Re-run:" >&2
    echo "        bash ${PROJECT_DIR}/setup_simpletuner_env.sh --install" >&2
    return 1 2>/dev/null || exit 1
fi

cd "${WORKDIR}"
echo "[setup] env=${ENV_NAME}  python=$(command -v python)  workdir=${WORKDIR}"
python -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
