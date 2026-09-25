# Optional local paths. Export these before sourcing, or edit the defaults.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export KONTEXT_MODEL_PATH="${KONTEXT_MODEL_PATH:-}"
export PICO_BANANA_PATH="${PICO_BANANA_PATH:-}"
export IMGEDIT_BENCH_ROOT="${IMGEDIT_BENCH_ROOT:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"

_EDITFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_EDITFLOW_DIR}:${PYTHONPATH:-}"
