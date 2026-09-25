# Hugging Face mirror (China)
export HF_ENDPOINT='https://hf-mirror.com'

# Local offline assets
export HF_HOME='/mnt/afs_gaochengmin/.cache/huggingface'
export KONTEXT_MODEL_PATH='/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev'
export QWEN_MODEL_PATH='/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511'
export KLEIN_MODEL_PATH='/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B'
export PICO_BANANA_PATH='/mnt/afs_gaochengmin/data/pico-banana-400k'
export IMGEDIT_BENCH_ROOT='/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark'
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

# Project on PYTHONPATH (resolved from this file's own location)
_EDITFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_EDITFLOW_DIR}:${PYTHONPATH:-}"

# Activate: source <project_dir>/setup_env.sh
