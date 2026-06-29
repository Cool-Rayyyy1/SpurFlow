# Hugging Face mirror (China)
export HF_ENDPOINT='https://hf-mirror.com'

# Local offline assets
export KONTEXT_MODEL_PATH='/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev'
export PICO_BANANA_PATH='/mnt/afs_zhangyunzhe/dataset/pico-banana-400k'
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

# Project on PYTHONPATH
export PYTHONPATH="/mnt/afs_zhangyunzhe/EditFlow:${PYTHONPATH:-}"

# Activate: source /mnt/afs_zhangyunzhe/EditFlow/setup_env.sh
