#!/usr/bin/env bash
# Render the display assets one checkpoint per process.
#
# render_display_assets.py can loop over several tags itself, but the model is
# not fully released between them (del + empty_cache still left ~79 GiB held),
# so the third build_student_model OOMs on a single 80 GB card. One process per
# tag lets the OS reclaim the memory instead.
#
# --skip_existing means finished tags cost nothing on a rerun.
set -uo pipefail

source /mnt/afs_gaochengmin/anaconda3/etc/profile.d/conda.sh
conda activate arcflow
cd /mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17/EditFlow

export CUDA_VISIBLE_DEVICES=0
export STUDENT_RESIZE_MODE=qwen
export QWEN_MODEL_PATH=/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511
export HF_HOME=/mnt/afs_gaochengmin/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TAGS=("${@:-}")
if [[ -z "${TAGS[0]}" ]]; then
  TAGS=(oss_pico_iter2400 oss_pico_iter2800 gan_pretrain_iter2800 arcflow_qwen
        qwen_lightning_4step telestyle_qie2511_4step)
fi

rc=0
for tag in "${TAGS[@]}"; do
  echo "############### ${tag} ###############"
  python evaluation/display/render_display_assets.py \
    --only_tags "${tag}" --num_seeds 100 --skip_existing \
    2>&1 | grep -vE "unexpected key|^Loading|^load checkpoint|shards|pipeline components|^ *$"
  status=${PIPESTATUS[0]}
  if [[ ${status} -ne 0 ]]; then
    echo "!!! ${tag} exited ${status}; continuing with the remaining tags"
    rc=${status}
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done

echo "############### done (rc=${rc}) ###############"
for tag in "${TAGS[@]}"; do
  d="work_dirs/display_new/${tag}"
  printf "%-24s %4d / 900\n" "${tag}" "$(find "${d}" -name 'tgt_seed*.png' 2>/dev/null | wc -l)"
done
df -h . | tail -1
exit ${rc}
