#!/usr/bin/env bash
# Deprecated alias: Qwen non-alpha DINO GAN now uses split-stage (no extra 2-NFE rollout),
# matching alph_dino_gan. Prefer:
#   bash train_flux_edit_fixedeps_data_qwen_split_stage_dino_gan.sh
echo "[warn] train_flux_edit_fixedeps_data_qwen_step2_dino_gan.sh redirects to split_stage_dino_gan (shared 2-NFE unroll, no extra rollout)." >&2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_flux_edit_fixedeps_data_qwen_split_stage_dino_gan.sh" "$@"
