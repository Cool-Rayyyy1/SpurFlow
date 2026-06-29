# Kontext x0 diff — dataset layout

Extract `train_0.tar.gz` / `train_1.tar.gz` under e.g. `/mnt/afs_zhangyunzhe/dataset/kontext_edit/`.

Supported formats:

## Option A: `manifest.jsonl` (recommended)

Each line is JSON with **reference image path** and **edit prompt**:

```json
{"id": "00001", "reference": "images/src_00001.png", "prompt": "Add a red hat."}
{"id": "00002", "reference_image": "source/2.jpg", "instruction": "Make the sky sunset orange."}
```

Accepted field names:

- reference: `reference`, `reference_image`, `source_image`, `source`, `image`, `input_image`
- prompt: `prompt`, `instruction`, `edit_prompt`, `caption`

## Option B: paired files

```
kontext_edit/
  reference/
    00001.png
    00001.txt   # edit prompt (one line)
```

## Run (pico-banana-400k, 2nd GPU / CUDA:1, 25 steps)

Manifest: `dataset/pico-banana-400k/manifests/kontext_x0_diff_10.jsonl` (10 random src/edit/prompt triplets).

Each sample prints **50 values**: 25 steps × `L1(x0_hat - x_src)` + 25 × `L1(x0_hat - x_edit)`.

```bash
# 2nd GPU (CUDA index 1) — latent space
bash run_kontext_x0_diff_latent.sh

# 2nd GPU (CUDA index 1) — decoded pixel space (VAE decode x0_hat)
bash run_kontext_x0_diff_image.sh
```

Logs: `work_dirs/kontext_x0_diff/latent_run.log` and `image_run.log`.

If manifest is missing, scripts fall back to `--demo` (Kontext `teaser.png`).
