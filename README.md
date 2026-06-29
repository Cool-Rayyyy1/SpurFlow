# EditFlow

EditFlow distills **FLUX.1-Kontext-dev** into a few-step image editor using the ArcFlow non-linear flow trajectory framework.

## Dataset

Training uses [pico-banana-400k](https://huggingface.co/datasets/pico-banana-400k) at `/mnt/afs_zhangyunzhe/dataset/pico-banana-400k`:

- **Source image**: `local_input_image` under `source_images/`
- **Edited target**: `output_image` under `edited_images/`
- **Instruction**: `text`

## Model

- Teacher / base: `FLUX.1-Kontext-dev`
- Student: ArcFlow adapter (K=16 Gaussians, LoRA) on Kontext transformer
- Conditioning: reference image latents concatenated along sequence dim (Kontext-style)

## Training

```bash
source /mnt/afs_zhangyunzhe/EditFlow/setup_env.sh
bash train_flux.sh
```

Common overrides:

```bash
NFE=2 NPROC=2 EVAL=1 bash train_flux.sh
NFE=4 CKPT_INTERVAL=500 EVAL=0 bash train_flux.sh
```

Checkpoints: `checkpoints/gmkontext_k16_{NFE}nfe_pico400k/`  
Logs / samples: `work_dirs/gmkontext_k16_{NFE}nfe_pico400k/`

## Config

Main config: `configs/kontext/editflux_2nfe_k16.py`

Derived from ArcFlow FLUX distillation with these edits:

- `LatentDiffusionImageEdit` encodes source + target images
- `ImageEdit` dataset reads pico-banana jsonl
- Kontext transformer (`patch_size=1`) replaces FLUX.1-dev

## Export / Inference

Use the same export script as ArcFlow after training:

```bash
python export_arcflow_to_diffusers.py configs/kontext/editflux_2nfe_k16.py \
  --ckpt checkpoints/gmkontext_k16_2nfe_pico400k/latest.pth \
  --out-dir checkpoints/gmkontext_k16_2nfe_pico400k/diffusers_adapter
```

For Kontext inference, load the adapter into `FluxKontextPipeline` (see ArcFlow `inference_flux.py` and adapt for Kontext + reference image).
