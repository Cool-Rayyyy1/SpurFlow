# SpurFlow

SpurFlow distills FLUX.1 Kontext into a 2-step image editor. A learned alpha map mixes the reference latent into the student velocity, and a second stage adds a source-conditional discriminator on the decoded edit.

The teacher is frozen FLUX.1 Kontext with distilled classifier-free guidance. The student is a LoRA-adapted Kontext transformer with K=16 Gaussian heads and an alpha head. The reference image is encoded to latents and concatenated along the sequence dimension, in the same layout as Kontext.

## Data

Training uses a paired edit dataset. Each example is a source image, an edited image, and an instruction. Index it with `metadata.jsonl`; the launcher builds that file when it is missing. The last 128 rows are held out for validation.

## Training

```bash
export KONTEXT_MODEL=/path/to/FLUX.1-Kontext-dev
export DATA_ROOT=/path/to/dataset
export CONDA_ROOT=/path/to/anaconda3
source setup_env.sh
```

Warmup learns the alpha editor with no discriminator, starting from the Kontext backbone and randomly initialized alpha and delta heads:

```bash
bash train_flux_kontext_warmup.sh
```

Formal training loads that checkpoint and continues with one shared 2-step rollout plus a GAN. `DINOV3_MODEL` is a DINOv3 ViT-L/16 weight file.

```bash
export DINOV3_MODEL=/path/to/dinov3-vitl16/model.safetensors
export PRETRAIN_CKPT=checkpoints/spurflow_warmup/<run_id>/latest.pth
bash train_flux_kontext.sh
```

Useful overrides:

```bash
NFE=2 TOTAL_ITERS=50000 bash train_flux_kontext_warmup.sh
NUM_GPUS=8 GPU_IDS=0,1,2,3,4,5,6,7 bash train_flux_kontext.sh
GAN_WEIGHT=0.05 GAN_WARMUP_ITERS=0 GAN_RAMP_ITERS=0 bash train_flux_kontext.sh
FRESH=1 bash train_flux_kontext_warmup.sh
```

Checkpoints and samples:

- `checkpoints/spurflow_warmup/`
- `checkpoints/spurflow/`
- `work_dirs/spurflow_warmup/`
- `work_dirs/spurflow/`

## What the formal stage optimizes

One 2-step split-stage rollout, not a second independent sampling pass:

1. At `t=1`, the student matches the teacher.
2. From that endpoint to `t=0`, the student matches the teacher, matches the direct edit residual, and receives a GAN loss on the VAE-decoded image.

The discriminator is a frozen DINOv3 backbone with a trainable head. A real example is the reference image paired with the dataset edit. A fake example is the same reference paired with the student decode. Both views share a full-frame crop and a local crop taken from the low-alpha edit region. Features are channel-concatenated before the head. The GAN term is a softplus logistic loss with weight `GAN_WEIGHT` (default `0.05`).

Configs:

- Warmup: `configs/kontext/editflux_kontext_warmup.py`
- Formal: `configs/kontext/editflux_kontext_split_stage_alpha_dino_gan.py`

Launch scripts override model, data, and checkpoint paths from the environment.
