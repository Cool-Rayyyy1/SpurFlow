# Flux.1-Kontext SFT with pico-banana via [SimpleTuner](https://github.com/bghira/SimpleTuner)

EditFlow distillation scripts (`train_flux.sh`, `train_flux_edit_*.sh`, `setup_env.sh`) are untouched.
This folder is a self-contained SimpleTuner workspace.

## Why SimpleTuner

Kontext needs paired **reference → edit** conditioning. SimpleTuner supports:

- `model_flavour=kontext`
- `conditioning_type=reference_strict|reference_loose`
- LoRA / LyCORIS / full-rank

Official note: full-rank is usually not worth it vs LoKr; default here is PEFT LoRA on 8×A100.

## Data layout (auto-built)

From `jsonl/sft_with_local_source_image_path.jsonl`:

| field | role |
| --- | --- |
| `local_input_image` | reference (source) |
| `output_image` under `edited_images/` | edit target |
| `text` | caption / instruction |

SimpleTuner requires **identical filenames** in `edit/` and `reference/`, so we symlink/hardlink:

```
$DATA_ROOT/simpletuner_pairs/
  train/edit/{id}.png + {id}.txt
  train/reference/{id}.png
  val/edit/...
  val/reference/...
```

## One-time setup

```bash
cd /mnt/afs_zhangyunzhe/EditFlow

# isolated conda env (python 3.12); does not modify arcflow
bash setup_simpletuner_env.sh --install

# smoke subset first if disk is tight (VAE cache is large)
MAX_SAMPLES=2000 bash prepare_pico_banana_simpletuner.sh

# full set (symlink/hardlink only; still plan disk for caches)
# bash prepare_pico_banana_simpletuner.sh
```

## Train

```bash
bash train_flux_kontext_sft_simpletuner.sh          # 8 GPUs
bash train_flux_kontext_sft_simpletuner.sh 2       # 2 GPUs

GPU_IDS=0,1 NUM_GPUS=2 MAX_TRAIN_STEPS=2000 LORA_RANK=16 \
  bash train_flux_kontext_sft_simpletuner.sh

# LyCORIS LoKr (often better than plain LoRA on Kontext)
LORA_TYPE=lycoris bash train_flux_kontext_sft_simpletuner.sh
```

Outputs: `work_dirs/simpletuner_kontext_sft/output/<run>/`  
Caches: `work_dirs/simpletuner_kontext_sft/cache/`

## Knobs

| env | default | meaning |
| --- | --- | --- |
| `NUM_GPUS` / positional | 8 | DDP processes |
| `GPU_IDS` | 0-7 | `CUDA_VISIBLE_DEVICES` |
| `MODEL_TYPE` | `lora` | `lora` or `full` |
| `LORA_TYPE` | `standard` | `standard` or `lycoris` |
| `LORA_RANK` | 32 | PEFT rank |
| `BASE_PRECISION` | `no_change` | full bf16 weights; try `int8-quanto` if VRAM tight |
| `MAX_TRAIN_STEPS` | 10000 | steps |
| `PAIR_ROOT` | `$DATA_ROOT/simpletuner_pairs` | prepared pairs |
| `KONTEXT_MODEL` | `.../FLUX.1-Kontext-dev` | local checkpoint |

## Disk warning

Pairing itself is cheap (links + `.txt`). SimpleTuner VAE/text embedding caches for ~257k pairs can need hundreds of GB. Prefer `MAX_SAMPLES=...` until you confirm free space.
