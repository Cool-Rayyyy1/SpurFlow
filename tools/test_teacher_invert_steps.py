#!/usr/bin/env python3
"""Compare teacher x_0->noise flow inversion at different step counts (EditFlow path).

Kontext inversion setup (matches training):
  - x_0: edited image VAE latent (flow starts here, t=0 -> t=1)
  - teacher kwargs: source image_latents + prompt + guidance (Kontext conditioning)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

EDITFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))

DEFAULT_MODEL = "/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev"
DEFAULT_DATA = "/mnt/afs_zhangyunzhe/dataset/pico-banana-400k"
DEFAULT_JSONL = "jsonl/sft_with_local_source_image_path.jsonl"
DEFAULT_EDITED_DIR = "edited_images"

from lakonlab.models.diffusions.arcflow import ArcFlowImitationBase  # noqa: E402


def resize_center_crop(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == size and w == size:
        return image
    scale = size / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    pil = Image.fromarray(image)
    pil = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
    arr = np.array(pil, dtype=np.uint8)
    top = max((arr.shape[0] - size) // 2, 0)
    left = max((arr.shape[1] - size) // 2, 0)
    return arr[top:top + size, left:left + size]


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    image = resize_center_crop(image, image_size)
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def resolve_source(data_root: Path, raw: str) -> Path | None:
    path = Path(raw).expanduser()
    if path.is_file():
        return path
    for cand in [data_root / raw, data_root / "source_images" / raw]:
        if cand.is_file():
            return cand
    return None


def resolve_target(data_root: Path, edited_root: Path, raw: str) -> Path | None:
    path = Path(raw).expanduser()
    if path.is_file():
        return path
    for cand in [edited_root / raw, data_root / raw]:
        if cand.is_file():
            return cand
    return None


def load_random_samples(
        data_root: Path,
        jsonl_path: Path,
        edited_root: Path,
        num_samples: int,
        seed: int,
) -> list[dict]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(rows)

    samples = []
    for row in rows:
        if len(samples) >= num_samples:
            break
        source_raw = row.get("local_input_image") or row.get("source_image") or row.get("image")
        target_raw = row.get("output_image") or row.get("edited_image")
        prompt = row.get("text") or row.get("prompt") or row.get("instruction")
        if source_raw is None or target_raw is None or prompt is None:
            continue
        source_path = resolve_source(data_root, source_raw)
        target_path = resolve_target(data_root, edited_root, target_raw)
        if source_path is None or target_path is None:
            continue
        samples.append(
            dict(
                sample_id=str(row.get("id", len(samples))),
                source_path=str(source_path),
                target_path=str(target_path),
                prompt=str(prompt),
            )
        )

    if len(samples) < num_samples:
        raise RuntimeError(
            f'Only found {len(samples)} valid samples (need {num_samples}).')
    return samples


def build_teacher(model_path: str, device: torch.device):
    from mmgen.models.builder import build_module

    transformer = f"{model_path}/transformer/diffusion_pytorch_model.safetensors.index.json"
    teacher = build_module(
        dict(
            type="GaussianFlow",
            denoising=dict(
                type="FluxTransformer2DModel",
                patch_size=2,
                freeze=True,
                pretrained=transformer,
                in_channels=64,
                num_layers=19,
                num_single_layers=38,
                attention_head_dim=128,
                num_attention_heads=24,
                joint_attention_dim=4096,
                pooled_projection_dim=768,
                guidance_embeds=True,
                torch_dtype="bfloat16",
            ),
            num_timesteps=1,
            denoising_mean_mode="U",
        )
    )
    return teacher.to(device).eval()


def build_vae(model_path: str, device: torch.device):
    from mmgen.models.builder import build_module

    vae = build_module(
        dict(
            type="PretrainedVAE",
            from_pretrained=model_path,
            subfolder="vae",
            freeze=True,
            torch_dtype="bfloat16",
        )
    )
    return vae.to(device).eval()


def build_text_encoder(model_path: str, device: torch.device):
    from mmgen.models.builder import build_module

    text_encoder = build_module(
        dict(
            type="PretrainedFluxTextEncoder",
            from_pretrained=model_path,
        )
    )
    return text_encoder.to(device).eval()


def build_timestep_sampler():
    from mmgen.models.builder import build_module

    return build_module(
        dict(type="ContinuousTimeStepSampler", shift=3.2, logit_normal_enable=False),
        default_args=dict(num_timesteps=1),
    )


@torch.no_grad()
def encode_images(vae, images: torch.Tensor) -> torch.Tensor:
    vae_dtype = vae.dtype if hasattr(vae, "dtype") else next(vae.parameters()).dtype
    return vae.encode((images * 2 - 1).to(vae_dtype)).float()


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a - b).float()
    abs_diff = diff.abs()
    denom = b.float().abs().mean().clamp(min=1e-6)
    return dict(
        l1_mean=float(abs_diff.mean().item()),
        l2_mean=float(torch.sqrt((diff ** 2).mean()).item()),
        max_abs=float(abs_diff.max().item()),
        rel_l1_mean=float((abs_diff.mean() / denom).item()),
    )


@torch.no_grad()
def run_inversion(
        inverter,
        teacher,
        x_0,
        teacher_kwargs,
        num_steps: int,
        seq_len: int,
) -> torch.Tensor:
    return ArcFlowImitationBase.teacher_invert_x0_to_noise(
        inverter,
        teacher,
        x_0,
        teacher_kwargs,
        num_steps=num_steps,
        seq_len=seq_len,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare teacher x_0->noise inversion: 10 steps vs 28 steps")
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--data-root", default=DEFAULT_DATA)
    p.add_argument("--jsonl-path", default=DEFAULT_JSONL)
    p.add_argument("--edited-images-dir", default=DEFAULT_EDITED_DIR)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--steps-a", type=int, default=10)
    p.add_argument("--steps-b", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-json", default=None)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    data_root = Path(args.data_root)
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.is_absolute():
        jsonl_path = data_root / jsonl_path
    edited_root = Path(args.edited_images_dir)
    if not edited_root.is_absolute():
        edited_root = data_root / edited_root

    samples = load_random_samples(
        data_root, jsonl_path, edited_root, args.num_samples, args.seed)

    print(f"Loaded {len(samples)} pico samples.")
    print("Inversion: x_0=edited latent, Kontext cond=source image_latents + prompt")
    print(f"Compare {args.steps_a}-step vs {args.steps_b}-step inversion")
    print(f"teacher guidance: {args.guidance_scale}")

    vae = build_vae(args.model_path, device)
    text_encoder = build_text_encoder(args.model_path, device)
    teacher = build_teacher(args.model_path, device)
    timestep_sampler = build_timestep_sampler()
    inverter = SimpleNamespace(timestep_sampler=timestep_sampler, num_timesteps=1)

    per_sample = []
    all_l1 = []
    all_rel_l1 = []

    for i, sample in enumerate(samples):
        source_img = to_tensor(load_rgb(sample["source_path"]), args.image_size).unsqueeze(0).to(device)
        edited_img = to_tensor(load_rgb(sample["target_path"]), args.image_size).unsqueeze(0).to(device)

        x_0 = encode_images(vae, edited_img)
        source_lat = encode_images(vae, source_img)
        seq_len = x_0.shape[2:].numel()

        prompt_embed = text_encoder(prompt=[sample["prompt"]])
        teacher_kwargs = {k: v.to(device) for k, v in prompt_embed.items()}
        teacher_kwargs["image_latents"] = source_lat
        teacher_kwargs["guidance"] = torch.full(
            (1,), args.guidance_scale, dtype=torch.float32, device=device)

        eps_a = run_inversion(
            inverter, teacher, x_0, teacher_kwargs, args.steps_a, seq_len)
        eps_b = run_inversion(
            inverter, teacher, x_0, teacher_kwargs, args.steps_b, seq_len)
        stats = diff_stats(eps_b, eps_a)

        print(
            f"[{i + 1:02d}/{len(samples)}] id={sample['sample_id']}  "
            f"L1={stats['l1_mean']:.6f}  rel_L1={stats['rel_l1_mean']:.6f}  "
            f"L2={stats['l2_mean']:.6f}  max={stats['max_abs']:.6f}")
        print(f"         source(cond)={sample['source_path']}")
        print(f"         edited(x_0)={sample['target_path']}")

        per_sample.append(dict(sample_id=sample["sample_id"], **stats))
        all_l1.append(stats["l1_mean"])
        all_rel_l1.append(stats["rel_l1_mean"])

    summary = dict(
        num_samples=len(per_sample),
        steps_a=args.steps_a,
        steps_b=args.steps_b,
        diff_l1_mean=float(np.mean(all_l1)),
        diff_l1_std=float(np.std(all_l1)),
        diff_rel_l1_mean=float(np.mean(all_rel_l1)),
        diff_rel_l1_std=float(np.std(all_rel_l1)),
        per_sample=per_sample,
    )

    print("")
    print("=" * 72)
    print(
        f"SUMMARY ({args.steps_b} - {args.steps_a} steps) over {len(per_sample)} samples")
    print(f"  L1 mean : {summary['diff_l1_mean']:.6f}  std: {summary['diff_l1_std']:.6f}")
    print(f"  rel L1  : {summary['diff_rel_l1_mean']:.6f}  std: {summary['diff_rel_l1_std']:.6f}")
    print("=" * 72)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON -> {out_path}")


if __name__ == "__main__":
    main()
