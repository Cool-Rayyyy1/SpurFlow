#!/usr/bin/env python3
"""Run teacher x_0->noise inversion at multiple step counts and save decoded visuals."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EDITFLOW_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = EDITFLOW_ROOT / "tools"
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from test_teacher_invert_steps import (  # noqa: E402
    build_teacher,
    build_text_encoder,
    build_timestep_sampler,
    build_vae,
    diff_stats,
    encode_images,
    load_random_samples,
    load_rgb,
    run_inversion,
    to_tensor,
)
from types import SimpleNamespace  # noqa: E402


def parse_step_list(raw: str) -> list[int]:
    steps = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not steps:
        raise ValueError("Empty --step-list")
    return steps


def save_rgb_array(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


@torch.no_grad()
def decode_latent_to_image(vae, latent: torch.Tensor) -> torch.Tensor:
    vae_dtype = vae.dtype if hasattr(vae, "dtype") else next(vae.parameters()).dtype
    img = vae.decode(latent.to(vae_dtype)).float()
    return (img / 2 + 0.5).clamp(0, 1)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    arr = (
        tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize teacher inversion eps at multiple step counts")
    p.add_argument("--model-path", default="/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev")
    p.add_argument("--data-root", default="/mnt/afs_zhangyunzhe/dataset/pico-banana-400k")
    p.add_argument("--jsonl-path", default="jsonl/sft_with_local_source_image_path.jsonl")
    p.add_argument("--edited-images-dir", default="edited_images")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--step-list", default="5,10,15,20,28")
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-dir",
        default="/mnt/afs_zhangyunzhe/EditFlow/work_dirs/teacher_invert_steps_viz",
    )
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    step_list = parse_step_list(args.step_list)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print("Inversion: x_0=edited latent, Kontext cond=source + prompt")
    print(f"Step counts: {step_list}")
    print(f"Output dir: {output_dir}")

    vae = build_vae(args.model_path, device)
    text_encoder = build_text_encoder(args.model_path, device)
    teacher = build_teacher(args.model_path, device)
    timestep_sampler = build_timestep_sampler()
    inverter = SimpleNamespace(timestep_sampler=timestep_sampler, num_timesteps=1)

    all_records = []
    ref_steps = step_list[-1]

    for i, sample in enumerate(samples):
        sample_dir = output_dir / f"{i:03d}_{sample['sample_id']}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        source_arr = load_rgb(sample["source_path"])
        edited_arr = load_rgb(sample["target_path"])
        save_rgb_array(source_arr, sample_dir / "source.png")
        save_rgb_array(edited_arr, sample_dir / "edited.png")
        (sample_dir / "prompt.txt").write_text(sample["prompt"], encoding="utf-8")

        source_img = to_tensor(source_arr, args.image_size).unsqueeze(0).to(device)
        edited_img = to_tensor(edited_arr, args.image_size).unsqueeze(0).to(device)

        x_0 = encode_images(vae, edited_img)
        source_lat = encode_images(vae, source_img)
        seq_len = x_0.shape[2:].numel()

        prompt_embed = text_encoder(prompt=[sample["prompt"]])
        teacher_kwargs = {k: v.to(device) for k, v in prompt_embed.items()}
        teacher_kwargs["image_latents"] = source_lat
        teacher_kwargs["guidance"] = torch.full(
            (1,), args.guidance_scale, dtype=torch.float32, device=device)

        eps_by_steps = {}
        for num_steps in step_list:
            eps = run_inversion(
                inverter, teacher, x_0, teacher_kwargs, num_steps, seq_len)
            eps_by_steps[num_steps] = eps
            decoded = decode_latent_to_image(vae, eps)
            save_tensor_image(decoded, sample_dir / f"eps_steps{num_steps}.png")

        step_stats = {}
        ref_eps = eps_by_steps[ref_steps]
        for num_steps in step_list:
            if num_steps == ref_steps:
                continue
            step_stats[num_steps] = diff_stats(ref_eps, eps_by_steps[num_steps])

        record = dict(
            index=i,
            sample_id=sample["sample_id"],
            source_path=sample["source_path"],
            edited_path=sample["target_path"],
            output_dir=str(sample_dir),
            step_list=step_list,
            diff_vs_ref=step_stats,
        )
        all_records.append(record)

        diffs = "  ".join(
            f"{s}v{ref_steps}={step_stats[s]['l1_mean']:.4f}"
            for s in step_list if s != ref_steps
        )
        print(f"[{i + 1:02d}/{len(samples)}] id={sample['sample_id']}  {diffs}")
        print(f"         saved -> {sample_dir}")

    summary = dict(
        num_samples=len(all_records),
        step_list=step_list,
        ref_steps=ref_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        samples=all_records,
    )
    for num_steps in step_list:
        if num_steps == ref_steps:
            continue
        vals = [r["diff_vs_ref"][num_steps]["l1_mean"] for r in all_records]
        summary[f"l1_mean_{num_steps}_vs_{ref_steps}"] = float(np.mean(vals))
        summary[f"l1_std_{num_steps}_vs_{ref_steps}"] = float(np.std(vals))

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("")
    print("=" * 72)
    print(f"Saved visuals under {output_dir}")
    for num_steps in step_list:
        if num_steps == ref_steps:
            continue
        key_mean = f"l1_mean_{num_steps}_vs_{ref_steps}"
        key_std = f"l1_std_{num_steps}_vs_{ref_steps}"
        print(f"  L1({ref_steps}-{num_steps}): mean={summary[key_mean]:.6f}  std={summary[key_std]:.6f}")
    print(f"Summary JSON -> {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
