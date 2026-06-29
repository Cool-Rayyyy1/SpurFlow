#!/usr/bin/env python3
"""Single-image FLUX Kontext teacher inference (no EditFlow adapter)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

EDITFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))

DEFAULT_MODEL = "/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev"


def parse_prompt(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty prompt.")
    if raw.startswith("["):
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
    return raw


def build_pipeline(model_path: str, device: str):
    from diffusers import FluxKontextPipeline, FlowMatchEulerDiscreteScheduler

    pipe = FluxKontextPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config, shift=3.2, shift_terminal=None, use_dynamic_shifting=False)
    return pipe.to(device)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run FLUX Kontext teacher on one edit pair.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, required=True, help="Reference / source image.")
    parser.add_argument("--prompt", type=str, default="", help="Edit instruction.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Prompt text file (JSON list ok).")
    parser.add_argument("--output", type=Path, required=True, help="Output image path.")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.prompt_file is not None:
        prompt = parse_prompt(args.prompt_file.read_text(encoding="utf-8"))
    elif args.prompt:
        prompt = args.prompt
    else:
        raise ValueError("Provide --prompt or --prompt-file.")

    source = Image.open(args.source).convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"Model: {args.model_path}")
    print(f"Source: {args.source}")
    print(f"Steps: {args.num_inference_steps}, guidance: {args.guidance_scale}, seed: {args.seed}")
    print(f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

    pipe = build_pipeline(args.model_path, device)
    generator = torch.Generator(device=pipe._execution_device).manual_seed(args.seed)
    result = pipe(
        image=source,
        prompt=prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    result.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
