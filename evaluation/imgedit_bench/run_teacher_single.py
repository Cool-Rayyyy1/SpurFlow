#!/usr/bin/env python3
"""Single-image FLUX.1-Kontext (teacher) edit inference.

Runs one local image + one prompt through FluxKontextPipeline and saves:
    0_src.png        preprocessed source (kontext resize, RGBA if input has alpha)
    1_edited.png     final teacher output after all steps
    2_x0_step1.png   predicted x0 at denoising step 1
    3_x0_step2.png   predicted x0 at denoising step 2  (2-step run)
    prompt.txt       prompt and run settings

x0 at each step uses flow-matching: x0 = latents - sigma * noise_pred (before scheduler update).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("STUDENT_RESIZE_MODE", "kontext")

import torch
from PIL import Image

EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_spurflow_imgedit_infer import build_teacher_pipeline  # noqa: E402

DEFAULT_MODEL = os.environ.get(
    "KONTEXT_MODEL_PATH", "/path/to/pretrained_models/FLUX.1-Kontext-dev")
DEFAULT_IMAGE = PROJECT_ROOT / "pikachu.png"
DEFAULT_PROMPT = (
    "Put an Ash Ketchum-style Pokemon Trainer cap on Pikachu, with a red-and-white "
    "design and a green emblem on the front.")
DEFAULT_OUTPUT = EVAL_ROOT / "outputs/teacher_single/pikachu_2step"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--num_inference_steps", type=int, default=2)
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu_offload", action="store_true")
    return p.parse_args()


def load_image_rgba(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        return img.convert("RGBA")
    return img.convert("RGB")


def preprocess_image_kontext(image: Image.Image) -> Image.Image:
    from lakonlab.datasets.image_edit import _pick_kontext_resolution

    width, height = image.size
    bucket_w, bucket_h = _pick_kontext_resolution(width, height)
    if (width, height) == (bucket_w, bucket_h):
        return image
    return image.resize((bucket_w, bucket_h), Image.Resampling.BICUBIC)


def to_pipe_input(image: Image.Image) -> Image.Image:
    """Pipeline expects RGB; flatten alpha onto white only for the forward pass."""
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        flat = Image.alpha_composite(bg, image)
        return flat.convert("RGB")
    return image.convert("RGB")


def decode_packed_x0(pipe, x0_packed: torch.Tensor, height: int, width: int) -> Image.Image:
    latents = pipe._unpack_latents(x0_packed, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.inference_mode()
def run_teacher_with_x0(
    pipe,
    image: Image.Image,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
    height: int,
    width: int,
) -> Tuple[Image.Image, List[Image.Image], List[dict]]:
    """Run teacher inference and capture per-step x0 predictions."""
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    x0_records: List[dict] = []
    orig_step = pipe.scheduler.step

    def hooked_step(model_output, timestep, sample, *args, **kwargs):
        idx = pipe.scheduler.step_index
        sigma = pipe.scheduler.sigmas[idx]
        x0_packed = sample - sigma * model_output
        x0_records.append({
            "step": len(x0_records) + 1,
            "sigma": float(sigma.detach().cpu()),
            "timestep": float(timestep.detach().cpu()) if torch.is_tensor(timestep) else float(timestep),
            "x0_packed": x0_packed.detach().clone(),
        })
        return orig_step(model_output, timestep, sample, *args, **kwargs)

    pipe.scheduler.step = hooked_step
    try:
        out = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            height=height,
            width=width,
        )
        edited = out.images[0]
    finally:
        pipe.scheduler.step = orig_step

    x0_pils = [decode_packed_x0(pipe, rec["x0_packed"], height, width) for rec in x0_records]
    return edited, x0_pils, x0_records


def main() -> None:
    args = parse_args()
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[teacher-single] model={args.model_path}")
    print(f"[teacher-single] image={args.image}")
    print(f"[teacher-single] prompt={args.prompt}")
    print(f"[teacher-single] steps={args.num_inference_steps} guidance={args.guidance_scale}")
    print(f"[teacher-single] output={args.output_dir}", flush=True)

    src_raw = load_image_rgba(args.image)
    src_pil = preprocess_image_kontext(src_raw)
    width, height = src_pil.size
    pipe_input = to_pipe_input(src_pil)

    pipe = build_teacher_pipeline(args.model_path, device, args.cpu_offload)
    edited, x0_pils, x0_records = run_teacher_with_x0(
        pipe, pipe_input, args.prompt,
        args.num_inference_steps, args.guidance_scale, args.seed,
        height=height, width=width)

    src_pil.save(args.output_dir / "0_src.png")
    edited.save(args.output_dir / "1_edited.png")
    for i, x0_img in enumerate(x0_pils, start=1):
        x0_img.save(args.output_dir / f"{i + 1}_x0_step{i}.png")

    meta_lines = [
        f"model: {args.model_path}",
        f"image: {args.image}",
        f"prompt: {args.prompt}",
        f"num_inference_steps: {args.num_inference_steps}",
        f"guidance_scale: {args.guidance_scale}",
        f"seed: {args.seed}",
        f"src_mode: {src_pil.mode} (transparent bg preserved in 0_src.png when RGBA)",
        f"resize_mode: kontext",
        f"height: {height} width: {width}",
        "",
        "Per-step x0: x0 = latents - sigma * noise_pred (flow-matching, before scheduler update)",
    ]
    for rec in x0_records:
        meta_lines.append(
            f"  step {rec['step']}: sigma={rec['sigma']:.6f} timestep={rec['timestep']:.2f}"
            f" -> {rec['step'] + 1}_x0_step{rec['step']}.png")
    (args.output_dir / "prompt.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    if len(x0_pils) != args.num_inference_steps:
        print(f"[warn] expected {args.num_inference_steps} x0 images, got {len(x0_pils)}", flush=True)

    print(f"\n[teacher-single] Done -> {args.output_dir}")
    for p in sorted(args.output_dir.iterdir()):
        print(f"   {p.name}")


if __name__ == "__main__":
    main()
