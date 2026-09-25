"""Official FLUX.2-klein-base-9B pipeline helpers.

HF: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B
Local: /mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B

Official undistilled base (not the 4-step distilled klein-9B):
  num_inference_steps=50
  guidance_scale=4.0

Resize: ImageEdit resize_mode='flux2' (~1MP area-cap, snap to 16px, bicubic).
"""

from __future__ import annotations

import os

import torch
from PIL import Image

from official_common import resize_for_mode

DEFAULT_MODEL = os.environ.get(
    "KLEIN_MODEL_PATH",
    "/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B",
)
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 4.0
RESIZE_MODE = "flux2"
HF_ID = "black-forest-labs/FLUX.2-klein-base-9B"
RUN_TAG_GEDIT = "flux2_klein_base_9b_gedit"
RUN_TAG_IMGEDIT = "flux2_klein_base_9b_imgedit"


def build_pipeline(model_path: str, device: str, cpu_offload: bool = False):
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise ImportError(
            "Flux2KleinPipeline requires diffusers>=0.37. "
            "pip install 'diffusers>=0.37.0'"
        ) from exc

    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.inference_mode()
def run_one(
    pipe,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    **_unused,
) -> Image.Image:
    resized = resize_for_mode(image, RESIZE_MODE)
    out_w, out_h = resized.size
    device = getattr(pipe, "_execution_device", "cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = dict(
        image=resized,
        prompt=prompt,
        height=out_h,
        width=out_w,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    try:
        out = pipe(**kwargs)
    except TypeError:
        kwargs.pop("height", None)
        kwargs.pop("width", None)
        out = pipe(**kwargs)
    return out.images[0].convert("RGB")
