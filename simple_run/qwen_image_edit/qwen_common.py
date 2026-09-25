"""Official Qwen-Image-Edit-2511 pipeline helpers.

HF: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
Local: /mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511

Official settings (model card):
  num_inference_steps=40
  true_cfg_scale=4.0
  guidance_scale=1.0
  negative_prompt=" "

Resize: ImageEdit resize_mode='qwen' (VAE-area ~1MP, bicubic, no crop).
"""

from __future__ import annotations

import os

import torch
from PIL import Image

from official_common import resize_for_mode

DEFAULT_MODEL = os.environ.get(
    "QWEN_MODEL_PATH",
    "/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511",
)
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = 1.0
DEFAULT_TRUE_CFG = float(os.environ.get("QWEN_TRUE_CFG_SCALE", "4.0"))
DEFAULT_NEGATIVE = os.environ.get("QWEN_NEGATIVE_PROMPT", " ")
RESIZE_MODE = "qwen"
HF_ID = "Qwen/Qwen-Image-Edit-2511"
RUN_TAG_GEDIT = "qwen_image_edit_2511_gedit"
RUN_TAG_IMGEDIT = "qwen_image_edit_2511_imgedit"


def build_pipeline(model_path: str, device: str, cpu_offload: bool = False):
    import diffusers

    pipe_cls = None
    for name in (
        "QwenImageEditPlusPipeline",
        "QwenImageEditPlusPipeline",
        "QwenImageEditPipeline",
    ):
        pipe_cls = getattr(diffusers, name, None)
        if pipe_cls is not None:
            break
    if pipe_cls is None:
        raise ImportError(
            "Need QwenImageEditPlusPipeline from a recent diffusers. "
            "pip install git+https://github.com/huggingface/diffusers"
        )

    pipe = pipe_cls.from_pretrained(
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
    true_cfg_scale: float = DEFAULT_TRUE_CFG,
    negative_prompt: str = DEFAULT_NEGATIVE,
) -> Image.Image:
    resized = resize_for_mode(image, RESIZE_MODE)
    out_w, out_h = resized.size
    kwargs = dict(
        image=[resized],
        prompt=prompt,
        generator=torch.manual_seed(seed),
        true_cfg_scale=true_cfg_scale,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        height=out_h,
        width=out_w,
    )
    try:
        out = pipe(**kwargs)
    except TypeError:
        kwargs.pop("height", None)
        kwargs.pop("width", None)
        out = pipe(**kwargs)
    return out.images[0].convert("RGB")
