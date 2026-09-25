"""Shared helpers for official FLUX.1-Kontext-dev baseline inference.

Resize matches EditFlow ``STUDENT_RESIZE_MODE=kontext`` / ``_pick_kontext_resolution``:
pick the closest FLUX.1-Kontext preferred-resolution bucket by aspect ratio,
then bicubic resize with no crop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image

# Same buckets as lakonlab/datasets/image_edit.py and FluxKontextPipeline.
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

DEFAULT_MODEL = os.environ.get(
    "KONTEXT_MODEL_PATH",
    "/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev",
)
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 2.5
HF_HOME = "/mnt/afs_gaochengmin/.cache/huggingface"


def configure_hf_env() -> None:
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def pick_kontext_resolution(width: int, height: int) -> Tuple[int, int]:
    """Pick the preferred (width, height) bucket with the closest aspect ratio."""
    aspect = width / max(height, 1)
    _, bw, bh = min(
        (abs(aspect - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
    )
    return bw, bh


def resize_kontext(image: Image.Image) -> Image.Image:
    """Bicubic resize to the nearest Kontext bucket. No crop."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    out_w, out_h = pick_kontext_resolution(width, height)
    if (width, height) == (out_w, out_h):
        return rgb
    return rgb.resize((out_w, out_h), Image.Resampling.BICUBIC)


def build_pipeline(model_path: str, device: str, cpu_offload: bool = False):
    from diffusers import FluxKontextPipeline

    pipe = FluxKontextPipeline.from_pretrained(
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
) -> Image.Image:
    resized = resize_kontext(image)
    out_w, out_h = resized.size
    device = getattr(pipe, "_execution_device", "cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    # Pass height/width of the Kontext bucket and max_area=w*h so the
    # pipeline does not squash generation back to a 1024^2 square.
    out = pipe(
        image=resized,
        prompt=prompt,
        height=out_h,
        width=out_w,
        max_area=out_w * out_h,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        _auto_resize=False,
    )
    return out.images[0].convert("RGB")


def resolve_gpu_ids(num_gpus: int) -> List[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [int(x) for x in visible.split(",") if x.strip()]
    else:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        ids = list(range(n))
    if not ids:
        return [0]
    if num_gpus > len(ids):
        raise ValueError(
            f"num_gpus={num_gpus} but only {len(ids)} GPU(s) visible ({visible!r})"
        )
    return ids[:num_gpus]


def split_tasks_round_robin(tasks: list, num_workers: int) -> list:
    buckets = [[] for _ in range(num_workers)]
    for idx, task in enumerate(tasks):
        buckets[idx % num_workers].append(task)
    return [bucket for bucket in buckets if bucket]


def merge_manifest_parts(output_dir: Path) -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    for part in sorted(output_dir.glob("manifest.gpu*.json")):
        manifest.update(json.loads(part.read_text(encoding="utf-8")))
        part.unlink()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
