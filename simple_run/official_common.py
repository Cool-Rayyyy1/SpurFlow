"""Shared helpers for official Qwen / FLUX.2-klein-base simple_run scripts.

Resize modes match training ImageEdit:
  qwen  — VAE-area (~1MP) bicubic, no crop  (Qwen-Image-Edit-2511)
  flux2 — area-cap ~1MP then snap to 16px   (FLUX.2-klein-base-9B)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, TypeVar

import numpy as np
import torch
from PIL import Image

T = TypeVar("T")

HF_HOME = "/mnt/afs_gaochengmin/.cache/huggingface"
EDITFLOW_ROOT = Path(__file__).resolve().parents[1]


def configure_hf_env() -> None:
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    root = str(EDITFLOW_ROOT)
    if root not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = f"{root}:{os.environ.get('PYTHONPATH', '')}"


def slice_split(items: Sequence[T], split_id: int, split_num: int) -> List[T]:
    """``metadata[split_id::split_num]``."""
    if split_num < 1:
        raise ValueError(f"split_num must be >= 1, got {split_num}")
    if split_id < 0 or split_id >= split_num:
        raise ValueError(f"split_id must be in [0, {split_num}), got {split_id}")
    return list(items[split_id::split_num])


def resize_for_mode(image: Image.Image, resize_mode: str) -> Image.Image:
    from lakonlab.datasets.image_edit import (
        _pick_flux2_resolution,
        _pick_qwen_vae_resolution,
        _resize_to,
    )

    mode = resize_mode.strip().lower()
    rgb = image.convert("RGB")
    width, height = rgb.size
    if mode == "qwen":
        out_w, out_h = _pick_qwen_vae_resolution(width, height)
    elif mode == "flux2":
        out_w, out_h = _pick_flux2_resolution(width, height)
    else:
        raise ValueError(f"Unsupported resize_mode={resize_mode!r}; expected 'qwen' or 'flux2'")
    arr = np.array(rgb, dtype=np.uint8)
    arr = _resize_to(arr, out_w, out_h)
    return Image.fromarray(arr)


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


def split_tasks_round_robin(tasks: Sequence[T], num_workers: int) -> List[List[T]]:
    buckets: List[List[T]] = [[] for _ in range(num_workers)]
    for idx, task in enumerate(tasks):
        buckets[idx % num_workers].append(task)
    return [bucket for bucket in buckets if bucket]


def merge_manifest_parts(
    output_dir: Path,
    *,
    glob_pat: str,
    dest_name: str,
) -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    for part in sorted(output_dir.glob(glob_pat)):
        manifest.update(json.loads(part.read_text(encoding="utf-8")))
        part.unlink()
    dest = output_dir / dest_name
    dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def manifest_name(split_id: int, split_num: int) -> str:
    if split_num <= 1:
        return "manifest.json"
    return f"manifest.split{split_id:02d}_of_{split_num:02d}.json"
