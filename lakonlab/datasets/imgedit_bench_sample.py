# Copyright (c) 2026 EditFlow contributors

"""Fixed ImgEdit-Bench validation subset for training-time sample dumps.

Randomly samples ``samples_per_category`` examples from each requested
``edit_type`` in ``basic_edit.json``, then keeps that subset fixed for the
whole run (seeded).
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from mmcv.parallel import DataContainer as DC
from mmgen.datasets.builder import DATASETS
from mmgen.utils import get_root_logger
from torch.utils.data import Dataset

from .image_edit import (
    FLUX2_LATENT_CHANNELS,
    _load_rgb,
    _pick_flux2_resolution,
    _pick_kontext_resolution,
    _pick_qwen_condition_resolution,
    _pick_qwen_vae_resolution,
    _resize_center_crop,
    _resize_to,
)


DEFAULT_CATEGORIES = (
    'action',
    'add',
    'adjust',
    'background',
    'compose',
    'extract',
    'remove',
    'replace',
    'style',
)


def _to_tensor(image, bucket=None, resize_mode='qwen'):
    if resize_mode in ('kontext', 'qwen', 'flux2'):
        assert bucket is not None
        image = _resize_to(image, bucket[0], bucket[1])
    else:
        image = _resize_center_crop(image, 1024)
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def _latent_size_for_bucket(resize_mode, bucket, vae_scale_factor, latent_channels):
    if resize_mode in ('kontext', 'qwen', 'flux2'):
        assert bucket is not None
        bw, bh = bucket
        return (latent_channels, bh // vae_scale_factor, bw // vae_scale_factor)
    size = 1024 // vae_scale_factor
    return (latent_channels, size, size)


def build_edit_val_item(
        meta: Dict,
        idx: int,
        seed: int,
        resize_mode: str,
        vae_scale_factor: int,
        latent_channels: int,
        split: str = ''):
    """Shared ImgEdit / GEdit / OSS val dump item."""
    source_arr = _load_rgb(meta['source_path'])
    src_w, src_h = source_arr.shape[1], source_arr.shape[0]

    bucket = None
    condition_bucket = None
    if resize_mode == 'kontext':
        bucket = _pick_kontext_resolution(src_w, src_h)
    elif resize_mode == 'qwen':
        bucket = _pick_qwen_vae_resolution(src_w, src_h)
        condition_bucket = _pick_qwen_condition_resolution(src_w, src_h)
    elif resize_mode == 'flux2':
        bucket = _pick_flux2_resolution(src_w, src_h)

    source_tensor = _to_tensor(source_arr, bucket, resize_mode)
    latent_size = _latent_size_for_bucket(
        resize_mode, bucket, vae_scale_factor, latent_channels)
    noise = torch.randn(
        latent_size,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(int(seed) + int(idx)))

    data = dict(
        ids=DC(idx, cpu_only=True),
        name=DC(meta['prompt'], cpu_only=True),
        prompt_kwargs=dict(prompt=DC(meta['prompt'], cpu_only=True)),
        source_images=source_tensor,
        category=DC(meta['category'], cpu_only=True),
        example_name=DC(meta['example_name'], cpu_only=True),
        noise=noise,
    )
    if split:
        data['split'] = DC(str(split), cpu_only=True)
    if condition_bucket is not None:
        data['condition_source_images'] = _to_tensor(
            source_arr, condition_bucket, resize_mode)
    target_path = meta.get('target_path')
    if target_path and os.path.isfile(target_path):
        target_arr = _load_rgb(target_path)
        data['edited_images'] = _to_tensor(target_arr, bucket, resize_mode)
    return data


@DATASETS.register_module()
class ImgEditBenchSample(Dataset):
    """ImgEdit-Bench basic suite subset for validation sampling.

    Each item exposes:
      - ``source_images`` / optional ``condition_source_images``
      - ``prompt_kwargs`` / ``name``
      - ``category`` / ``example_name`` (e.g. add / example1)
      - deterministic ``noise`` for reproducible val dumps
    """

    def __init__(
            self,
            annotations_path: str = (
                ''
                'evaluation/imgedit_bench/annotations/basic_edit.json'),
            bench_root: str = (
                '/path/to/data/imgedit/benchmark/Benchmark'),
            categories: Optional[Sequence[str]] = None,
            samples_per_category: int = 2,
            seed: int = 42,
            resize_mode: str = 'qwen',
            vae_scale_factor: int = 8,
            latent_channels: int = 16,
            split: str = '',
            **kwargs):
        del kwargs  # allow unused mmgen dataset kwargs
        assert resize_mode in ('center_crop', 'kontext', 'qwen', 'flux2'), (
            f'Unsupported resize_mode={resize_mode}')
        self.annotations_path = annotations_path
        self.bench_root = bench_root
        self.categories = tuple(
            c.lower() for c in (categories or DEFAULT_CATEGORIES))
        self.samples_per_category = int(samples_per_category)
        self.seed = int(seed)
        self.split = str(split or '')
        self.resize_mode = resize_mode
        self.vae_scale_factor = vae_scale_factor
        if resize_mode == 'flux2' and latent_channels == 16:
            latent_channels = FLUX2_LATENT_CHANNELS
        self.latent_channels = latent_channels

        with open(annotations_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        by_cat: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        for key, item in raw.items():
            cat = str(item.get('edit_type', '')).lower()
            if cat in self.categories:
                by_cat[cat].append((key, item))

        rng = random.Random(self.seed)
        self.samples: List[Dict] = []
        missing_cats = []
        for cat in self.categories:
            pool = by_cat.get(cat, [])
            if len(pool) < self.samples_per_category:
                missing_cats.append(
                    f'{cat}(have={len(pool)}, need={self.samples_per_category})')
                chosen = pool
            else:
                chosen = rng.sample(pool, self.samples_per_category)
            for i, (key, item) in enumerate(chosen, start=1):
                src_rel = item['id']
                src_path = os.path.join(self.bench_root, 'singleturn', src_rel)
                self.samples.append(dict(
                    key=key,
                    category=cat,
                    example_name=f'example{i}',
                    prompt=item['prompt'],
                    source_path=src_path,
                    source_rel=src_rel,
                ))

        logger = get_root_logger()
        logger.info(
            f'ImgEditBenchSample: {len(self.samples)} fixed samples '
            f'({self.samples_per_category}/cat × {len(self.categories)} cats, '
            f'seed={self.seed}, resize_mode={self.resize_mode}) '
            f'from {annotations_path}')
        if missing_cats:
            logger.warning(
                'ImgEditBenchSample: insufficient examples for: '
                + ', '.join(missing_cats))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return build_edit_val_item(
            self.samples[idx],
            idx=idx,
            seed=self.seed,
            resize_mode=self.resize_mode,
            vae_scale_factor=self.vae_scale_factor,
            latent_channels=self.latent_channels,
            split=self.split,
        )
