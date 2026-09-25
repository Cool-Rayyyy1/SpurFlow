# Copyright (c) 2026 EditFlow contributors
"""Fixed GEdit / OSS validation subsets for training-time sample dumps."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from mmcv.utils import build_from_cfg
from mmgen.datasets.builder import DATASETS
from mmgen.utils import get_root_logger
from torch.utils.data import Dataset

from .image_edit import FLUX2_LATENT_CHANNELS
from .imgedit_bench_sample import build_edit_val_item


# Official GEdit-Bench English tasks (used by the Step1X GEdit-Bench dump).
DEFAULT_GEDIT_CATEGORIES = (
    'background_change',
    'motion_change',
    'style_change',
    'subject-add',
    'text_change',
)

# GEdit-v2 edit types from /mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json
DEFAULT_GEDIT_V2_META = '/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json'
DEFAULT_GEDIT_V2_ROOT = '/mnt/afs_caiqi/data/benchmark/GEdit_v2'
DEFAULT_GEDIT_V2_CATEGORIES = (
    'background_change',
    'camera_motion',
    'character_reference',
    'chart_editing',
    'color_alteration',
    'enhancement',
    'hybrid',
    'in_image_text_translation',
    'line2image',
    'material_modification',
    'motion_change',
    'object_reference',
    'openset',
    'portrait_beautification',
    'relation_change',
    'size_adjustment',
    'style_reference',
    'style_transfer',
    'subject_addition',
    'subject_removal',
    'subject_replace',
    'text_editing',
    'tone_transfer',
)

DEFAULT_OSS_TASKS = (
    'camera_motion',
    'relation_change',
    'size_adjustment',
    'enhancement',
    'text_editing',
)


def _resolve_resize(resize_mode: str, latent_channels: int):
    assert resize_mode in ('center_crop', 'kontext', 'qwen', 'flux2'), (
        f'Unsupported resize_mode={resize_mode}')
    if resize_mode == 'flux2' and latent_channels == 16:
        latent_channels = FLUX2_LATENT_CHANNELS
    return resize_mode, latent_channels


@DATASETS.register_module()
class ConcatEditValSample(Dataset):
    """Concatenate ImgEdit / GEdit / OSS val dumps into one dataloader."""

    def __init__(self, datasets: Sequence[dict], **kwargs):
        del kwargs
        self.datasets = [
            build_from_cfg(cfg, DATASETS) if isinstance(cfg, dict) else cfg
            for cfg in datasets]
        self.cumlens: List[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumlens.append(total)
        get_root_logger().info(
            'ConcatEditValSample: '
            + ' + '.join(f'{type(d).__name__}({len(d)})' for d in self.datasets)
            + f' = {len(self)}')

    def __len__(self):
        return self.cumlens[-1] if self.cumlens else 0

    def __getitem__(self, idx):
        for dataset, end in zip(self.datasets, self.cumlens):
            start = end - len(dataset)
            if idx < end:
                return dataset[idx - start]
        raise IndexError(idx)


@DATASETS.register_module()
class GEditBenchSample(Dataset):
    """Fixed English GEdit-Bench subset. Default: 1 example × 5 categories."""

    def __init__(
            self,
            annotations_path: str = (
                '/mnt/afs_gaochengmin/data/benchmark/GEdit-Bench/GEdit-Bench.json'),
            bench_root: str = '/mnt/afs_gaochengmin/data/benchmark/GEdit-Bench',
            categories: Optional[Sequence[str]] = None,
            samples_per_category: int = 1,
            language: Optional[str] = 'en',
            seed: int = 42,
            resize_mode: str = 'qwen',
            vae_scale_factor: int = 8,
            latent_channels: int = 16,
            split: str = 'gedit',
            **kwargs):
        del kwargs
        self.resize_mode, self.latent_channels = _resolve_resize(
            resize_mode, latent_channels)
        self.vae_scale_factor = vae_scale_factor
        self.seed = int(seed)
        self.split = str(split or 'gedit')
        self.categories = tuple(categories or DEFAULT_GEDIT_CATEGORIES)
        self.samples_per_category = int(samples_per_category)

        with open(annotations_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f'{annotations_path} must be a JSON list')

        by_cat: Dict[str, List[dict]] = defaultdict(list)
        for item in raw:
            if language and str(item.get('instruction_language', '')).lower() != language:
                continue
            cat = str(item.get('task') or item.get('edit_type') or '').strip()
            if cat not in self.categories:
                continue
            src_rel = str(item.get('src_image_path') or '').strip()
            if not src_rel:
                continue
            src_path = src_rel if os.path.isabs(src_rel) else os.path.join(
                bench_root, src_rel)
            if not os.path.isfile(src_path):
                continue
            by_cat[cat].append(dict(
                prompt=str(item.get('instruction') or item.get('prompt') or ''),
                source_path=src_path,
                category=cat,
            ))

        rng = random.Random(self.seed)
        self.samples: List[Dict] = []
        missing = []
        for cat in self.categories:
            pool = by_cat.get(cat, [])
            if len(pool) < self.samples_per_category:
                missing.append(f'{cat}(have={len(pool)}, need={self.samples_per_category})')
                chosen = pool
            else:
                chosen = rng.sample(pool, self.samples_per_category)
            for i, item in enumerate(chosen, start=1):
                self.samples.append(dict(
                    category=item['category'],
                    example_name=f'example{i}',
                    prompt=item['prompt'],
                    source_path=item['source_path'],
                ))

        logger = get_root_logger()
        logger.info(
            f'{type(self).__name__}: {len(self.samples)} fixed samples '
            f'({self.samples_per_category}/cat × {len(self.categories)} cats, '
            f'lang={language}, seed={self.seed}, resize_mode={self.resize_mode}) '
            f'from {annotations_path}')
        if missing:
            logger.warning(
                f'{type(self).__name__}: insufficient examples for: '
                + ', '.join(missing))

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


@DATASETS.register_module()
class GEditV2Sample(GEditBenchSample):
    """Fixed GEdit-v2 subset used by evaluation/run_gmkontext_*_gedit_infer.sh.

    Default: 2 examples × 23 edit_type from gedit_v2_meta.json.
    """

    def __init__(
            self,
            annotations_path: str = DEFAULT_GEDIT_V2_META,
            bench_root: str = DEFAULT_GEDIT_V2_ROOT,
            categories: Optional[Sequence[str]] = None,
            samples_per_category: int = 2,
            language: Optional[str] = None,
            split: str = 'gedit_v2',
            **kwargs):
        super().__init__(
            annotations_path=annotations_path,
            bench_root=bench_root,
            categories=categories or DEFAULT_GEDIT_V2_CATEGORIES,
            samples_per_category=samples_per_category,
            language=language,
            split=split,
            **kwargs)


@DATASETS.register_module()
class OssEditSample(Dataset):
    """Fixed OSS-edit subset. Default: 1 example from 5 tasks including camera_motion."""

    def __init__(
            self,
            data_root: str = '/mnt/afs_gaochengmin/data/oss_edit',
            jsonl_path: str = 'metadata.jsonl',
            tasks: Optional[Sequence[str]] = None,
            samples_per_task: int = 1,
            seed: int = 42,
            resize_mode: str = 'qwen',
            vae_scale_factor: int = 8,
            latent_channels: int = 16,
            split: str = 'oss',
            **kwargs):
        del kwargs
        self.resize_mode, self.latent_channels = _resolve_resize(
            resize_mode, latent_channels)
        self.vae_scale_factor = vae_scale_factor
        self.seed = int(seed)
        self.split = str(split or 'oss')
        self.tasks = tuple(tasks or DEFAULT_OSS_TASKS)
        self.samples_per_task = int(samples_per_task)

        jsonl_full = jsonl_path if os.path.isabs(jsonl_path) else os.path.join(
            data_root, jsonl_path)
        by_task: Dict[str, List[dict]] = defaultdict(list)
        with open(jsonl_full, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task = str(rec.get('task') or rec.get('edit_type') or '').strip()
                if task not in self.tasks:
                    continue
                src = str(rec.get('input_path') or '').strip()
                if src and not os.path.isabs(src):
                    src = os.path.join(data_root, src)
                if not src:
                    continue
                tgt = str(rec.get('output_path') or '').strip()
                if tgt and not os.path.isabs(tgt):
                    tgt = os.path.join(data_root, tgt)
                prompt = str(
                    rec.get('instruction')
                    or rec.get('Edit_Instruction')
                    or rec.get('prompt')
                    or '')
                by_task[task].append(dict(
                    prompt=prompt,
                    source_path=src,
                    target_path=tgt,
                    category=task,
                ))

        rng = random.Random(self.seed)
        self.samples: List[Dict] = []
        missing = []
        for task in self.tasks:
            pool = list(by_task.get(task, []))
            rng.shuffle(pool)
            chosen = []
            for item in pool:
                if not os.path.isfile(item['source_path']):
                    continue
                tgt = item['target_path']
                chosen.append(dict(
                    category=item['category'],
                    example_name=f'example{len(chosen)+1}',
                    prompt=item['prompt'],
                    source_path=item['source_path'],
                    target_path=tgt if tgt and os.path.isfile(tgt) else '',
                ))
                if len(chosen) >= self.samples_per_task:
                    break
            if len(chosen) < self.samples_per_task:
                missing.append(f'{task}(have={len(chosen)}, need={self.samples_per_task})')
            self.samples.extend(chosen)

        logger = get_root_logger()
        logger.info(
            f'OssEditSample: {len(self.samples)} fixed samples '
            f'({self.samples_per_task}/task × {len(self.tasks)} tasks, '
            f'seed={self.seed}, resize_mode={self.resize_mode}) from {jsonl_full}')
        if missing:
            logger.warning(
                'OssEditSample: insufficient examples for: ' + ', '.join(missing))

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
