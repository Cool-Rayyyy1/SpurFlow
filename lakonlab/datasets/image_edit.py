# Copyright (c) 2026 EditFlow contributors

import json
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmgen.datasets.builder import DATASETS
from mmgen.utils import get_root_logger
from PIL import Image
from torch.utils.data import Dataset


def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img.convert('RGB'), dtype=np.uint8)
    return arr


def _resize_center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Resize the shorter side to `size`, then center-crop a square."""
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


# FLUX.1 Kontext preferred resolutions as (width, height); all multiples of 16.
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
]


def _pick_kontext_resolution(width: int, height: int) -> Tuple[int, int]:
    """Pick the preferred (width, height) bucket with the closest aspect ratio."""
    aspect = width / max(height, 1)
    _, bw, bh = min(
        (abs(aspect - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
    return bw, bh


def _resize_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Plain bicubic resize to (width, height) (FLUX Kontext style, no crop)."""
    h, w = image.shape[:2]
    if w == width and h == height:
        return image
    pil = Image.fromarray(image).resize((width, height), Image.Resampling.BICUBIC)
    return np.array(pil, dtype=np.uint8)


@DATASETS.register_module()
class ImageEdit(Dataset):
    """Image editing dataset for FLUX Kontext distillation.

    Each sample provides:
      - source image (reference / input)
      - edited image (target, used for latent shape; optional in test_mode)
      - editing instruction (prompt)

    Missing source/target files are skipped automatically (next row is used).
    """

    def __init__(
            self,
            data_root: str,
            jsonl_path: str = 'jsonl/sft_with_local_source_image_path.jsonl',
            edited_images_dir: str = 'edited_images',
            source_column: str = 'local_input_image',
            target_column: str = 'output_image',
            prompt_column: str = 'text',
            image_size: int = 1024,
            vae_scale_factor: int = 8,
            latent_size: Optional[Tuple[int]] = (16, 128, 128),
            repeat: int = 1,
            start_ind: Optional[int] = None,
            end_ind: Optional[int] = None,
            test_mode: bool = False,
            require_edited: bool = False,
            resize_mode: str = 'center_crop'):
        super().__init__()
        assert resize_mode in ('center_crop', 'kontext'), (
            f'Unsupported resize_mode={resize_mode}; expected center_crop or kontext.')
        self.data_root = os.path.abspath(data_root)
        self.edited_root = os.path.join(self.data_root, edited_images_dir)
        self.image_size = image_size
        self.vae_scale_factor = vae_scale_factor
        self.latent_size = latent_size
        self.repeat = repeat
        self.test_mode = test_mode
        self.require_edited = require_edited
        self.resize_mode = resize_mode
        self._skip_warned = False

        jsonl_full = jsonl_path if os.path.isabs(jsonl_path) else os.path.join(self.data_root, jsonl_path)
        if not os.path.isfile(jsonl_full):
            raise FileNotFoundError(f'Missing jsonl: {jsonl_full}')

        self.records = []
        with open(jsonl_full, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        self.source_keys = [
            source_column, 'local_input_image', 'reference', 'reference_image',
            'source_image', 'source', 'input_image', 'image']
        self.target_keys = [
            target_column, 'output_image', 'edited_image', 'target', 'target_image']
        self.prompt_keys = [
            prompt_column, 'text', 'prompt', 'instruction', 'edit_prompt', 'caption']

        dataset_len = len(self.records)
        if start_ind is not None:
            start_ind = max(min(start_ind, dataset_len - 1), -dataset_len) % dataset_len
        else:
            start_ind = 0
        if end_ind is not None:
            end_ind = max(min(end_ind - 1, dataset_len - 1), -dataset_len) % dataset_len + 1
        else:
            end_ind = dataset_len
        assert start_ind < end_ind, 'Invalid start_ind and end_ind.'
        self.start_ind = start_ind
        self.end_ind = end_ind

        get_root_logger().info(
            f'ImageEdit: loaded {dataset_len} rows, using [{self.start_ind}, {self.end_ind}), '
            f'image_size={self.image_size}, require_edited={self.require_edited}, skip_missing=True')

    def _pick(self, row: dict, keys):
        for key in keys:
            if key in row and row[key] not in (None, ''):
                return row[key]
        return None

    def _resolve_source(self, raw: str) -> Optional[str]:
        path = os.path.expanduser(raw)
        if os.path.isfile(path):
            return path
        for cand in [
                os.path.join(self.data_root, raw),
                os.path.join(self.data_root, 'source_images', raw),
        ]:
            if os.path.isfile(cand):
                return cand
        return None

    def _resolve_target(self, raw: str) -> Optional[str]:
        path = os.path.expanduser(raw)
        if os.path.isfile(path):
            return path
        for cand in [
                os.path.join(self.edited_root, raw),
                os.path.join(self.data_root, raw),
        ]:
            if os.path.isfile(cand):
                return cand
        return None

    def _warn_skip_once(self, mapped_idx: int, reason: str) -> None:
        if not self._skip_warned:
            get_root_logger().warning(
                f'ImageEdit: skipping rows with missing/broken files (e.g. idx={mapped_idx}: {reason}). '
                f'Further skips are silent.')
            self._skip_warned = True

    def _to_tensor(self, image: np.ndarray, bucket: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if self.resize_mode == 'kontext':
            assert bucket is not None, 'kontext resize_mode requires a (w, h) bucket.'
            image = _resize_to(image, bucket[0], bucket[1])
        else:
            image = _resize_center_crop(image, self.image_size)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor

    def _latent_size_for_image(self, bucket: Optional[Tuple[int, int]] = None):
        if self.resize_mode == 'kontext':
            assert bucket is not None, 'kontext resize_mode requires a (w, h) bucket.'
            bw, bh = bucket
            return (self.latent_size[0], bh // self.vae_scale_factor, bw // self.vae_scale_factor)
        h = w = self.image_size // self.vae_scale_factor
        return (self.latent_size[0], h, w)

    def _map_idx(self, idx):
        return self.start_ind + (idx // self.repeat) % (self.end_ind - self.start_ind)

    def __len__(self):
        return self.repeat * (self.end_ind - self.start_ind)

    def __getitem__(self, idx):
        num_records = self.end_ind - self.start_ind
        slot = (idx // self.repeat) % num_records
        for offset in range(num_records):
            mapped_idx = self.start_ind + (slot + offset) % num_records
            sample = self._build_sample(mapped_idx, idx)
            if sample is not None:
                return sample
        raise RuntimeError(
            f'ImageEdit: no valid sample in [{self.start_ind}, {self.end_ind})')

    def _build_sample(self, mapped_idx: int, idx: int):
        row = self.records[mapped_idx]

        source_raw = self._pick(row, self.source_keys)
        target_raw = self._pick(row, self.target_keys)
        prompt = self._pick(row, self.prompt_keys)
        if source_raw is None or prompt is None:
            self._warn_skip_once(mapped_idx, 'missing source or prompt')
            return None

        source_path = self._resolve_source(source_raw)
        if source_path is None:
            self._warn_skip_once(mapped_idx, f'source not found: {source_raw}')
            return None

        try:
            source_arr = _load_rgb(source_path)
        except OSError:
            self._warn_skip_once(mapped_idx, f'broken source: {source_path}')
            return None

        # In kontext mode the bucket is chosen from the SOURCE aspect ratio and
        # applied to both source and edited so x_ref and x_0 share the latent grid.
        bucket = None
        if self.resize_mode == 'kontext':
            bucket = _pick_kontext_resolution(source_arr.shape[1], source_arr.shape[0])
        source_tensor = self._to_tensor(source_arr, bucket)

        latent_size = self._latent_size_for_image(bucket)

        data = dict(
            ids=DC(idx, cpu_only=True),
            name=DC(prompt, cpu_only=True),
            prompt_kwargs=dict(prompt=DC(prompt, cpu_only=True)),
            source_images=source_tensor,
        )

        if self.test_mode:
            data['noise'] = torch.randn(
                latent_size, dtype=torch.float32, generator=torch.Generator().manual_seed(idx))
            if target_raw is not None:
                target_path = self._resolve_target(target_raw)
                if target_path is not None:
                    try:
                        data['edited_images'] = self._to_tensor(_load_rgb(target_path), bucket)
                    except OSError:
                        pass
        elif target_raw is not None:
            target_path = self._resolve_target(target_raw)
            if target_path is None:
                self._warn_skip_once(mapped_idx, f'edited not found: {target_raw}')
                return None
            try:
                edited_tensor = self._to_tensor(_load_rgb(target_path), bucket)
            except OSError:
                self._warn_skip_once(mapped_idx, f'broken edited: {target_path}')
                return None
            data['edited_images'] = edited_tensor
            data['latents'] = torch.empty(latent_size, dtype=torch.float32)
        elif self.require_edited:
            self._warn_skip_once(mapped_idx, 'missing edited image path')
            return None
        else:
            data['latents'] = torch.empty(latent_size, dtype=torch.float32)

        return data
