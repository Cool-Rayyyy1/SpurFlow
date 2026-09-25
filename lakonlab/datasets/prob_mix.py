# Copyright (c) 2026 SpurFlow contributors
"""Probability mixture of image-edit datasets."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch.distributed as dist
from mmgen.datasets.builder import DATASETS, build_dataset
from mmgen.utils import get_root_logger
from torch.utils.data import Dataset


def _rank_world() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


@DATASETS.register_module()
class ProbMixDataset(Dataset):
    """Mix child datasets by probability.

    ``uid = idx * world_size + rank`` so ranks do not draw independently.
    A 10-slot cycle maps uid -> dataset (7/3 for 70/30) and a running
    counter maps to the inner index, wrapping only after one pass.
    """

    def __init__(
            self,
            datasets: Sequence[dict],
            probs: Sequence[float],
            length: Optional[int] = None,
            cycle: int = 10,
            **kwargs):
        super().__init__()
        if len(datasets) != len(probs):
            raise ValueError(
                f'datasets ({len(datasets)}) and probs ({len(probs)}) length mismatch')
        if len(datasets) == 0:
            raise ValueError('ProbMixDataset needs at least one child dataset')
        total = float(sum(probs))
        if total <= 0:
            raise ValueError(f'probs must sum to > 0, got {probs}')
        self.probs = np.array([float(p) / total for p in probs], dtype=np.float64)
        self.datasets = [build_dataset(cfg) for cfg in datasets]
        child_lens = [len(ds) for ds in self.datasets]
        if any(n <= 0 for n in child_lens):
            raise ValueError(f'empty child dataset: lengths={child_lens}')
        self._length = int(length) if length is not None else int(max(child_lens))
        self.cycle = int(cycle)
        if self.cycle < 1:
            raise ValueError(f'cycle must be >= 1, got {cycle}')
        # e.g. 70/30 -> thresh [7, 10], slots_per_ds [7, 3]
        thresh = np.round(np.cumsum(self.probs) * self.cycle).astype(np.int64)
        thresh[-1] = self.cycle
        self._thresh = thresh
        self._slots = np.diff(np.concatenate([[0], thresh])).astype(np.int64)
        get_root_logger().info(
            'ProbMixDataset: '
            + ', '.join(
                f'{ds.__class__.__name__}(n={n}, p={p:.2f}, slots={s})'
                for ds, n, p, s in zip(
                    self.datasets, child_lens, self.probs, self._slots)
            )
            + f', length={self._length}')

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        rank, world = _rank_world()
        uid = int(idx) * world + rank
        slot = uid % self.cycle
        ds_idx = int(np.searchsorted(self._thresh, slot, side='right'))
        ds_idx = min(ds_idx, len(self.datasets) - 1)
        prev = 0 if ds_idx == 0 else int(self._thresh[ds_idx - 1])
        n_this = int(self._slots[ds_idx])
        k = (uid // self.cycle) * n_this + (slot - prev)
        ds = self.datasets[ds_idx]
        inner = k % len(ds)
        return ds[inner]
