# Copyright (c) 2025 Hansheng Chen

import bisect
import os
from typing import Dict, List, Optional

import numpy as np
import orjson
import pyarrow.parquet as pq
import torch.distributed as dist
from mmcv.runner import get_dist_info


def build_jsonl_byte_offsets(jsonl_path: str) -> np.ndarray:
    offsets = []
    with open(jsonl_path, 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    return np.asarray(offsets, dtype=np.int64)


class LazyJsonlPromptStore:
    """TDM-style lazy jsonl reader with byte-offset index."""

    def __init__(self,
                 jsonl_path: str,
                 prompt_column: str = 'prompt',
                 offsets_path: Optional[str] = None):
        self.jsonl_path = str(jsonl_path)
        self.prompt_column = prompt_column
        if offsets_path is None:
            offsets_path = self.jsonl_path + '.offsets.npy'
        self.offsets_path = offsets_path

        rank, world_size = get_dist_info()
        if rank == 0 and not os.path.isfile(self.offsets_path):
            os.makedirs(os.path.dirname(self.offsets_path) or '.', exist_ok=True)
            np.save(self.offsets_path, build_jsonl_byte_offsets(self.jsonl_path))
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        self.offsets = np.load(self.offsets_path, mmap_mode='r')
        self._fh = None

    def __len__(self):
        return len(self.offsets)

    def _read_row(self, idx: int) -> Dict:
        if self._fh is None:
            self._fh = open(self.jsonl_path, 'rb')
        self._fh.seek(int(self.offsets[idx]))
        row = orjson.loads(self._fh.readline())
        prompt = row.get(self.prompt_column, row.get('ori_prompt', row.get('text', '')))
        result = dict(prompt=prompt)
        if 'height' in row:
            result['height'] = int(row['height'])
        if 'width' in row:
            result['width'] = int(row['width'])
        if 'frames' in row:
            result['frames'] = int(row['frames'])
        return result

    def __getitem__(self, idx: int) -> Dict:
        return self._read_row(idx)


class LazyParquetPromptStore:
    """Lazy prompt reader for local parquet shards (no HuggingFace datasets)."""

    def __init__(self, parquet_files: List[str]):
        self.parquet_files = sorted(parquet_files)
        self.row_group_starts: List[int] = []
        self.row_group_files: List[str] = []
        self.row_group_ids: List[int] = []
        self.row_group_lengths: List[int] = []
        total = 0

        for parquet_file in self.parquet_files:
            pf = pq.ParquetFile(parquet_file)
            for row_group_id in range(pf.num_row_groups):
                num_rows = pf.metadata.row_group(row_group_id).num_rows
                self.row_group_starts.append(total)
                self.row_group_files.append(parquet_file)
                self.row_group_ids.append(row_group_id)
                self.row_group_lengths.append(num_rows)
                total += num_rows
        self._len = total
        self._cache_file = None
        self._cache_rg = None
        self._cache_table = None

    def __len__(self):
        return self._len

    def _locate(self, idx: int):
        rg_idx = bisect.bisect_right(self.row_group_starts, idx) - 1
        local_idx = idx - self.row_group_starts[rg_idx]
        return rg_idx, local_idx

    def __getitem__(self, idx: int) -> Dict:
        rg_idx, local_idx = self._locate(idx)
        parquet_file = self.row_group_files[rg_idx]
        row_group_id = self.row_group_ids[rg_idx]

        if self._cache_file != parquet_file or self._cache_rg != row_group_id:
            pf = pq.ParquetFile(parquet_file)
            columns = ['prompt', 'height', 'width']
            if 'frames' in pf.schema_arrow.names:
                columns = ['prompt', 'frames', 'height', 'width']
            self._cache_table = pf.read_row_group(row_group_id, columns=columns)
            self._cache_file = parquet_file
            self._cache_rg = row_group_id

        row = self._cache_table.slice(local_idx, 1)
        result = dict(
            prompt=row['prompt'][0].as_py(),
            height=int(row['height'][0].as_py()),
            width=int(row['width'][0].as_py()),
        )
        if 'frames' in row.column_names:
            result['frames'] = int(row['frames'][0].as_py())
        return result
