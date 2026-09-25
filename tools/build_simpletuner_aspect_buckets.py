#!/usr/bin/env python3
"""Build SimpleTuner aspect_ratio_bucket_*.json without launching train.py.

Writes the same files SimpleTuner would place next to pair dirs:
  aspect_ratio_bucket_indices_<id>.json
  aspect_ratio_bucket_metadata_<id>.json

Optionally also writes output_dir/all_image_files_<id>.json so
SKIP_FILE_DISCOVERY=aspect,metadata does not listdir the pair folder.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _png_size(path: str) -> Optional[tuple[int, int]]:
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return None
            length = struct.unpack(">I", f.read(4))[0]
            if f.read(4) != b"IHDR" or length < 8:
                return None
            w, h = struct.unpack(">II", f.read(8))
            return int(w), int(h)
    except OSError:
        return None


def _pil_size(path: str) -> Optional[tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return tuple(im.size)
    except Exception:
        return None


def read_image_size(path: str) -> Optional[tuple[int, int]]:
    if path.lower().endswith(".png"):
        size = _png_size(path)
        if size is not None:
            return size
    return _pil_size(path)


def iter_images(folder: Path) -> list[str]:
    files: list[str] = []
    with os.scandir(folder) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=True):
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in IMAGE_EXTS:
                files.append(entry.path)
    files.sort()
    return files


def backend_config(instance_data_dir: str, dataset_type: str, extra: dict) -> dict:
    cfg = {
        "start_epoch": 1,
        "start_step": 0,
        "end_epoch": None,
        "end_step": None,
        "vae_cache_clear_each_epoch": False,
        "vae_cache_ondemand": True,
        "vae_cache_disable": False,
        "repeats": 0,
        "crop": True,
        "crop_aspect": "square",
        "crop_aspect_buckets": None,
        "crop_style": extra.get("crop_style", "random"),
        "disable_validation": False,
        "resolution": 1.048576,
        "resolution_type": "area",
        "caption_strategy": extra.get("caption_strategy"),
        "instance_data_dir": instance_data_dir,
        "hash_filenames": True,
        "maximum_image_size": 4.194304,
        "target_downsample_size": 1.048576,
        "dataset_type": dataset_type,
        "probability": extra.get("probability", 1.0),
    }
    if extra.get("conditioning_data") is not None:
        cfg["conditioning_data"] = extra["conditioning_data"]
    if extra.get("conditioning_type") is not None:
        cfg["conditioning_type"] = extra["conditioning_type"]
    return cfg


def setup_simpletuner(backend_id: str, config: dict) -> None:
    from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
    from simpletuner.helpers.training.state_tracker import StateTracker

    StateTracker.set_args(
        SimpleNamespace(
            aspect_bucket_alignment=64,
            aspect_bucket_rounding=2,
            framerate=None,
            output_dir="/tmp",
        )
    )
    StateTracker.set_data_backend_config(backend_id, config)
    return TrainingSample


def compute_one(TrainingSample, backend_id: str, path: str, size: tuple[int, int]) -> dict:
    sample = TrainingSample(
        image=None,
        data_backend_id=backend_id,
        image_metadata={"original_size": [int(size[0]), int(size[1])]},
        image_path=path,
        model=None,
    )
    prepared = sample.prepare()
    return {
        "original_size": [int(prepared.original_size[0]), int(prepared.original_size[1])],
        "crop_coordinates": [int(prepared.crop_coordinates[0]), int(prepared.crop_coordinates[1])],
        "target_size": [int(prepared.target_size[0]), int(prepared.target_size[1])],
        "intermediary_size": [int(prepared.intermediary_size[0]), int(prepared.intermediary_size[1])],
        "aspect_ratio": float(prepared.aspect_ratio),
    }


def remap_paths(obj, src_dir: str, dst_dir: str):
    src = src_dir.rstrip("/")
    dst = dst_dir.rstrip("/")
    if isinstance(obj, dict):
        return {k.replace(src, dst): remap_paths(v, src, dst) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_paths(x, src, dst) for x in obj]
    if isinstance(obj, str):
        return obj.replace(src, dst)
    return obj


def write_bucket_pair(folder: Path, backend_id: str, config: dict, indices: dict, metadata: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    idx_path = folder / f"aspect_ratio_bucket_indices_{backend_id}.json"
    meta_path = folder / f"aspect_ratio_bucket_metadata_{backend_id}.json"
    idx_path.write_text(
        json.dumps({"config": config, "aspect_ratio_bucket_indices": indices}, indent=None),
        encoding="utf-8",
    )
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    print(f"[write] {idx_path}  ({idx_path.stat().st_size / 1e6:.1f}MB)")
    print(f"[write] {meta_path}  ({meta_path.stat().st_size / 1e6:.1f}MB)")


def write_all_image_files(output_dir: Path, backend_id: str, paths: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {p: False for p in paths}
    out = output_dir / f"all_image_files_{backend_id}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[write] {out}  n={len(payload)}")


def build_from_folder(
    *,
    folder: Path,
    backend_id: str,
    dataset_type: str,
    extra: dict,
    workers: int,
    min_edge: int,
) -> tuple[dict, dict, dict, list[str]]:
    print(f"[scan] {folder}")
    files = iter_images(folder)
    print(f"[scan] found {len(files)} images")
    sizes: dict[str, tuple[int, int]] = {}
    skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(read_image_size, p): p for p in files}
        done = 0
        for fut in as_completed(futs):
            path = futs[fut]
            size = fut.result()
            done += 1
            if done % 5000 == 0 or done == len(files):
                print(f"[size] {done}/{len(files)}", flush=True)
            if size is None or min(size) < min_edge:
                skipped += 1
                continue
            sizes[path] = size
    print(f"[size] kept={len(sizes)} skipped={skipped}")

    import random

    random.seed(42)
    TrainingSample = setup_simpletuner(backend_id, backend_config(str(folder), dataset_type, extra))
    metadata: dict[str, dict] = {}
    indices: dict[str, list[str]] = {}
    for i, (path, size) in enumerate(sizes.items()):
        meta = compute_one(TrainingSample, backend_id, path, size)
        metadata[path] = meta
        key = str(meta["aspect_ratio"])
        indices.setdefault(key, []).append(path)
        if (i + 1) % 10000 == 0:
            print(f"[meta] {i + 1}/{len(sizes)}", flush=True)
    cfg = backend_config(str(folder), dataset_type, extra)
    return cfg, indices, metadata, list(sizes.keys())


def dump_all_image_files_from_existing(indices_json: Path, output_dir: Path, backend_id: str) -> None:
    data = json.loads(indices_json.read_text())
    buckets = data.get("aspect_ratio_bucket_indices", data)
    paths: list[str] = []
    if isinstance(buckets, dict):
        for v in buckets.values():
            if isinstance(v, list):
                paths.extend(v)
    write_all_image_files(output_dir, backend_id, paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oss-root", default="/mnt/afs_gaochengmin/data/oss_edit/simpletuner_pairs")
    parser.add_argument(
        "--output-dir",
        default="/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17/EditFlow/work_dirs/simpletuner_kontext_sft/output/flux_kontext_sft_oss30_pico70",
    )
    parser.add_argument(
        "--pico-root",
        default="/mnt/afs_gaochengmin/data/pico-banana-400k/simpletuner_pairs",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--min-edge", type=int, default=512)
    parser.add_argument("--skip-pico-filelist", action="store_true")
    args = parser.parse_args()

    oss_root = Path(args.oss_root)
    edit_dir = oss_root / "train" / "edit"
    ref_dir = oss_root / "train" / "reference"
    if not edit_dir.is_dir():
        print(f"missing {edit_dir}", file=sys.stderr)
        return 1

    cfg, indices, metadata, paths = build_from_folder(
        folder=edit_dir,
        backend_id="oss-edit",
        dataset_type="image",
        extra={
            "crop_style": "random",
            "caption_strategy": "textfile",
            "conditioning_data": ["oss-reference"],
            "probability": 0.3,
        },
        workers=args.workers,
        min_edge=args.min_edge,
    )
    write_bucket_pair(edit_dir, "oss-edit", cfg, indices, metadata)

    ref_cfg = backend_config(
        str(ref_dir),
        "conditioning",
        {
            "crop_style": "random",
            "caption_strategy": None,
            "conditioning_type": "reference_strict",
            "probability": 1.0,
        },
    )
    ref_indices = remap_paths(indices, str(edit_dir), str(ref_dir))
    ref_metadata = remap_paths(metadata, str(edit_dir), str(ref_dir))
    write_bucket_pair(ref_dir, "oss-reference", ref_cfg, ref_indices, ref_metadata)

    out_dir = Path(args.output_dir)
    write_all_image_files(out_dir, "oss-edit", paths)
    write_all_image_files(out_dir, "oss-reference", [p.replace(str(edit_dir), str(ref_dir)) for p in paths])

    if not args.skip_pico_filelist:
        pico = Path(args.pico_root)
        mapping = [
            (pico / "train/edit/aspect_ratio_bucket_indices_pico-banana-edit.json", "pico-banana-edit"),
            (
                pico / "train/reference/aspect_ratio_bucket_indices_pico-banana-reference.json",
                "pico-banana-reference",
            ),
            (pico / "val/edit/aspect_ratio_bucket_indices_pico-banana-edit-val.json", "pico-banana-edit-val"),
            (
                pico / "val/reference/aspect_ratio_bucket_indices_pico-banana-reference-val.json",
                "pico-banana-reference-val",
            ),
        ]
        for json_path, backend_id in mapping:
            if json_path.is_file():
                dump_all_image_files_from_existing(json_path, out_dir, backend_id)
            else:
                print(f"[skip] missing {json_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
