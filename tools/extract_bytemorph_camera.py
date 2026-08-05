#!/usr/bin/env python3
"""Extract Camera Zoom / Camera Motion pairs from ByteDance-Seed/BM-6M-Demo.

BM-Bench only has ~114 labeled camera pairs. This script filters BM-6M-Demo
(~780k) by instruction text into:
  - camera_zoom
  - camera_motion
and downloads ~10k pairs into an independent folder (not HQ_Edit_filtered).

Output:
  ByteMorph_camera/
    camera_zoom/
      000001_input.png
      000001_output.png
      000001_prompt.txt
    camera_motion/
      ...
    metadata.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fsspec
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url
from PIL import Image
from tqdm import tqdm


REPO = "ByteDance-Seed/BM-6M-Demo"
TEXT_COLS = [
    "image_id",
    "edit_prompt",
    "edit_prompt_rewrite_instruction",
    "src_img_caption",
    "tgt_img_caption",
]

# Prefer rewrite instruction; fall back to edit_prompt.
ZOOM_PAT = re.compile(
    r"\bzoom\s*(in|out)?\b|"
    r"\b(closer|farther|further)\s+(view|shot|angle|perspective)\b|"
    r"\b(moves?\s+)?closer\b|"
    r"\b(moves?\s+)?further\s+away\b|"
    r"\bpull\s+back\b|"
    r"\bdolly\s*(in|out)\b",
    re.I,
)
MOTION_PAT = re.compile(
    r"\bcamera\b.{0,40}\b(left|right|up|down|upward|downward|pan|tilt|orbit)\b|"
    r"\b(left|right|up|down|upward|downward|pan|tilt|orbit).{0,40}\bcamera\b|"
    r"\bcamera\s+(moves?|shifts?|pans?|tilts?)\b|"
    r"\b(moves?|shifts?)\s+the\s+camera\b|"
    r"\bviewpoint\s+(moves?|shifts?)\b|"
    r"\bcamera\s+angle\s+shifts?\b",
    re.I,
)


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def parquet_url(filename: str) -> str:
    return hf_hub_url(
        repo_id=REPO,
        filename=filename,
        repo_type="dataset",
        endpoint=endpoint(),
    )


def list_parquets() -> List[str]:
    api = HfApi(endpoint=endpoint())
    files = [
        f
        for f in api.list_repo_files(REPO, repo_type="dataset")
        if f.endswith(".parquet") and f.startswith("data/")
    ]
    return sorted(files)


def classify_camera(edit: str, rewrite: str) -> Optional[str]:
    text = f"{rewrite or ''}\n{edit or ''}".strip()
    if not text:
        return None
    has_zoom = bool(ZOOM_PAT.search(text))
    has_motion = bool(MOTION_PAT.search(text))
    # Explicit zoom wins over generic camera-shift wording.
    if has_zoom and re.search(r"\bzoom\b", text, re.I):
        return "camera_zoom"
    if has_zoom and not has_motion:
        return "camera_zoom"
    if has_motion and not has_zoom:
        return "camera_motion"
    if has_zoom and has_motion:
        # both: prefer zoom if "zoom" present else motion
        return "camera_zoom" if re.search(r"\bzoom\b", text, re.I) else "camera_motion"
    # weaker camera-only cues for motion (left/right framing changes)
    if re.search(r"\bcamera\b", text, re.I) and re.search(
        r"\b(left|right|up|down)\b", text, re.I
    ):
        return "camera_motion"
    return None


def fetch_text_rows(filename: str, retries: int = 4) -> List[Dict[str, Any]]:
    url = parquet_url(filename)
    last_err: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            fs = fsspec.filesystem("https")
            with fs.open(url, "rb") as f:
                table = pq.ParquetFile(f).read(columns=TEXT_COLS)
            data = table.to_pydict()
            n = len(data["image_id"])
            rows = []
            for i in range(n):
                rows.append(
                    {
                        "parquet": filename,
                        "row_idx": i,
                        "image_id": data["image_id"][i],
                        "edit_prompt": data["edit_prompt"][i],
                        "edit_prompt_rewrite_instruction": data[
                            "edit_prompt_rewrite_instruction"
                        ][i],
                        "src_img_caption": data["src_img_caption"][i],
                        "tgt_img_caption": data["tgt_img_caption"][i],
                    }
                )
            return rows
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed {filename}: {last_err}")


def extract_metadata(meta_dir: Path, workers: int) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    files = list_parquets()
    print(f"[meta] {len(files)} parquet shards from {REPO}")
    todo = []
    for f in files:
        out = meta_dir / (Path(f).stem + ".jsonl")
        if out.exists() and out.stat().st_size > 0:
            continue
        todo.append(f)
    print(f"[meta] need {len(todo)} / {len(files)}")

    def job(fn: str) -> Tuple[str, int, Optional[str]]:
        try:
            rows = fetch_text_rows(fn)
            out = meta_dir / (Path(fn).stem + ".jsonl")
            tmp = out.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as wf:
                for r in rows:
                    cat = classify_camera(
                        r.get("edit_prompt") or "",
                        r.get("edit_prompt_rewrite_instruction") or "",
                    )
                    if cat is None:
                        continue
                    r = dict(r)
                    r["category"] = cat
                    wf.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(out)
            # count kept
            n = sum(1 for _ in out.open())
            return fn, n, None
        except Exception as e:  # noqa: BLE001
            return fn, 0, repr(e)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(job, f) for f in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="meta"):
                fn, n, err = fut.result()
                if err:
                    print(f"[meta] FAIL {fn}: {err}", file=sys.stderr)
                else:
                    print(f"[meta] {fn}: kept {n}")

    all_meta = meta_dir / "camera_candidates.jsonl"
    with all_meta.open("w", encoding="utf-8") as out:
        for f in files:
            p = meta_dir / (Path(f).stem + ".jsonl")
            if p.exists():
                out.write(p.read_text(encoding="utf-8"))
    # stats
    c = Counter()
    total = 0
    with all_meta.open() as f:
        for line in f:
            if line.strip():
                c[json.loads(line)["category"]] += 1
                total += 1
    print(f"[meta] candidates total={total} {dict(c)}")
    print(f"[meta] wrote {all_meta}")
    return all_meta


def dedup_key(row: Dict[str, Any]) -> str:
    raw = f"{row.get('image_id','')}||{row.get('edit_prompt_rewrite_instruction','')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def sample_selection(
    candidates_path: Path,
    out_path: Path,
    n_zoom: int,
    n_motion: int,
    seed: int,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "camera_zoom": [],
        "camera_motion": [],
    }
    with candidates_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            buckets[r["category"]].append(r)

    print(
        f"[sample] pool zoom={len(buckets['camera_zoom'])} "
        f"motion={len(buckets['camera_motion'])}"
    )
    rng = random.Random(seed)
    used = set()
    selected: List[Dict[str, Any]] = []
    # Prefer packing by parquet shard to reduce downloads.
    for cat, need in [("camera_zoom", n_zoom), ("camera_motion", n_motion)]:
        by_shard: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in buckets[cat]:
            by_shard[r["parquet"]].append(r)
        # richest shards first
        shards = sorted(by_shard.keys(), key=lambda s: len(by_shard[s]), reverse=True)
        taken = 0
        for sh in shards:
            if taken >= need:
                break
            rows = by_shard[sh]
            rng.shuffle(rows)
            for r in rows:
                if taken >= need:
                    break
                k = dedup_key(r)
                if k in used:
                    continue
                used.add(k)
                selected.append(r)
                taken += 1
        print(f"[sample] {cat}: {taken}/{need}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sample] wrote {out_path} ({len(selected)} pairs)")
    print(f"[sample] unique shards={len({r['parquet'] for r in selected})}")
    return selected


def _save_pair(
    in_b: bytes,
    out_b: bytes,
    instruction: str,
    cat_dir: Path,
    idx: int,
) -> Tuple[str, str]:
    stem = f"{idx:06d}"
    in_path = cat_dir / f"{stem}_input.png"
    out_path = cat_dir / f"{stem}_output.png"
    prompt_path = cat_dir / f"{stem}_prompt.txt"
    Image.open(io.BytesIO(in_b)).convert("RGB").save(in_path, format="PNG")
    Image.open(io.BytesIO(out_b)).convert("RGB").save(out_path, format="PNG")
    prompt_path.write_text(instruction.strip() + "\n", encoding="utf-8")
    return str(in_path), str(out_path)


def download_selected(
    selected: List[Dict[str, Any]],
    out_root: Path,
    shard_workers: int,
    save_workers: int,
) -> None:
    for cat in ("camera_zoom", "camera_motion"):
        (out_root / cat).mkdir(parents=True, exist_ok=True)

    # stable indices per category
    selected_sorted = sorted(
        selected, key=lambda r: (r["category"], r["parquet"], r["row_idx"])
    )
    counters: Dict[str, int] = defaultdict(int)
    for r in selected_sorted:
        counters[r["category"]] += 1
        r["save_idx"] = counters[r["category"]]

    by_shard: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in selected_sorted:
        by_shard[r["parquet"]].append(r)

    meta_jsonl = out_root / "metadata.jsonl"
    existing = set()
    if meta_jsonl.exists():
        for line in meta_jsonl.open():
            if line.strip():
                existing.add(json.loads(line)["id"])

    shard_jobs = []
    for sh, rows in by_shard.items():
        pending = []
        for r in rows:
            rid = f"{r['category']}/{r['save_idx']:06d}"
            in_p = out_root / r["category"] / f"{r['save_idx']:06d}_input.png"
            out_p = out_root / r["category"] / f"{r['save_idx']:06d}_output.png"
            if rid in existing and in_p.exists() and out_p.exists():
                continue
            pending.append(r)
        if pending:
            shard_jobs.append((sh, pending))

    print(f"[download] {len(shard_jobs)} shards pending / {len(by_shard)} selected")
    written = 0
    lock = threading.Lock()

    def one_shard(sh: str, pending: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        url = parquet_url(sh)
        table = None
        last_err = None
        for attempt in range(5):
            try:
                fs = fsspec.filesystem("https")
                with fs.open(url, "rb") as f:
                    table = pq.ParquetFile(f).read(columns=["src_img", "tgt_img"])
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        if table is None:
            print(f"[download] SKIP {sh}: {last_err}", file=sys.stderr)
            return []

        # src_img / tgt_img may be struct{bytes,path} or raw bytes depending on writer
        def to_bytes(cell) -> bytes:
            v = cell.as_py()
            if isinstance(v, dict):
                b = v.get("bytes")
                if b is not None:
                    return b
                raise ValueError(f"unexpected image dict keys: {list(v)}")
            if isinstance(v, (bytes, bytearray)):
                return bytes(v)
            raise ValueError(f"unexpected image type: {type(v)}")

        recs = []
        with ThreadPoolExecutor(max_workers=max(2, save_workers)) as ex:
            futs = []
            for r in pending:
                i = int(r["row_idx"])
                instr = (
                    r.get("edit_prompt_rewrite_instruction")
                    or r.get("edit_prompt")
                    or ""
                )
                futs.append(
                    ex.submit(
                        _save_pair,
                        to_bytes(table.column("src_img")[i]),
                        to_bytes(table.column("tgt_img")[i]),
                        instr,
                        out_root / r["category"],
                        int(r["save_idx"]),
                    )
                )
            for fut, r in zip(futs, pending):
                in_path, out_path = fut.result()
                instr = (
                    r.get("edit_prompt_rewrite_instruction")
                    or r.get("edit_prompt")
                    or ""
                )
                recs.append(
                    {
                        "id": f"{r['category']}/{r['save_idx']:06d}",
                        "input_path": in_path,
                        "output_path": out_path,
                        "instruction": instr,
                        "category": r["category"],
                        "image_id": r.get("image_id"),
                        "edit_prompt": r.get("edit_prompt"),
                        "src_img_caption": r.get("src_img_caption"),
                        "tgt_img_caption": r.get("tgt_img_caption"),
                        "parquet": r["parquet"],
                        "row_idx": r["row_idx"],
                        "source_dataset": REPO,
                    }
                )
        del table
        return recs

    with meta_jsonl.open("a", encoding="utf-8") as meta_f:
        with ThreadPoolExecutor(max_workers=max(1, shard_workers)) as ex:
            futs = {
                ex.submit(one_shard, sh, pending): sh for sh, pending in shard_jobs
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc="download"):
                sh = futs[fut]
                try:
                    recs = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[download] FAIL {sh}: {e}", file=sys.stderr)
                    continue
                with lock:
                    for rec in recs:
                        meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                    meta_f.flush()

    print(f"[download] newly wrote {written} pairs -> {out_root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/mnt/afs_zhangyunzhe/dataset/ByteMorph_camera"),
    )
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--shard-workers", type=int, default=3)
    p.add_argument("--n-zoom", type=int, default=5000)
    p.add_argument("--n-motion", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stage",
        choices=["meta", "sample", "download", "all"],
        default="all",
    )
    p.add_argument("--force-resample", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    meta_dir = out_root / "_meta"
    cand_path = meta_dir / "camera_candidates.jsonl"
    sel_path = meta_dir / "selection.jsonl"

    t0 = time.time()
    if args.stage in ("meta", "all"):
        cand_path = extract_metadata(meta_dir, workers=args.workers)

    if args.stage in ("sample", "all"):
        if not cand_path.exists():
            raise SystemExit(f"missing {cand_path}; run --stage meta")
        if sel_path.exists() and not args.force_resample and args.stage != "sample":
            print(f"[sample] reuse existing {sel_path}")
        else:
            sample_selection(
                cand_path,
                sel_path,
                n_zoom=args.n_zoom,
                n_motion=args.n_motion,
                seed=args.seed,
            )

    if args.stage in ("download", "all"):
        selected = [
            json.loads(l)
            for l in sel_path.open()
            if l.strip()
        ]
        download_selected(
            selected,
            out_root=out_root,
            shard_workers=args.shard_workers,
            save_workers=max(2, args.workers // max(1, args.shard_workers)),
        )

    print(f"[done] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
