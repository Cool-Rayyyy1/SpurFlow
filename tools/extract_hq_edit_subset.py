#!/usr/bin/env python3
"""Extract a keyword-filtered subset from UCSC-VLAA/HQ-Edit.

Pipeline:
  1) Remotely read only text columns from parquet shards (no full image download)
  2) Filter by edit-instruction keywords
  3) Print category stats and sample to quotas
  4) Estimate disk usage, then download only selected input/output images

Uses HF_ENDPOINT (default https://hf-mirror.com). Unset HF_HUB_OFFLINE while running.
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fsspec
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url
from PIL import Image
from tqdm import tqdm


REPO_ID = "UCSC-VLAA/HQ-Edit"

# Priority order: earlier categories win when a sample matches multiple.
# Patterns follow the requested keyword list, with light FP guards.
CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "camera_motion": [
        r"\bzoom\s*in\b",
        r"\bzoom\s*out\b",
        r"\bzoom\b",
        r"\bviewpoint\b",
        r"\bperspective\b",
        r"\bangle\b",
        r"\bclose[\s-]?up\b",
        r"\bwide\s+shot\b",
        r"\bcloser\b",
        r"\bfarther\b",
        r"\bchange\s+view\b",
        # "camera" as viewpoint/motion, not object-only ("holding a camera")
        r"\bcamera\s+(angle|perspective|view|zoom|pan|tilt|move|movement|shot)\b",
        r"\b(move|bring|pull|push)\s+the\s+camera\b",
        r"\bfrom\s+the\s+camera\b",
    ],
    "size_adjustment": [
        r"\bincrease\s+size\b",
        r"\bdecrease\s+size\b",
        r"\blarger\b",
        r"\bsmaller\b",
        r"\bbigger\b",
        r"\bresize\b",
        r"\benlarg(e|ed|es|ing)\b",
        r"\bshrink\b",
        r"\bscale\b",
    ],
    "portrait_beautification": [
        r"\bbeautify\b",
        r"\bbeauty\b",
        r"\bretouch\b",
        r"\bsmooth\s+skin\b",
        r"\benhance\s+face\b",
        r"\bremove\s+wrinkles\b",
        r"\battractive\b",
        r"\bimprove\s+appearance\b",
        r"\bfacial\b",
        # useful near-synonyms still in-distribution for portrait edits
        r"\bwrinkles?\b",
        r"\bskin\s+texture\b",
        r"\bcomplexion\b",
    ],
    "composition": [
        r"\bin\s+front\s+of\b",
        r"\bnext\s+to\b",
        r"\bbehind\b",
        r"\bposition\b",
        r"\bmove\b",
        r"\bto\s+the\s+left\b",
        r"\bto\s+the\s+right\b",
        r"\bleft\s+of\b",
        r"\bright\s+of\b",
        r"\bon\s+the\s+left\b",
        r"\bon\s+the\s+right\b",
    ],
}

# Reject obvious false positives after a keyword hit.
CATEGORY_NEGATIVE: Dict[str, List[str]] = {
    "size_adjustment": [
        r"\blarger\s+(narrative|context|environment|scene|setting|world)\b",
        r"\bsmaller\s+(version\s+of\s+the\s+same\s+scene)\b",
    ],
    "portrait_beautification": [
        r"\bbeauty\s+of\s+the\s+(scene|landscape|nature|garden|sunset|ocean|view)\b",
        r"\bdelicate\s+beauty\s+of\s+the\s+scene\b",
    ],
    "camera_motion": [
        r"\b(holding|holds|with|equipped with|carrying)\s+(a\s+)?(vintage\s+|digital\s+)?camera\b",
        r"\breplace\b.{0,40}\bcamera\b",
        r"\badd\b.{0,40}\bcamera\b",
    ],
}

SAMPLE_QUOTAS = {
    "camera_motion": 2000,
    "size_adjustment": 3000,
    "portrait_beautification": 3000,
    "composition": 2000,
}


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


COMPILED = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in CATEGORY_PATTERNS.items()
}
COMPILED_NEG = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in CATEGORY_NEGATIVE.items()
}


def classify_edit(edit: str) -> Optional[str]:
    text = (edit or "").strip()
    if not text:
        return None
    for cat, pats in COMPILED.items():
        if not any(pat.search(text) for pat in pats):
            continue
        if any(pat.search(text) for pat in COMPILED_NEG.get(cat, [])):
            continue
        return cat
    return None


def list_parquet_files() -> List[str]:
    api = HfApi(endpoint=_endpoint())
    files = [
        f
        for f in api.list_repo_files(REPO_ID, repo_type="dataset")
        if f.endswith(".parquet") and f.startswith("data/")
    ]
    # numeric sort: HQ-Edit_0, HQ-Edit_1, ... HQ-Edit_607
    def key(p: str) -> int:
        stem = Path(p).stem  # HQ-Edit_12
        return int(stem.split("_")[-1])

    return sorted(files, key=key)


def parquet_url(filename: str) -> str:
    return hf_hub_url(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        endpoint=_endpoint(),
    )


def shard_id_from_path(filename: str) -> int:
    return int(Path(filename).stem.split("_")[-1])


def fetch_text_metadata(filename: str, retries: int = 4) -> List[Dict[str, Any]]:
    """Read only text columns from one remote parquet shard."""
    url = parquet_url(filename)
    sid = shard_id_from_path(filename)
    text_cols = ["input", "edit", "inverse_edit", "output"]
    last_err: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            fs = fsspec.filesystem("https")
            with fs.open(url, "rb") as f:
                table = pq.ParquetFile(f).read(columns=text_cols)
            rows = []
            data = table.to_pydict()
            n = len(data["edit"])
            for i in range(n):
                rows.append(
                    {
                        "shard_id": sid,
                        "row_idx": i,
                        "parquet": filename,
                        "input": data["input"][i],
                        "edit": data["edit"][i],
                        "inverse_edit": data["inverse_edit"][i],
                        "output": data["output"][i],
                    }
                )
            return rows
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to read {filename} after {retries} tries: {last_err}")


def _meta_shard_path(meta_dir: Path, shard_id: int) -> Path:
    return meta_dir / f"shard_{shard_id:04d}.jsonl"


def extract_all_metadata(
    meta_dir: Path,
    workers: int,
    resume: bool = True,
) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    files = list_parquet_files()
    print(f"[meta] {len(files)} parquet shards from {REPO_ID} via {_endpoint()}")

    todo = []
    for f in files:
        sid = shard_id_from_path(f)
        out = _meta_shard_path(meta_dir, sid)
        if resume and out.exists() and out.stat().st_size > 0:
            continue
        todo.append(f)
    print(f"[meta] need to fetch {len(todo)} / {len(files)} shards")

    def _job(filename: str) -> Tuple[str, int, Optional[str]]:
        try:
            rows = fetch_text_metadata(filename)
            sid = shard_id_from_path(filename)
            out = _meta_shard_path(meta_dir, sid)
            tmp = out.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as wf:
                for r in rows:
                    wf.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(out)
            return filename, len(rows), None
        except Exception as e:  # noqa: BLE001
            return filename, 0, repr(e)

    errors = []
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_job, f) for f in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="metadata"):
                fn, n, err = fut.result()
                if err:
                    errors.append((fn, err))
                    print(f"[meta] FAIL {fn}: {err}", file=sys.stderr)

    if errors:
        print(f"[meta] {len(errors)} shards failed; re-run to resume", file=sys.stderr)

    all_meta = meta_dir / "all_metadata.jsonl"
    # Concatenate shard jsonls into one file for convenience
    with all_meta.open("w", encoding="utf-8") as out:
        for f in files:
            sid = shard_id_from_path(f)
            p = _meta_shard_path(meta_dir, sid)
            if not p.exists():
                continue
            out.write(p.read_text(encoding="utf-8"))
    print(f"[meta] wrote {all_meta}")
    return all_meta


def iter_metadata(meta_path: Path) -> Iterable[Dict[str, Any]]:
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def filter_candidates(meta_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORY_PATTERNS}
    total = 0
    matched = 0
    for row in tqdm(iter_metadata(meta_path), desc="filter"):
        total += 1
        cat = classify_edit(row.get("edit", ""))
        if cat is None:
            continue
        matched += 1
        row = dict(row)
        row["category"] = cat
        buckets[cat].append(row)
    print(f"[filter] total={total} matched={matched}")
    for cat, rows in buckets.items():
        print(f"  {cat}: {len(rows)}")
    return buckets


def dedup_key(row: Dict[str, Any]) -> str:
    """Dedup by edit instruction + input caption (stable content key)."""
    raw = f"{row.get('edit','')}||{row.get('input','')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def sample_buckets(
    buckets: Dict[str, List[Dict[str, Any]]],
    quotas: Dict[str, int],
    seed: int,
    greedy_shards: bool = True,
) -> List[Dict[str, Any]]:
    """Sample per-category with global content dedup.

    If greedy_shards=True, fill quotas by iterating shards so fewer parquet
    files need to be downloaded for images.
    """
    rng = random.Random(seed)
    used_keys = set()
    selected: List[Dict[str, Any]] = []
    remaining = {c: quotas[c] for c in quotas}

    if greedy_shards:
        by_shard: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for cat, rows in buckets.items():
            for r in rows:
                by_shard[int(r["shard_id"])].append(r)

        def shard_score(sid: int) -> Tuple[int, int]:
            rows = by_shard[sid]
            # Prefer shards that still help scarce categories and have more usable rows.
            useful = 0
            scarce_help = 0
            for r in rows:
                cat = r["category"]
                if remaining.get(cat, 0) > 0:
                    useful += 1
                    # size/portrait are scarce; prioritize packing them
                    if cat in ("size_adjustment", "portrait_beautification"):
                        scarce_help += 3
                    elif cat == "camera_motion":
                        scarce_help += 1
            return (scarce_help, useful)

        # Recompute order iteratively in batches for better packing.
        leftover = set(by_shard.keys())
        while leftover and any(v > 0 for v in remaining.values()):
            ranked = sorted(leftover, key=shard_score, reverse=True)
            sid = ranked[0]
            leftover.remove(sid)
            rows = by_shard[sid]
            rng.shuffle(rows)
            gained = False
            for r in rows:
                cat = r["category"]
                if remaining.get(cat, 0) <= 0:
                    continue
                k = dedup_key(r)
                if k in used_keys:
                    continue
                used_keys.add(k)
                selected.append(r)
                remaining[cat] -= 1
                gained = True
            if not gained and all(shard_score(s)[1] == 0 for s in leftover):
                break
    else:
        for cat, need in quotas.items():
            rows = list(buckets.get(cat, []))
            rng.shuffle(rows)
            taken = 0
            for r in rows:
                if taken >= need:
                    break
                k = dedup_key(r)
                if k in used_keys:
                    continue
                used_keys.add(k)
                selected.append(r)
                taken += 1
            remaining[cat] = need - taken

    print("[sample] selected:")
    for cat in quotas:
        n = sum(1 for r in selected if r["category"] == cat)
        print(f"  {cat}: {n} / {quotas[cat]} (shortfall {remaining[cat]})")
    print(f"  total pairs: {len(selected)}")
    print(f"  unique shards touched: {len({r['shard_id'] for r in selected})}")
    return selected


def estimate_disk_usage(selected: List[Dict[str, Any]], probe_shards: int = 3) -> Dict[str, Any]:
    """Download a few selected rows' images to estimate average pair size."""
    by_shard: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_shard[int(r["shard_id"])].append(r)
    shard_ids = sorted(by_shard.keys())[:probe_shards]
    sizes = []
    probed = 0
    fs = fsspec.filesystem("https")
    for sid in shard_ids:
        rows = by_shard[sid][:8]
        parquet = rows[0]["parquet"]
        url = parquet_url(parquet)
        print(f"[estimate] probing shard {sid} ({parquet}) with {len(rows)} rows...")
        with fs.open(url, "rb") as f:
            table = pq.ParquetFile(f).read(columns=["input_image", "output_image"])
        for r in rows:
            i = int(r["row_idx"])
            in_b = table.column("input_image")[i].as_py()
            out_b = table.column("output_image")[i].as_py()
            # decode + re-encode as PNG to match save path
            def png_nbytes(b: bytes) -> int:
                im = Image.open(io.BytesIO(b))
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="PNG")
                return buf.tell()

            pair = png_nbytes(in_b) + png_nbytes(out_b)
            sizes.append(pair)
            probed += 1
    avg = sum(sizes) / len(sizes) if sizes else 0
    est_total = avg * len(selected)
    # prompts/jsonl negligible
    info = {
        "probed_pairs": probed,
        "avg_pair_png_bytes": avg,
        "avg_pair_png_mb": avg / 1024 / 1024,
        "num_pairs": len(selected),
        "estimated_total_gb": est_total / 1024**3,
    }
    print("[estimate]", json.dumps(info, indent=2))
    return info


def _save_one_pair(
    args: Tuple[bytes, bytes, str, Path, str, int],
) -> Tuple[str, str, str]:
    in_b, out_b, instruction, cat_dir, category, idx = args
    stem = f"{idx:06d}"
    in_path = cat_dir / f"{stem}_input.png"
    out_path = cat_dir / f"{stem}_output.png"
    prompt_path = cat_dir / f"{stem}_prompt.txt"
    Image.open(io.BytesIO(in_b)).convert("RGB").save(in_path, format="PNG")
    Image.open(io.BytesIO(out_b)).convert("RGB").save(out_path, format="PNG")
    prompt_path.write_text(instruction.strip() + "\n", encoding="utf-8")
    return str(in_path), str(out_path), category


def _download_one_shard(
    sid: int,
    pending: List[Dict[str, Any]],
    out_root: Path,
    save_workers: int,
) -> List[Dict[str, Any]]:
    """Download one parquet shard and save pending pairs. Returns metadata records."""
    parquet = pending[0]["parquet"]
    url = parquet_url(parquet)
    table = None
    last_err: Optional[BaseException] = None
    for attempt in range(4):
        try:
            fs = fsspec.filesystem("https")
            with fs.open(url, "rb") as f:
                table = pq.ParquetFile(f).read(columns=["input_image", "output_image"])
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    if table is None:
        print(f"[download] SKIP shard {sid}: {last_err}", file=sys.stderr)
        return []

    jobs = []
    for r in pending:
        i = int(r["row_idx"])
        jobs.append(
            (
                table.column("input_image")[i].as_py(),
                table.column("output_image")[i].as_py(),
                r.get("edit") or "",
                out_root / r["category"],
                r["category"],
                int(r["save_idx"]),
            )
        )

    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(2, save_workers)) as ex:
        futs = {ex.submit(_save_one_pair, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            in_path, out_path, category = fut.result()
            idx = j[-1]
            instruction = j[2]
            row = next(x for x in pending if int(x["save_idx"]) == idx)
            records.append(
                {
                    "id": f"{category}/{idx:06d}",
                    "input_path": in_path,
                    "output_path": out_path,
                    "instruction": instruction,
                    "category": category,
                    "shard_id": row["shard_id"],
                    "row_idx": row["row_idx"],
                    "input_caption": row.get("input"),
                    "output_caption": row.get("output"),
                    "inverse_edit": row.get("inverse_edit"),
                }
            )
    del table
    return records


def download_selected_images(
    selected: List[Dict[str, Any]],
    out_root: Path,
    workers: int,
    max_gb: float,
    estimated_gb: Optional[float],
    shard_workers: int = 4,
) -> Path:
    if estimated_gb is not None and estimated_gb > max_gb:
        raise SystemExit(
            f"Estimated disk usage {estimated_gb:.2f} GB exceeds limit {max_gb:.2f} GB. Abort."
        )

    for cat in SAMPLE_QUOTAS:
        (out_root / cat).mkdir(parents=True, exist_ok=True)

    # Assign stable indices per category
    selected_sorted = sorted(
        selected, key=lambda r: (r["category"], r["shard_id"], r["row_idx"])
    )
    cat_counters = defaultdict(int)
    for r in selected_sorted:
        cat_counters[r["category"]] += 1
        r["save_idx"] = cat_counters[r["category"]]

    by_shard: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in selected_sorted:
        by_shard[int(r["shard_id"])].append(r)

    meta_jsonl = out_root / "metadata.jsonl"
    # Resume: skip pairs that already have both images
    existing = set()
    if meta_jsonl.exists():
        for line in meta_jsonl.open("r", encoding="utf-8"):
            if line.strip():
                existing.add(json.loads(line)["id"])

    shard_jobs: List[Tuple[int, List[Dict[str, Any]]]] = []
    for sid in sorted(by_shard.keys()):
        pending = []
        for r in by_shard[sid]:
            rid = f"{r['category']}/{r['save_idx']:06d}"
            in_p = out_root / r["category"] / f"{r['save_idx']:06d}_input.png"
            out_p = out_root / r["category"] / f"{r['save_idx']:06d}_output.png"
            if rid in existing and in_p.exists() and out_p.exists():
                continue
            pending.append(r)
        if pending:
            shard_jobs.append((sid, pending))

    print(
        f"[download] {len(shard_jobs)} shards pending "
        f"(shard_workers={shard_workers}, save_workers={workers})"
    )

    written = 0
    meta_lock = threading.Lock()
    save_workers = max(2, workers // max(1, shard_workers))
    with meta_jsonl.open("a", encoding="utf-8") as meta_f:
        with ThreadPoolExecutor(max_workers=max(1, shard_workers)) as ex:
            futs = {
                ex.submit(_download_one_shard, sid, pending, out_root, save_workers): sid
                for sid, pending in shard_jobs
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc="download shards"):
                sid = futs[fut]
                try:
                    records = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[download] FAIL shard {sid}: {e}", file=sys.stderr)
                    continue
                with meta_lock:
                    for rec in records:
                        meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                    meta_f.flush()

    print(f"[download] newly wrote {written} pairs -> {out_root}")
    return meta_jsonl


def save_selection(selected: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sample] wrote selection list {path} ({len(selected)} rows)")


def load_selection(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/mnt/afs_zhangyunzhe/dataset/HQ_Edit_filtered"),
    )
    p.add_argument("--meta-dir", type=Path, default=None, help="metadata cache dir")
    p.add_argument("--workers", type=int, default=16, help="image save threads (total budget)")
    p.add_argument(
        "--shard-workers",
        type=int,
        default=4,
        help="parallel parquet shard downloads",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-gb", type=float, default=80.0)
    p.add_argument(
        "--stage",
        choices=["meta", "filter", "estimate", "download", "all"],
        default="all",
    )
    p.add_argument("--no-greedy-shards", action="store_true")
    p.add_argument("--probe-shards", type=int, default=3)
    p.add_argument("--force-resample", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Ensure we can reach the hub
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        print("[warn] unsetting HF_HUB_OFFLINE for download", file=sys.stderr)
        os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    meta_dir = args.meta_dir or (out_root / "_meta")
    selection_path = out_root / "_meta" / "selection.jsonl"
    all_meta = meta_dir / "all_metadata.jsonl"

    t0 = time.time()

    if args.stage in ("meta", "all"):
        all_meta = extract_all_metadata(meta_dir, workers=args.workers, resume=True)

    if args.stage in ("filter", "estimate", "download", "all"):
        if not all_meta.exists():
            raise SystemExit(f"Missing metadata {all_meta}; run --stage meta first")
        if selection_path.exists() and not args.force_resample and args.stage != "filter":
            selected = load_selection(selection_path)
            print(f"[sample] loaded existing selection ({len(selected)})")
            # still print stats from filter for visibility
            _ = filter_candidates(all_meta)
        else:
            buckets = filter_candidates(all_meta)
            selected = sample_buckets(
                buckets,
                SAMPLE_QUOTAS,
                seed=args.seed,
                greedy_shards=not args.no_greedy_shards,
            )
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            save_selection(selected, selection_path)

    if args.stage in ("estimate", "all"):
        selected = load_selection(selection_path)
        est = estimate_disk_usage(selected, probe_shards=args.probe_shards)
        (out_root / "_meta" / "disk_estimate.json").write_text(
            json.dumps(est, indent=2), encoding="utf-8"
        )
        if est["estimated_total_gb"] > args.max_gb:
            raise SystemExit(
                f"Estimated {est['estimated_total_gb']:.2f} GB > max {args.max_gb} GB"
            )

    if args.stage in ("download", "all"):
        selected = load_selection(selection_path)
        est_path = out_root / "_meta" / "disk_estimate.json"
        est_gb = None
        if est_path.exists():
            est_gb = json.loads(est_path.read_text())["estimated_total_gb"]
        download_selected_images(
            selected,
            out_root=out_root,
            workers=max(4, args.workers),
            max_gb=args.max_gb,
            estimated_gb=est_gb,
            shard_workers=max(1, args.shard_workers),
        )

    print(f"[done] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
