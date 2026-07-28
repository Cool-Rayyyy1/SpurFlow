#!/usr/bin/env python3
"""Convert pico-banana-400k SFT jsonl into SimpleTuner Kontext paired folders.

SimpleTuner Flux Kontext requires:
  edit/       # target (edited) images + matching .txt captions
  reference/  # source images with *identical* filenames (incl. extension)

pico-banana stores mismatched names (OpenImages hash jpg vs positive-edit/N.png),
so we materialize a symlink/hardlink layout with shared stems.

Default split mirrors EditFlow ``_data_trainval.py``: last ``val_size`` rows -> val.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple


def _link_or_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from e


def _resolve_pair(
    row: dict,
    data_root: Path,
    edited_images_dir: str,
    source_column: str,
    target_column: str,
    prompt_column: str,
    summarized_prompt_column: Optional[str],
    use_summarized: bool,
) -> Optional[Tuple[Path, Path, str]]:
    src = Path(row[source_column])
    if not src.is_absolute():
        src = data_root / src
    tgt_rel = row[target_column]
    tgt = Path(tgt_rel)
    if not tgt.is_absolute():
        tgt = data_root / edited_images_dir / tgt_rel

    if use_summarized and summarized_prompt_column and row.get(summarized_prompt_column):
        prompt = str(row[summarized_prompt_column]).strip()
    else:
        prompt = str(row[prompt_column]).strip()
    if not prompt:
        return None
    if not src.is_file() or not tgt.is_file():
        return None
    return src, tgt, prompt


def _unified_ext(src: Path, tgt: Path) -> str:
    """Pick one extension so edit/reference filenames match exactly."""
    ext = tgt.suffix.lower() or src.suffix.lower() or ".png"
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        ext = ".png"
    return ext


def materialize_split(
    rows: list[Tuple[Path, Path, str]],
    out_edit: Path,
    out_ref: Path,
    start_index: int = 0,
) -> int:
    out_edit.mkdir(parents=True, exist_ok=True)
    out_ref.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, (src, tgt, prompt) in enumerate(rows):
        stem = f"{start_index + i:08d}"
        ext = _unified_ext(src, tgt)
        edit_img = out_edit / f"{stem}{ext}"
        ref_img = out_ref / f"{stem}{ext}"
        caption = out_edit / f"{stem}.txt"
        _link_or_symlink(tgt, edit_img)
        _link_or_symlink(src, ref_img)
        caption.write_text(prompt + "\n", encoding="utf-8")
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/afs_zhangyunzhe/dataset/pico-banana-400k"),
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="jsonl/sft_with_local_source_image_path.jsonl",
    )
    parser.add_argument("--edited-images-dir", type=str, default="edited_images")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Default: <data-root>/simpletuner_pairs",
    )
    parser.add_argument("--source-column", default="local_input_image")
    parser.add_argument("--target-column", default="output_image")
    parser.add_argument("--prompt-column", default="text")
    parser.add_argument("--summarized-prompt-column", default="summarized_text")
    parser.add_argument(
        "--use-summarized",
        action="store_true",
        help="Use summarized_text when available (shorter Kontext prompts).",
    )
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, only keep the first N valid pairs (after missing-file filter).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing out edit/reference dirs before writing.",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_absolute():
        jsonl_path = data_root / jsonl_path
    out_root = args.out_root or (data_root / "simpletuner_pairs")

    if not jsonl_path.is_file():
        print(f"[error] jsonl not found: {jsonl_path}", file=sys.stderr)
        return 1

    pairs: list[Tuple[Path, Path, str]] = []
    skipped = 0
    for row in _iter_jsonl(jsonl_path):
        resolved = _resolve_pair(
            row,
            data_root=data_root,
            edited_images_dir=args.edited_images_dir,
            source_column=args.source_column,
            target_column=args.target_column,
            prompt_column=args.prompt_column,
            summarized_prompt_column=args.summarized_prompt_column,
            use_summarized=args.use_summarized,
        )
        if resolved is None:
            skipped += 1
            continue
        pairs.append(resolved)
        if args.max_samples > 0 and len(pairs) >= args.max_samples:
            break
        if len(pairs) % 20000 == 0 and len(pairs):
            print(f"[scan] valid={len(pairs)} skipped={skipped}", flush=True)

    if not pairs:
        print("[error] no valid pairs found", file=sys.stderr)
        return 1

    val_size = min(max(args.val_size, 0), len(pairs))
    if val_size > 0 and len(pairs) > val_size:
        train_rows = pairs[:-val_size]
        val_rows = pairs[-val_size:]
    else:
        train_rows = pairs
        val_rows = []

    splits = {
        "train": train_rows,
        "val": val_rows,
    }

    for split, rows in splits.items():
        edit_dir = out_root / split / "edit"
        ref_dir = out_root / split / "reference"
        if args.force and edit_dir.exists():
            for p in edit_dir.iterdir():
                p.unlink()
            for p in ref_dir.iterdir() if ref_dir.exists() else []:
                p.unlink()
        n = materialize_split(rows, edit_dir, ref_dir, start_index=0)
        print(f"[write] {split}: {n} pairs -> {edit_dir} + {ref_dir}")

    meta = {
        "data_root": str(data_root),
        "jsonl": str(jsonl_path),
        "out_root": str(out_root),
        "valid_pairs": len(pairs),
        "skipped": skipped,
        "train": len(train_rows),
        "val": len(val_rows),
        "use_summarized": bool(args.use_summarized),
        "max_samples": args.max_samples,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(
        "[note] Images are hardlinked/symlinked; captions are small .txt sidecars.\n"
        "       SimpleTuner VAE/text caches for the full ~257k set need a lot of disk;\n"
        "       use --max-samples for a smoke run if free space is tight."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
