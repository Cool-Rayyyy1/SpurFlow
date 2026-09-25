#!/usr/bin/env python3
"""Build metadata.jsonl for /mnt/afs_gaochengmin/data/oss_edit.

Prompt is always taken from the sidecar json:
  Edit_Instruction, else gemma_instruction_separate.

By default scans each task dir for complete
{uuid}.src.jpg / .tgt.jpg / .json triples via os.scandir (no pathlib glob).
Pass --from-existing-jsonl to reuse an existing metadata.jsonl as the index
and only refresh the prompt fields.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INSTRUCTION_KEYS = (
    "Edit_Instruction",
    "gemma_instruction_separate",
)
SKIP_JSON = {"sample_stats.json"}


def pick_instruction(meta: dict) -> str:
    for key in INSTRUCTION_KEYS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def scandir_stems(task_dir: Path) -> set[str]:
    srcs: set[str] = set()
    tgts: set[str] = set()
    jsons: set[str] = set()
    with os.scandir(task_dir) as it:
        for ent in it:
            name = ent.name
            if name.endswith(".src.jpg"):
                srcs.add(name[: -len(".src.jpg")])
            elif name.endswith(".tgt.jpg"):
                tgts.add(name[: -len(".tgt.jpg")])
            elif name.endswith(".json") and name not in SKIP_JSON:
                jsons.add(name[: -len(".json")])
    return srcs & tgts & jsons


def instruction_source(meta: dict) -> str:
    for key in INSTRUCTION_KEYS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return key
    return ""


def row_from_meta(task_dir: Path, task: str, stem: str, meta: dict) -> dict | None:
    instruction = pick_instruction(meta)
    if not instruction:
        return None
    return {
        "uuid": stem,
        "task": task,
        "input_path": str((task_dir / f"{stem}.src.jpg").resolve()),
        "output_path": str((task_dir / f"{stem}.tgt.jpg").resolve()),
        "instruction": instruction,
        "prompt_source": instruction_source(meta),
        "edit_type": meta.get("Edit_Type") or meta.get("edit_type") or "",
    }


def collect_task(task_dir: Path) -> list[dict]:
    task = task_dir.name
    stems = scandir_stems(task_dir)
    rows: dict[str, dict] = {}

    pairs_path = task_dir / "pairs.jsonl"
    n_from_pairs = 0
    pair_stems: set[str] = set()
    if pairs_path.is_file():
        with pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                stem = rec.get("uuid") or ""
                if stem in stems:
                    pair_stems.add(stem)
        n_from_pairs = len(pair_stems)

    # Always take prompt from sidecar json: Edit_Instruction, else gemma_instruction_separate.
    for stem in sorted(stems):
        js = task_dir / f"{stem}.json"
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row = row_from_meta(task_dir, task, stem, meta)
        if row is not None:
            rows[stem] = row

    print(
        f"{task}: {len(rows)} (listed_in_pairs.jsonl={n_from_pairs})",
        flush=True,
    )
    return [rows[k] for k in sorted(rows)]


def collect_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows.extend(collect_task(task_dir))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/afs_gaochengmin/data/oss_edit"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: {root}/metadata.jsonl",
    )
    parser.add_argument(
        "--from-existing-jsonl",
        type=Path,
        default=None,
        help="Reuse this jsonl as the pair index; only refresh prompts from sidecar json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Thread workers when refreshing from an existing jsonl.",
    )
    return parser.parse_args()


def sidecar_json_path(rec: dict) -> Path:
    inp = rec.get("input_path") or ""
    if inp.endswith(".src.jpg"):
        return Path(inp[: -len(".src.jpg")] + ".json")
    uuid = rec.get("uuid") or ""
    task = rec.get("task") or ""
    root = rec.get("data_root")
    if uuid and task:
        base = Path(root) if root else Path("/mnt/afs_gaochengmin/data/oss_edit")
        return base / task / f"{uuid}.json"
    raise ValueError(f"cannot locate sidecar json for {rec!r}")


def refresh_one(rec: dict) -> dict | None:
    js = sidecar_json_path(rec)
    try:
        meta = json.loads(js.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    instruction = pick_instruction(meta)
    if not instruction:
        return None
    rec = dict(rec)
    rec["instruction"] = instruction
    rec["prompt_source"] = instruction_source(meta)
    if not rec.get("edit_type"):
        rec["edit_type"] = meta.get("Edit_Type") or meta.get("edit_type") or ""
    return rec


def refresh_from_jsonl(src: Path, workers: int) -> list[dict]:
    with src.open("r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    rows: list[dict | None] = [None] * len(recs)
    dropped = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(refresh_one, rec): i for i, rec in enumerate(recs)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            row = fut.result()
            if row is None:
                dropped += 1
            else:
                rows[i] = row
            done += 1
            if done % 10000 == 0:
                print(f"refreshed {done}/{len(recs)}", flush=True)
    kept = [r for r in rows if r is not None]
    from_edit = sum(1 for r in kept if r.get("prompt_source") == "Edit_Instruction")
    from_gemma = sum(1 for r in kept if r.get("prompt_source") == "gemma_instruction_separate")
    print(
        f"kept {len(kept)} dropped {dropped} "
        f"(Edit_Instruction={from_edit} gemma_instruction_separate={from_gemma})",
        flush=True,
    )
    return kept


def main() -> None:
    args = parse_args()
    out = args.out or (args.root / "metadata.jsonl")
    if args.from_existing_jsonl:
        rows = refresh_from_jsonl(args.from_existing_jsonl, args.workers)
    else:
        rows = collect_rows(args.root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(f"wrote {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
