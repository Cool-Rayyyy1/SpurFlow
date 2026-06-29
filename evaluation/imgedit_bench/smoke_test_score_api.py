#!/usr/bin/env python3
"""Smoke test: verify ImgEdit-Bench GPT scoring API (no GPU needed)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
EDITFLOW_ROOT = EVAL_ROOT.parents[1]

DEFAULT_BENCH_ROOT = Path(
    os.environ.get(
        "IMGEDIT_BENCH_ROOT",
        "/mnt/afs_zhangyunzhe/dataset/imgedit/benchmark/Benchmark",
    )
)


def load_openai_env() -> None:
    env_file = EVAL_ROOT / "openai.env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def image_to_base64(path: Path, max_side: int = 512) -> str:
    from io import BytesIO
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_gpt(
    original_path: Path,
    result_path: Path,
    edit_prompt: str,
    edit_type: str,
    prompts: dict,
    timeout: int = 120,
) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (check evaluation/imgedit_bench/openai.env)")

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.nuwaapi.com/v1"),
        timeout=timeout,
    )
    template = prompts[edit_type]
    full_prompt = template.replace("<edit_prompt>", edit_prompt)

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        stream=False,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(original_path)}"}},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(result_path)}"}},
            ],
        }],
    )
    return response.choices[0].message.content


def resolve_result_image(bench_root: Path, item: dict, key: str, result_root: Path | None) -> Path:
    if result_root is not None:
        cand = result_root / "basic" / f"{key}.png"
        if cand.is_file():
            return cand
    # No generated image yet — use source as placeholder (API connectivity test only).
    src = bench_root / "singleturn" / item["id"]
    if not src.is_file():
        raise FileNotFoundError(f"Source image not found: {src}")
    return src


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test ImgEdit GPT scoring API.")
    p.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT)
    p.add_argument("--result_root", type=Path, default=EVAL_ROOT / "outputs" / "teacher",
                   help="Optional folder with basic/*.png; falls back to source image.")
    p.add_argument("--sample_key", type=str, default="1082")
    p.add_argument("--timeout", type=int, default=120, help="API timeout in seconds")
    args = p.parse_args()

    load_openai_env()

    basic_edit = json.loads(
        (EVAL_ROOT / "annotations" / "basic_edit.json").read_text(encoding="utf-8"))
    prompts = json.loads(
        (EVAL_ROOT / "scoring" / "basic" / "prompts.json").read_text(encoding="utf-8"))

    if args.sample_key not in basic_edit:
        raise KeyError(f"Sample key {args.sample_key!r} not in basic_edit.json")

    item = basic_edit[args.sample_key]
    origin = args.bench_root / "singleturn" / item["id"]
    result = resolve_result_image(args.bench_root, item, args.sample_key, args.result_root)

    print(f"[smoke] API base: {os.environ.get('OPENAI_BASE_URL', '(default)')}")
    print(f"[smoke] sample key: {args.sample_key}")
    print(f"[smoke] edit_type:   {item['edit_type']}")
    print(f"[smoke] prompt:      {item['prompt'][:80]}...")
    print(f"[smoke] origin:      {origin}")
    print(f"[smoke] result:      {result}" + (" (source placeholder)" if result == origin else ""))
    print("[smoke] calling gpt-4o-2024-11-20 ...")

    text = call_gpt(origin, result, item["prompt"], item["edit_type"], prompts, args.timeout)
    print("\n[smoke] API call succeeded.")
    print("----- GPT response (first 500 chars) -----")
    print(text[:500])
    if len(text) > 500:
        print("...")
    print("------------------------------------------")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[smoke] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
