#!/usr/bin/env python3
"""Build comparison assets: Src | Teacher? | Student? grid + individual files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from run_spurflow_imgedit_infer import load_tasks, resolve_source_path


PANEL_HEIGHT = 512
LABEL_HEIGHT = 28
PROMPT_BAR_HEIGHT = 72
BG_COLOR = (245, 245, 245)
LABEL_BG = (30, 30, 30)
LABEL_FG = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ImgEdit comparison grids: Src + Teacher (optional) + Student."
    )
    p.add_argument("--bench_root", type=Path, required=True)
    p.add_argument("--annotations_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True, help="Where to write comparisons/")
    p.add_argument(
        "--teacher_output",
        type=Path,
        default=None,
        help="Teacher generation folder (basic/, uge/). Omit if teacher was not run.",
    )
    p.add_argument(
        "--student_output",
        type=Path,
        default=None,
        help="Student generation folder (basic/, uge/).",
    )
    p.add_argument(
        "--require_student",
        action="store_true",
        help="Skip samples without student output (use when student inference was run).",
    )
    p.add_argument("--suite", choices=("basic", "uge", "all"), default="all")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def output_png(output_root: Optional[Path], task_key: str) -> Optional[Path]:
    if output_root is None:
        return None
    suite_name, sample_key = task_key.split(":", 1)
    if suite_name == "multiturn":
        return None
    path = output_root / suite_name / f"{sample_key}.png"
    return path if path.is_file() else None


def wrap_prompt(prompt: str, width: int = 110) -> List[str]:
    lines = textwrap.wrap(prompt.strip(), width=width) or [""]
    return lines[:3]


def resize_to_height(image: Image.Image, target_h: int) -> Image.Image:
    width, height = image.size
    if height == target_h:
        return image
    new_w = max(1, int(width * target_h / height))
    return image.resize((new_w, target_h), Image.Resampling.LANCZOS)


def build_panel(
    columns: List[Tuple[str, Image.Image]],
    prompt: str,
    panel_height: int = PANEL_HEIGHT,
) -> Image.Image:
    resized = [(label, resize_to_height(img, panel_height)) for label, img in columns]
    total_width = sum(img.size[0] for _, img in resized)
    prompt_lines = wrap_prompt(prompt)
    prompt_font = load_font(18)
    label_font = load_font(16)

    canvas_h = panel_height + LABEL_HEIGHT + PROMPT_BAR_HEIGHT
    canvas = Image.new("RGB", (total_width, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    x = 0
    for label, img in resized:
        canvas.paste(img, (x, 0))
        label_box = (x, panel_height, x + img.size[0], panel_height + LABEL_HEIGHT)
        draw.rectangle(label_box, fill=LABEL_BG)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (x + (img.size[0] - text_w) // 2, panel_height + (LABEL_HEIGHT - text_h) // 2),
            label,
            fill=LABEL_FG,
            font=label_font,
        )
        x += img.size[0]

    prompt_y = panel_height + LABEL_HEIGHT + 8
    for line in prompt_lines:
        draw.text((12, prompt_y), line, fill=(20, 20, 20), font=prompt_font)
        prompt_y += 22

    return canvas


def _remove_path_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file() or path.is_symlink():
        path.unlink()


def save_image_atomic(image: Image.Image, path: Path, **save_kw) -> None:
    """Write image via temp file + rename to avoid FileExistsError on some FS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove_path_if_exists(path)
    fd, tmp_name = tempfile.mkstemp(suffix=path.suffix, dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        image.save(tmp_path, **save_kw)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def save_src_copy(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = src_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        _remove_path_if_exists(dst_path)
        shutil.copy2(src_path, dst_path)
        return
    save_image_atomic(Image.open(src_path).convert("RGB"), dst_path, quality=95)


def save_png_copy(src_path: Path, dst_path: Path) -> None:
    save_image_atomic(Image.open(src_path).convert("RGB"), dst_path)


def comparison_outputs_complete(
    grid_path: Path,
    src_out: Path,
    require_student: bool,
    student_out: Path,
) -> bool:
    if not grid_path.is_file() or not src_out.is_file():
        return False
    if require_student and not student_out.is_file():
        return False
    return True


def build_comparisons(args: argparse.Namespace) -> Dict[str, Dict]:
    tasks = load_tasks(args.suite, args.annotations_dir, args.bench_root)
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict] = {}

    for task_key, item in tqdm(tasks, desc="Building comparisons"):
        suite_name, sample_key = task_key.split(":", 1)
        if suite_name == "multiturn":
            continue

        grid_path = args.output_dir / suite_name / f"{sample_key}.png"
        src_out = args.output_dir / suite_name / f"{sample_key}_src.jpg"
        teacher_out = args.output_dir / suite_name / f"{sample_key}_teacher.png"
        student_out = args.output_dir / suite_name / f"{sample_key}_student.png"
        prompt_out = args.output_dir / suite_name / f"{sample_key}_prompt.txt"

        if args.skip_existing and comparison_outputs_complete(
            grid_path, src_out, args.require_student, student_out
        ):
            continue

        src_path = resolve_source_path(args.bench_root, item, task_key)
        if not src_path.is_file():
            print(f"[skip] missing src for {task_key}: {src_path}")
            continue

        teacher_path = output_png(args.teacher_output, task_key)
        student_path = output_png(args.student_output, task_key)

        if args.require_student and student_path is None:
            print(f"[skip] missing student output for {task_key}")
            continue
        if teacher_path is None and student_path is None:
            print(f"[skip] no teacher/student outputs for {task_key}")
            continue

        prompt = item.get("prompt") or ""

        save_src_copy(src_path, src_out)
        prompt_out.parent.mkdir(parents=True, exist_ok=True)
        prompt_out.write_text(prompt + "\n", encoding="utf-8")

        columns: List[Tuple[str, Image.Image]] = [
            ("Src", Image.open(src_path).convert("RGB"))
        ]
        if teacher_path is not None:
            save_png_copy(teacher_path, teacher_out)
            columns.append(("Teacher", Image.open(teacher_path).convert("RGB")))
        elif teacher_out.is_file():
            teacher_out.unlink()

        if student_path is not None:
            save_png_copy(student_path, student_out)
            columns.append(("Student", Image.open(student_path).convert("RGB")))
        elif student_out.is_file():
            student_out.unlink()

        panel = build_panel(columns, prompt=prompt)
        save_image_atomic(panel, grid_path)

        manifest[task_key] = {
            "suite": suite_name,
            "key": sample_key,
            "prompt": prompt,
            "src": str(src_out),
            "src_dataset": str(src_path),
            "teacher": str(teacher_out) if teacher_path is not None else None,
            "teacher_output": str(teacher_path) if teacher_path else None,
            "student": str(student_out) if student_path is not None else None,
            "student_output": str(student_path) if student_path else None,
            "grid": str(grid_path),
            "prompt_file": str(prompt_out),
        }

    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing.update(manifest)
        manifest = existing
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(manifest)} comparison records to {args.output_dir}")
    return manifest


def main() -> None:
    args = parse_args()
    build_comparisons(args)


if __name__ == "__main__":
    main()
