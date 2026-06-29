#!/usr/bin/env python3
"""Run ImgEdit-Bench GPT-4o scoring (Basic + UGE) for teacher or student outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

EVAL_ROOT = Path(__file__).resolve().parent


def scoring_model_name() -> str:
    return os.environ.get("OPENAI_SCORING_MODEL", "gpt-4o").strip() or "gpt-4o"


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ImgEdit-Bench GPT scoring.")
    p.add_argument("--role", choices=("teacher", "student", "klein"), default="student")
    p.add_argument("--model_output", type=Path, required=True,
                   help="Folder with basic/ and uge/ generated images.")
    p.add_argument("--scores_dir", type=Path, default=None,
                   help="Where to store JSON score files (default: <model_output>/scores).")
    p.add_argument("--scores_txt", type=Path, default=None,
                   help="Human-readable summary txt (default: <model_output>/scores.txt).")
    p.add_argument("--bench_root", type=Path, required=True)
    p.add_argument("--annotations_dir", type=Path, default=EVAL_ROOT / "annotations")
    p.add_argument("--num_processes", type=int, default=64)
    p.add_argument("--force", action="store_true")
    p.add_argument("--print_only", action="store_true",
                   help="Only print existing scores.txt, do not run GPT.")
    return p.parse_args()


def has_api_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return bool(key) and key not in {"YOUR_API_KEY_HERE", "api-key", "sk-..."}


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def basic_overall(average_score_json: Path) -> Optional[float]:
    if not average_score_json.is_file():
        return None
    data = json.loads(average_score_json.read_text(encoding="utf-8"))
    if not data:
        return None
    return round(sum(data.values()) / len(data), 2)


def uge_overall(uge_avg_json: Path) -> Optional[float]:
    if not uge_avg_json.is_file():
        return None
    data = json.loads(uge_avg_json.read_text(encoding="utf-8"))
    return data.get("__final_average__")


def write_scores_txt(
    scores_txt: Path,
    role: str,
    basic_avg: Path,
    basic_type: Path,
    uge_avg: Path,
) -> None:
    lines = [
        f"ImgEdit-Bench GPT Scores ({role})",
        f"Scoring model: {scoring_model_name()}",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    basic_all = basic_overall(basic_avg)
    if basic_all is not None:
        lines.append(f"Basic overall: {basic_all}")

    if basic_type.is_file():
        types: Dict[str, float] = json.loads(basic_type.read_text(encoding="utf-8"))
        if types:
            lines.append("Basic by edit_type:")
            for name, score in sorted(types.items()):
                lines.append(f"  {name}: {score}")

    uge_all = uge_overall(uge_avg)
    if uge_all is not None:
        lines.append(f"UGE overall: {uge_all}")

    scores_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[score] Summary -> {scores_txt}")


def print_scores_txt(scores_txt: Path, label: str) -> None:
    print(f"\n========== {label} ==========")
    if scores_txt.is_file():
        print(scores_txt.read_text(encoding="utf-8"), end="")
    else:
        print(f"(no scores yet: {scores_txt})")


def run_scoring(args: argparse.Namespace) -> bool:
    """Run GPT scoring if needed. Returns True if scores txt exists/was written."""
    scores_dir = args.scores_dir or (args.model_output / "scores")
    scores_txt = args.scores_txt or (args.model_output / "scores.txt")
    scores_dir.mkdir(parents=True, exist_ok=True)

    if args.print_only:
        print_scores_txt(scores_txt, args.role)
        return scores_txt.is_file()

    if not has_api_key():
        print(
            "[skip] OPENAI_API_KEY not set. GPT scoring skipped.\n"
            "       Configure evaluation/imgedit_bench/openai.env then re-run."
        )
        return scores_txt.is_file()

    basic_result = scores_dir / "basic_gpt_result.json"
    basic_avg = scores_dir / "basic_average_score.json"
    basic_type = scores_dir / "basic_typescore.json"
    uge_result = scores_dir / "uge_gpt_result.json"
    uge_avg = scores_dir / "uge_average_score.json"

    basic_dir = args.model_output / "basic"
    uge_dir = args.model_output / "uge"
    origin_basic = args.bench_root / "singleturn"
    origin_uge = args.bench_root / "hard"

    scored = False

    if basic_dir.is_dir() and (args.force or not basic_type.is_file()):
        basic_cmd = [
            sys.executable,
            str(EVAL_ROOT / "scoring" / "basic" / "basic_bench.py"),
            "--result_img_folder", str(basic_dir),
            "--edit_json", str(args.annotations_dir / "basic_edit.json"),
            "--origin_img_root", str(origin_basic),
            "--prompts_json", str(EVAL_ROOT / "scoring" / "basic" / "prompts.json"),
            "--num_processes", str(args.num_processes),
            "--output_json", str(basic_result),
        ]
        if not args.force:
            basic_cmd.append("--skip_existing")
        run_cmd(basic_cmd)
        run_cmd([
            sys.executable,
            str(EVAL_ROOT / "scoring" / "basic" / "step1_get_avgscore.py"),
            "--result_json", str(basic_result),
            "--average_score_json", str(basic_avg),
        ])
        run_cmd([
            sys.executable,
            str(EVAL_ROOT / "scoring" / "basic" / "step2_typescore.py"),
            "--average_score_json", str(basic_avg),
            "--typescore_json", str(basic_type),
            "--basic_edit", str(args.annotations_dir / "basic_edit.json"),
        ])
        scored = True
    elif basic_dir.is_dir():
        print(f"[skip] Basic scores already exist: {basic_type}")

    if uge_dir.is_dir() and (args.force or not uge_avg.is_file()):
        uge_cmd = [
            sys.executable,
            str(EVAL_ROOT / "scoring" / "uge" / "UGE_bench.py"),
            "--result_img_folder", str(uge_dir),
            "--edit_json", str(args.annotations_dir / "UGE_edit.json"),
            "--origin_img_root", str(origin_uge),
            "--num_processes", str(args.num_processes),
            "--output_json", str(uge_result),
        ]
        if not args.force:
            uge_cmd.append("--skip_existing")
        run_cmd(uge_cmd)
        run_cmd([
            sys.executable,
            str(EVAL_ROOT / "scoring" / "uge" / "get_average_score.py"),
            "--result_json", str(uge_result),
            "--average_score_json", str(uge_avg),
        ])
        scored = True
    elif uge_dir.is_dir():
        print(f"[skip] UGE scores already exist: {uge_avg}")

    if scored or basic_type.is_file() or uge_avg.is_file():
        write_scores_txt(scores_txt, args.role, basic_avg, basic_type, uge_avg)

    summary = {
        "role": args.role,
        "scoring_model": scoring_model_name(),
        "model_output": str(args.model_output),
        "scores_dir": str(scores_dir),
        "scores_txt": str(scores_txt),
        "basic_overall": basic_overall(basic_avg),
        "uge_overall": uge_overall(uge_avg),
    }
    (scores_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    return scores_txt.is_file()


def main() -> None:
    load_openai_env()
    args = parse_args()
    run_scoring(args)


if __name__ == "__main__":
    main()
