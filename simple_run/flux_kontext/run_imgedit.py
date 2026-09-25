#!/usr/bin/env python3
"""Official FLUX.1-Kontext-dev baseline on ImgEdit-Bench.

No student/teacher split. Writes directly under the run folder:

  {output_dir}/basic/{Category}/{key}/{src.png,pred.png,prompt.txt}
  {output_dir}/basic/{key}.png
  {output_dir}/uge/{key}.png   (if --suite all/uge)

Resize: FLUX.1-Kontext preferred-resolution buckets, bicubic, no crop.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
EDITFLOW_ROOT = HERE.parents[1]
IMGEDIT_ROOT = EDITFLOW_ROOT / "evaluation" / "imgedit_bench"
for path in (str(HERE), str(IMGEDIT_ROOT), str(EDITFLOW_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from kontext_common import (  # noqa: E402
    DEFAULT_GUIDANCE,
    DEFAULT_MODEL,
    DEFAULT_STEPS,
    build_pipeline,
    configure_hf_env,
    merge_manifest_parts,
    resolve_gpu_ids,
    run_one,
    split_tasks_round_robin,
)
from run_editflow_imgedit_infer import (  # noqa: E402
    BASIC_CATEGORY_DIRS,
    basic_case_dir,
    case_bundle_ready,
    expected_outputs,
    load_basic_tasks,
    load_uge_tasks,
    resolve_source_path,
    try_open_rgb_image,
    write_basic_case_bundle,
)

DEFAULT_BENCH_ROOT = Path(
    os.environ.get(
        "IMGEDIT_BENCH_ROOT",
        "/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark",
    )
)
DEFAULT_RUN_TAG = os.environ.get("RUN_TAG", "flux_kontext_official")
DEFAULT_OUTPUT = IMGEDIT_ROOT / "outputs" / "runs" / DEFAULT_RUN_TAG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Official FLUX.1-Kontext-dev ImgEdit-Bench inference."
    )
    p.add_argument("--suite", choices=("basic", "uge", "all"), default="all")
    p.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT)
    p.add_argument(
        "--annotations_dir",
        type=Path,
        default=IMGEDIT_ROOT / "annotations",
    )
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--num_inference_steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument(
        "--num_gpus",
        type=int,
        default=int(os.environ.get("NUM_GPUS", "8")),
    )
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def process_tasks(
    pipe,
    tasks: List[Tuple[str, Dict]],
    bench_root: Path,
    output_dir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    skip_existing: bool,
    desc: str,
) -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    for task_key, item in tqdm(tasks, desc=desc):
        suite_name, sample_key = task_key.split(":", 1)
        rel_paths = expected_outputs(task_key, item)
        abs_paths = [output_dir / rel for rel in rel_paths]
        case_dir = (
            basic_case_dir(output_dir, item, sample_key)
            if suite_name == "basic"
            else None
        )
        score_ready = all(try_open_rgb_image(p) is not None for p in abs_paths)
        case_ready = case_dir is None or case_bundle_ready(case_dir)
        if skip_existing and score_ready and case_ready:
            continue

        src_path = resolve_source_path(bench_root, item, task_key)
        image = try_open_rgb_image(src_path)
        if image is None:
            print(f"[skip] unreadable source: {src_path}", flush=True)
            manifest[task_key] = {
                "suite": suite_name,
                "key": sample_key,
                "source": str(src_path),
                "error": f"unreadable source: {src_path}",
            }
            continue

        task_seed = (
            seed + int(sample_key.split(":")[-1])
            if sample_key.split(":")[-1].isdigit()
            else seed
        )
        need_infer = (not abs_paths[0].is_file()) or (
            case_dir is not None and not case_bundle_ready(case_dir)
        )
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            print(f"[skip] missing prompt: {task_key}", flush=True)
            continue

        if need_infer:
            result = run_one(
                pipe,
                image=image,
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=task_seed,
            )
            abs_paths[0].parent.mkdir(parents=True, exist_ok=True)
            result.save(abs_paths[0])
        else:
            result = Image.open(abs_paths[0]).convert("RGB")

        if case_dir is not None:
            write_basic_case_bundle(
                case_dir, image, result, item.get("prompt", "")
            )

        manifest[task_key] = {
            "suite": suite_name,
            "key": sample_key,
            "source": str(src_path),
            "prompt": item.get("prompt"),
            "outputs": [str(p) for p in abs_paths],
            "case_dir": str(case_dir) if case_dir is not None else None,
            "edit_type": item.get("edit_type"),
        }
    return manifest


def _infer_worker(gpu_id: int, tasks: List[Tuple[str, Dict]], cfg: dict) -> None:
    configure_hf_env()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pipe = build_pipeline(cfg["model_path"], "cuda", cfg["cpu_offload"])
    manifest = process_tasks(
        pipe,
        tasks,
        Path(cfg["bench_root"]),
        Path(cfg["output_dir"]),
        cfg["num_steps"],
        cfg["guidance_scale"],
        cfg["seed"],
        cfg["skip_existing"],
        desc=f"GPU {gpu_id}",
    )
    part = Path(cfg["output_dir"]) / f"manifest.gpu{gpu_id}.json"
    part.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    del pipe


def main() -> None:
    configure_hf_env()
    args = parse_args()
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.bench_root.is_dir():
        raise FileNotFoundError(f"Bench root not found: {args.bench_root}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Official scoring only uses basic + UGE. Do not pull multiturn
    # (those items have turn1/turn2, no "prompt" key).
    tasks = []
    if args.suite in ("basic", "all"):
        tasks.extend(load_basic_tasks(args.annotations_dir))
    if args.suite in ("uge", "all"):
        tasks.extend(load_uge_tasks(args.annotations_dir))
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]
    print(
        f"[imgedit] {len(tasks)} samples  suite={args.suite}  "
        f"steps={args.num_inference_steps}  guidance={args.guidance_scale}  "
        f"resize=kontext-buckets  model={args.model_path}"
    )
    print(f"[imgedit] output -> {output_dir}")
    print(f"[imgedit] basic categories: {', '.join(BASIC_CATEGORY_DIRS)}")

    gpu_ids = resolve_gpu_ids(args.num_gpus)
    worker_cfg = {
        "model_path": args.model_path,
        "bench_root": str(args.bench_root),
        "output_dir": str(output_dir),
        "num_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "skip_existing": args.skip_existing,
        "cpu_offload": args.cpu_offload,
    }

    if len(gpu_ids) > 1 and not args.cpu_offload:
        print(f"[infer] parallel on GPUs: {gpu_ids}")
        buckets = split_tasks_round_robin(tasks, len(gpu_ids))
        ctx = mp.get_context("spawn")
        procs = []
        for gpu_id, bucket in zip(gpu_ids, buckets):
            p = ctx.Process(target=_infer_worker, args=(gpu_id, bucket, worker_cfg))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker failed with exit code {p.exitcode}")
        merge_manifest_parts(output_dir)
    else:
        if len(gpu_ids) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        pipe = build_pipeline(args.model_path, "cuda", args.cpu_offload)
        manifest = process_tasks(
            pipe,
            tasks,
            args.bench_root,
            output_dir,
            args.num_inference_steps,
            args.guidance_scale,
            args.seed,
            args.skip_existing,
            desc="imgedit",
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"[imgedit] done -> {output_dir}")


if __name__ == "__main__":
    main()
