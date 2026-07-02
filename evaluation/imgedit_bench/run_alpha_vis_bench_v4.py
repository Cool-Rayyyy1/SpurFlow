#!/usr/bin/env python3
"""v4: per sample save src / edit / heatmap / alpha<0.1 overlay (4 images)."""

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

EVAL_ROOT = Path(__file__).resolve().parent
EDITFLOW_ROOT = EVAL_ROOT.parents[1]
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))

from alpha_vis import (  # noqa: E402
    capture_student_alphas,
    render_alpha_heatmap,
    render_src_alpha_low_overlay,
)
from alpha_model_utils import build_alpha_vis_model  # noqa: E402
from run_editflow_imgedit_infer import (  # noqa: E402
    DEFAULT_BENCH_ROOT,
    expected_outputs,
    infer_nfe,
    load_tasks,
    merge_manifest_parts,
    pil_to_tensor,
    preprocess_image_for_student,
    resolve_gpu_ids,
    resolve_source_path,
    split_tasks_round_robin,
    tensor_to_pil,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpha v4: src/edit/heatmap/alpha-low per sample.")
    p.add_argument("--suite", choices=("basic", "uge", "basic_uge", "all"), default="basic_uge")
    p.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT)
    p.add_argument("--annotations_dir", type=Path, default=EVAL_ROOT / "annotations")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_gpus", type=int, default=int(os.environ.get("NUM_GPUS", "2")))
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument(
        "--alpha_threshold",
        type=float,
        default=float(os.environ.get("ALPHA_THRESHOLD", "0.1")),
    )
    return p.parse_args()


def load_suite_tasks(suite: str, annotations_dir: Path, bench_root: Path):
    if suite == "basic_uge":
        tasks = load_tasks("basic", annotations_dir, bench_root)
        tasks.extend(load_tasks("uge", annotations_dir, bench_root))
        return tasks
    return load_tasks(suite, annotations_dir, bench_root)


def v4_output_paths(output_dir: Path, rel_score_path: Path, threshold: float) -> Dict[str, Path]:
    stem = rel_score_path.stem
    parent = rel_score_path.parent
    base = output_dir / parent
    return {
        "src": base / f"{stem}_src.png",
        "edit": base / f"{stem}_edit.png",
        "heatmap": base / f"{stem}_heatmap.png",
        "alpha_low": base / f"{stem}_overlay.png",
    }


def process_tasks(
    model,
    tasks: List[Tuple[str, Dict]],
    bench_root: Path,
    output_dir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    skip_existing: bool,
    device: str,
    desc: str,
    alpha_threshold: float,
) -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    step_label = f"step {num_steps}/{num_steps}"

    for task_key, item in tqdm(tasks, desc=desc):
        suite_name, sample_key = task_key.split(":", 1)
        if suite_name == "multiturn":
            continue

        rel_paths = expected_outputs(task_key, item)
        paths = v4_output_paths(output_dir, rel_paths[0], alpha_threshold)
        if skip_existing and all(p.is_file() for p in paths.values()):
            continue

        src_path = resolve_source_path(bench_root, item, task_key)
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_path}")

        task_seed = seed + int(sample_key.split(":")[-1]) if sample_key.split(":")[-1].isdigit() else seed
        image = Image.open(src_path).convert("RGB")

        src_pil, alphas, edit_pil = capture_student_alphas(
            model,
            image,
            item["prompt"],
            num_steps,
            guidance_scale,
            task_seed,
            device,
            preprocess_image_for_student,
            pil_to_tensor,
            return_edited=True,
            tensor_to_pil_fn=tensor_to_pil,
        )
        if not alphas:
            raise RuntimeError(f"No alpha captured for {task_key}")
        alpha_final = alphas[-1]
        img_w, img_h = src_pil.size

        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)

        src_pil.save(paths["src"])
        edit_pil.save(paths["edit"])
        render_alpha_heatmap(img_w, img_h, alpha_final, step_label=step_label).save(paths["heatmap"])
        render_src_alpha_low_overlay(
            src_pil,
            alpha_final,
            threshold=alpha_threshold,
            step_label=step_label,
        ).save(paths["alpha_low"])

        manifest[task_key] = {
            "suite": suite_name,
            "key": sample_key,
            "source": str(src_path),
            "prompt": item.get("prompt"),
            "alpha_step_used": num_steps,
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    return manifest


def _worker(gpu_id: int, tasks: List[Tuple[str, Dict]], worker_cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model = build_alpha_vis_model(
        Path(worker_cfg["config"]),
        Path(worker_cfg["ckpt"]),
        "cuda",
    )
    manifest = process_tasks(
        model,
        tasks,
        Path(worker_cfg["bench_root"]),
        Path(worker_cfg["output_dir"]),
        worker_cfg["num_steps"],
        worker_cfg["guidance_scale"],
        worker_cfg["seed"],
        worker_cfg["skip_existing"],
        "cuda",
        desc=f"GPU {gpu_id}",
        alpha_threshold=worker_cfg["alpha_threshold"],
    )
    part = Path(worker_cfg["output_dir"]) / f"manifest.gpu{gpu_id}.json"
    part.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)

    num_steps = args.num_inference_steps or infer_nfe(args.run_name, None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_suite_tasks(args.suite, args.annotations_dir, args.bench_root)
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]

    gpu_ids = resolve_gpu_ids(args.num_gpus)
    buckets = split_tasks_round_robin(tasks, len(gpu_ids))

    worker_cfg = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "bench_root": str(args.bench_root),
        "output_dir": str(args.output_dir),
        "num_steps": num_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "skip_existing": args.skip_existing,
        "alpha_threshold": args.alpha_threshold,
    }

    if len(buckets) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        model = build_alpha_vis_model(args.config, args.ckpt, args.device)
        manifest = process_tasks(
            model,
            buckets[0],
            args.bench_root,
            args.output_dir,
            num_steps,
            args.guidance_scale,
            args.seed,
            args.skip_existing,
            args.device,
            desc="alpha v4",
            alpha_threshold=args.alpha_threshold,
        )
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for gpu_id, bucket in zip(gpu_ids, buckets):
            if not bucket:
                continue
            p = ctx.Process(target=_worker, args=(gpu_id, bucket, worker_cfg))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker failed with exit code {p.exitcode}")
        manifest = merge_manifest_parts(args.output_dir, {})

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved v4 packs for {len(manifest)} tasks -> {args.output_dir}")
    print("Per sample: *_src.png  *_edit.png  *_heatmap.png  *_overlay.png")


if __name__ == "__main__":
    main()
