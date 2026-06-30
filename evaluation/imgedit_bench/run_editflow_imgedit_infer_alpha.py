#!/usr/bin/env python3
"""ImgEdit-Bench inference for EditFlow alpha student models.

Same benchmark layout as run_editflow_imgedit_infer.py for scoring images
(basic/, uge/, multiturn/). Additionally writes alpha patch visualizations under
alpha_vis/ — these are NOT used by GPT scoring.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
EDITFLOW_ROOT = EVAL_ROOT.parents[1]
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))

from alpha_vis import capture_student_alphas, render_alpha_grid_blank  # noqa: E402
from run_editflow_imgedit_infer import (  # noqa: E402
    build_klein_pipeline,
    build_student_model,
    build_teacher_pipeline,
    expected_outputs,
    load_tasks,
    merge_manifest_parts,
    multiturn_prompts,
    parse_args,
    preprocess_image_for_student,
    pil_to_tensor,
    resolve_gpu_ids,
    resolve_inference_settings,
    resolve_output_dir,
    resolve_source_path,
    run_multiturn_chain,
    run_one,
    split_tasks_round_robin,
    suite_complete,
    tensor_to_pil,
)

ALPHA_VIS_SUBDIR = "alpha_vis"


def alpha_vis_paths(output_dir: Path, rel_score_path: Path, n_steps: int) -> List[Path]:
    """Map a scoring image path to alpha visualization paths."""
    rel = rel_score_path.as_posix()
    if rel.startswith("multiturn/"):
        return []
    stem = rel_score_path.stem
    parent = rel_score_path.parent
    return [
        output_dir / ALPHA_VIS_SUBDIR / parent / f"{stem}_alpha_step{step}.png"
        for step in range(1, n_steps + 1)
    ]


@torch.inference_mode()
def run_one_student_alpha(
    model,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
) -> Tuple[Image.Image, List[Image.Image]]:
    src_pil, captured_alphas, edited_pil = capture_student_alphas(
        model,
        image,
        prompt,
        num_inference_steps,
        guidance_scale,
        seed,
        device,
        preprocess_image_for_student,
        pil_to_tensor,
        return_edited=True,
        tensor_to_pil_fn=tensor_to_pil,
    )
    img_w, img_h = src_pil.size
    alpha_pils: List[Image.Image] = []
    for step_idx, alpha_tensor in enumerate(captured_alphas, start=1):
        alpha_pils.append(
            render_alpha_grid_blank(
                img_w,
                img_h,
                alpha_tensor,
                step_label=f"step {step_idx}/{num_inference_steps}",
            )
        )
    return edited_pil, alpha_pils


def process_tasks(
    runner,
    role: str,
    tasks: List[Tuple[str, Dict]],
    bench_root: Path,
    output_dir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    skip_existing: bool,
    device: str,
    desc: str,
) -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    for task_key, item in tqdm(tasks, desc=desc):
        suite_name, sample_key = task_key.split(":", 1)
        rel_paths = expected_outputs(task_key, item)
        abs_paths = [output_dir / rel for rel in rel_paths]
        alpha_paths = alpha_vis_paths(output_dir, rel_paths[0], num_steps)
        score_ready = all(p.is_file() for p in abs_paths)
        alpha_ready = (not alpha_paths) or all(p.is_file() for p in alpha_paths)
        if skip_existing and score_ready and alpha_ready:
            continue

        src_path = resolve_source_path(bench_root, item, task_key)
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_path}")

        task_seed = seed + int(sample_key.split(":")[-1]) if sample_key.split(":")[-1].isdigit() else seed
        image = Image.open(src_path).convert("RGB")

        if suite_name == "multiturn":
            run_multiturn_chain(
                runner,
                role,
                source=image,
                prompts=multiturn_prompts(item),
                out_paths=abs_paths,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=task_seed,
                skip_existing=skip_existing,
                device=device,
            )
        else:
            abs_paths[0].parent.mkdir(parents=True, exist_ok=True)
            if role == "student":
                result, alpha_items = run_one_student_alpha(
                    runner,
                    image=image,
                    prompt=item["prompt"],
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=task_seed,
                    device=device,
                )
                result.save(abs_paths[0])
                for alpha_pil, alpha_path in zip(alpha_items, alpha_paths):
                    alpha_path.parent.mkdir(parents=True, exist_ok=True)
                    alpha_pil.save(alpha_path)
            else:
                result = run_one(
                    runner,
                    role,
                    image=image,
                    prompt=item["prompt"],
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=task_seed,
                    device=device,
                )
                result.save(abs_paths[0])

        manifest[task_key] = {
            "suite": suite_name,
            "key": sample_key,
            "source": str(src_path),
            "prompt": item.get("prompt"),
            "turns": multiturn_prompts(item) if suite_name == "multiturn" else None,
            "outputs": [str(p) for p in abs_paths],
            "alpha_outputs": [str(p) for p in alpha_paths],
            "edit_type": item.get("edit_type"),
        }
    return manifest


def _infer_worker(gpu_id: int, tasks: List[Tuple[str, Dict]], worker_cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"
    if worker_cfg["role"] == "teacher":
        runner = build_teacher_pipeline(
            worker_cfg["model_path"], device, worker_cfg["cpu_offload"])
    elif worker_cfg["role"] == "klein":
        runner = build_klein_pipeline(
            worker_cfg["model_path"], device, worker_cfg["cpu_offload"])
    else:
        runner = build_student_model(
            Path(worker_cfg["config"]),
            Path(worker_cfg["ckpt"]),
            device,
        )
    manifest = process_tasks(
        runner,
        worker_cfg["role"],
        tasks,
        Path(worker_cfg["bench_root"]),
        Path(worker_cfg["output_dir"]),
        worker_cfg["num_steps"],
        worker_cfg["guidance_scale"],
        worker_cfg["seed"],
        worker_cfg["skip_existing"],
        device,
        desc=f"GPU {gpu_id}",
    )
    part_path = Path(worker_cfg["output_dir"]) / f"manifest.gpu{gpu_id}.json"
    part_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    del runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_parallel_inference(args, tasks, output_dir, num_steps, guidance_scale, gpu_ids):
    buckets = split_tasks_round_robin(tasks, len(gpu_ids))
    worker_cfg = {
        "role": args.role,
        "model_path": args.model_path,
        "config": str(args.config) if args.config is not None else "",
        "ckpt": str(args.ckpt) if args.ckpt is not None else "",
        "bench_root": str(args.bench_root),
        "output_dir": str(output_dir),
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "seed": args.seed,
        "skip_existing": args.skip_existing,
        "cpu_offload": args.cpu_offload,
    }
    if len(buckets) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        device = args.device
        if args.role == "teacher":
            runner = build_teacher_pipeline(args.model_path, device, args.cpu_offload)
        elif args.role == "klein":
            runner = build_klein_pipeline(args.model_path, device, args.cpu_offload)
        else:
            runner = build_student_model(args.config, args.ckpt, device)
        return process_tasks(
            runner,
            args.role,
            buckets[0],
            args.bench_root,
            output_dir,
            num_steps,
            guidance_scale,
            args.seed,
            args.skip_existing,
            device,
            desc=f"ImgEdit {args.role}/{args.suite} (alpha)",
        )

    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, bucket in zip(gpu_ids, buckets):
        p = ctx.Process(target=_infer_worker, args=(gpu_id, bucket, worker_cfg))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Inference worker failed with exit code {p.exitcode}")
    return merge_manifest_parts(output_dir, {})


def main() -> None:
    args = parse_args()
    if args.role == "student":
        if args.config is None or args.ckpt is None:
            raise ValueError("Student inference requires --config and --ckpt.")
        if not args.config.is_file():
            raise FileNotFoundError(f"Config not found: {args.config}")
        if not args.ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    num_steps, guidance_scale = resolve_inference_settings(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.suite, args.annotations_dir, args.bench_root)
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]

    complete, report = suite_complete(output_dir, tasks)
    if args.check_only:
        print(json.dumps({"output_dir": str(output_dir), "complete": complete, **report}, indent=2))
        raise SystemExit(0 if complete else 1)
    if complete and args.skip_existing:
        print(f"[skip] {args.role} outputs already complete at {output_dir} ({report['found']} files)")
        return

    manifest_path = output_dir / "manifest.json"
    manifest: Dict[str, Dict] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    gpu_ids = resolve_gpu_ids(args.num_gpus)
    use_parallel = len(gpu_ids) > 1 and not args.cpu_offload

    if use_parallel:
        print(f"[infer] parallel on GPUs: {gpu_ids}")
        new_manifest = run_parallel_inference(args, tasks, output_dir, num_steps, guidance_scale, gpu_ids)
        manifest.update(new_manifest)
    else:
        if len(gpu_ids) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        device = args.device
        if args.role == "teacher":
            runner = build_teacher_pipeline(args.model_path, device, args.cpu_offload)
        elif args.role == "klein":
            runner = build_klein_pipeline(args.model_path, device, args.cpu_offload)
        else:
            runner = build_student_model(args.config, args.ckpt, device)
        new_manifest = process_tasks(
            runner,
            args.role,
            tasks,
            args.bench_root,
            output_dir,
            num_steps,
            guidance_scale,
            args.seed,
            args.skip_existing,
            device,
            desc=f"ImgEdit {args.role}/{args.suite} (alpha)",
        )
        manifest.update(new_manifest)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "role": args.role,
        "model_path": args.model_path,
        "config": str(args.config) if args.config is not None else "",
        "ckpt": str(args.ckpt) if args.ckpt is not None else "",
        "run_name": args.run_name,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "suite": args.suite,
        "num_tasks": len(manifest),
        "num_gpus": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "student_resize_mode": os.environ.get("STUDENT_RESIZE_MODE", "center_crop"),
        "student_image_size": int(os.environ.get("STUDENT_IMAGE_SIZE", "1024")),
        "alpha_vis_subdir": ALPHA_VIS_SUBDIR,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(manifest)} task records to {output_dir}")
    print(f"Alpha patch visualizations -> {output_dir / ALPHA_VIS_SUBDIR}")


if __name__ == "__main__":
    main()
