#!/usr/bin/env python3
"""ImgEdit-Bench inference for SpurFlow alpha student models (gen + alpha v6 in one pass).

Basic layout (9 categories, no flat ``basic/{key}.png`` duplicates):
  student/basic/{Action,Add,...}/{key}/
    src.png
    edit.png / pred.png   (same edit; pred kept for scoring helpers)
    prompt.txt
    step{1,2}_{alpha,heatmap,overlay}.png   # alpha v6 continuous maps

UGE / multiturn still use flat score paths (no category folders there).

GPT scoring for basic resolves ``Category/{key}/pred.png`` (or edit.png).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alpha_vis import (  # noqa: E402
    capture_student_alphas,
    render_alpha_grid_blank,
    render_continuous_alpha_on_src,
)
from alpha_model_utils import build_alpha_vis_model  # noqa: E402
from run_spurflow_imgedit_infer import (  # noqa: E402
    basic_case_dir,
    build_klein_pipeline,
    build_teacher_pipeline,
    category_dir_name,
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
    run_one_student,
    split_tasks_round_robin,
    tensor_to_pil,
    try_open_rgb_image,
)

HEATMAP_BLEND = float(os.environ.get("HEATMAP_BLEND", "0.36"))
OVERLAY_BLEND = float(os.environ.get("OVERLAY_BLEND", "0.52"))
SKIP_ALPHA_VIS = os.environ.get("SKIP_ALPHA_VIS", "0").strip() not in {"", "0", "false", "False"}

BASIC_CATEGORY_DIRS = (
    "Action",
    "Add",
    "Adjust",
    "Background",
    "Compose",
    "Extract",
    "Remove",
    "Replace",
    "Style",
)


def v6_case_paths(case_dir: Path) -> Dict[str, Path]:
    return {
        "prompt": case_dir / "prompt.txt",
        "src": case_dir / "src.png",
        "edit": case_dir / "edit.png",
        "pred": case_dir / "pred.png",
        "step1_alpha": case_dir / "step1_alpha.png",
        "step1_heatmap": case_dir / "step1_heatmap.png",
        "step1_overlay": case_dir / "step1_overlay.png",
        "step2_alpha": case_dir / "step2_alpha.png",
        "step2_heatmap": case_dir / "step2_heatmap.png",
        "step2_overlay": case_dir / "step2_overlay.png",
    }


def v6_case_ready(case_dir: Path) -> bool:
    paths = v6_case_paths(case_dir)
    required = [
        paths["prompt"], paths["src"], paths["edit"], paths["pred"],
        paths["step1_alpha"], paths["step1_heatmap"], paths["step1_overlay"],
        paths["step2_alpha"], paths["step2_heatmap"], paths["step2_overlay"],
    ]
    return all(p.is_file() for p in required)


def photo_case_ready(case_dir: Path) -> bool:
    return (case_dir / "pred.png").is_file()


def case_skip_ready(case_dir: Path) -> bool:
    if SKIP_ALPHA_VIS:
        return photo_case_ready(case_dir)
    return v6_case_ready(case_dir)


def write_v6_case_bundle(
    case_dir: Path,
    src_pil: Image.Image,
    edit_pil: Image.Image,
    prompt: str,
    alphas: List[torch.Tensor],
    num_steps: int,
    heatmap_blend: float,
    overlay_blend: float,
) -> Dict[str, str]:
    if len(alphas) < 2:
        raise RuntimeError(f"Expected >=2 alpha maps for v6, got {len(alphas)}")
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = v6_case_paths(case_dir)
    paths["prompt"].write_text((prompt or "").rstrip() + "\n", encoding="utf-8")
    src_pil.save(paths["src"])
    edit_pil.save(paths["edit"])
    edit_pil.save(paths["pred"])  # scoring looks for pred.png

    img_w, img_h = src_pil.size
    for step_i, alpha in enumerate(alphas[:2], start=1):
        step_label = f"step {step_i}/{num_steps}"
        render_alpha_grid_blank(
            img_w, img_h, alpha, step_label=step_label, continuous=True,
        ).save(paths[f"step{step_i}_alpha"])
        render_continuous_alpha_on_src(
            src_pil,
            alpha,
            blend=heatmap_blend,
            step_label=step_label,
            title_prefix="Alpha heatmap",
            draw_grid=True,
        ).save(paths[f"step{step_i}_heatmap"])
        render_continuous_alpha_on_src(
            src_pil,
            alpha,
            blend=overlay_blend,
            step_label=step_label,
            title_prefix="Alpha overlay",
            draw_grid=True,
        ).save(paths[f"step{step_i}_overlay"])
    return {k: str(v) for k, v in paths.items()}


def write_photo_case_bundle(
    case_dir: Path,
    src_pil: Image.Image,
    edit_pil: Image.Image,
    prompt: str,
) -> Dict[str, str]:
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = v6_case_paths(case_dir)
    paths["prompt"].write_text((prompt or "").rstrip() + "\n", encoding="utf-8")
    src_pil.save(paths["src"])
    edit_pil.save(paths["edit"])
    edit_pil.save(paths["pred"])
    return {k: str(paths[k]) for k in ("prompt", "src", "edit", "pred")}


def basic_suite_complete(output_dir: Path, tasks: List[Tuple[str, Dict]]) -> Tuple[bool, Dict[str, int]]:
    """Completion check: basic uses category packs; uge/multiturn use flat paths."""
    expected = 0
    found = 0
    for task_key, item in tasks:
        suite_name, sample_key = task_key.split(":", 1)
        if suite_name == "basic":
            expected += 1
            case_dir = basic_case_dir(output_dir, item, sample_key)
            if case_skip_ready(case_dir):
                found += 1
            continue
        rel_paths = expected_outputs(task_key, item)
        expected += len(rel_paths)
        for rel in rel_paths:
            if (output_dir / rel).is_file():
                found += 1
    return found == expected and expected > 0, {"expected": expected, "found": found}


@torch.inference_mode()
def run_one_student_alpha_v6(
    model,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
) -> Tuple[Image.Image, Image.Image, List[torch.Tensor]]:
    """One forward: edited image + per-step alpha tensors (for v6 renders)."""
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
    return src_pil, edited_pil, list(captured_alphas)


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
    heatmap_blend: float = HEATMAP_BLEND,
    overlay_blend: float = OVERLAY_BLEND,
) -> Dict[str, Dict]:
    if role == "student" and num_steps != 2:
        raise ValueError(f"Alpha v6 joint infer expects nfe=2, got {num_steps}.")

    manifest: Dict[str, Dict] = {}
    for task_key, item in tqdm(tasks, desc=desc):
        suite_name, sample_key = task_key.split(":", 1)
        rel_paths = expected_outputs(task_key, item)
        abs_paths = [output_dir / rel for rel in rel_paths]
        case_dir = (
            basic_case_dir(output_dir, item, sample_key)
            if suite_name == "basic" else None)

        if suite_name == "basic" and role == "student":
            if skip_existing and case_dir is not None and case_skip_ready(case_dir):
                continue
        else:
            score_ready = all(try_open_rgb_image(p) is not None for p in abs_paths)
            if skip_existing and score_ready:
                if case_dir is None or case_skip_ready(case_dir) or role != "student":
                    # non-basic: flat only; basic non-student shouldn't happen
                    if suite_name != "basic" or role != "student":
                        continue

        src_path = resolve_source_path(bench_root, item, task_key)
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_path}")

        task_seed = seed + int(sample_key.split(":")[-1]) if sample_key.split(":")[-1].isdigit() else seed
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

        try:
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
                alpha_outputs = []
            elif suite_name == "basic" and role == "student":
                assert case_dir is not None
                if SKIP_ALPHA_VIS:
                    src_pil = preprocess_image_for_student(image)
                    edit_pil = run_one_student(
                        runner,
                        image,
                        item["prompt"],
                        num_steps,
                        guidance_scale,
                        task_seed,
                        device,
                    )
                    alpha_outputs = write_photo_case_bundle(
                        case_dir,
                        src_pil,
                        edit_pil,
                        item.get("prompt", ""),
                    )
                else:
                    src_pil, edit_pil, alphas = run_one_student_alpha_v6(
                        runner,
                        image=image,
                        prompt=item["prompt"],
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        seed=task_seed,
                        device=device,
                    )
                    alpha_outputs = write_v6_case_bundle(
                        case_dir,
                        src_pil,
                        edit_pil,
                        item.get("prompt", ""),
                        alphas,
                        num_steps,
                        heatmap_blend=heatmap_blend,
                        overlay_blend=overlay_blend,
                    )
                # No flat basic/{key}.png — scoring reads Category/key/pred.png
            else:
                # UGE student / teacher / etc.: flat score png only
                abs_paths[0].parent.mkdir(parents=True, exist_ok=True)
                if role == "student":
                    if SKIP_ALPHA_VIS:
                        edit_pil = run_one_student(
                            runner,
                            image,
                            item["prompt"],
                            num_steps,
                            guidance_scale,
                            task_seed,
                            device,
                        )
                        edit_pil.save(abs_paths[0])
                        alpha_outputs = {"skip_alpha_vis": True}
                    else:
                        src_pil, edit_pil, alphas = run_one_student_alpha_v6(
                            runner,
                            image=image,
                            prompt=item["prompt"],
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale,
                            seed=task_seed,
                            device=device,
                        )
                        edit_pil.save(abs_paths[0])
                        alpha_outputs = {
                            f"step{i}_alpha_tensor_shape": list(a.shape)
                            for i, a in enumerate(alphas[:2], start=1)
                        }
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
                    alpha_outputs = []
        except (UnidentifiedImageError, OSError) as e:
            print(f"[skip] {task_key}: {e}", flush=True)
            manifest[task_key] = {
                "suite": suite_name,
                "key": sample_key,
                "source": str(src_path),
                "error": str(e),
            }
            continue

        manifest[task_key] = {
            "suite": suite_name,
            "key": sample_key,
            "source": str(src_path),
            "prompt": item.get("prompt"),
            "turns": multiturn_prompts(item) if suite_name == "multiturn" else None,
            "outputs": [str(p) for p in abs_paths] if suite_name != "basic" else [],
            "case_dir": str(case_dir) if case_dir is not None else None,
            "alpha_v6": alpha_outputs if isinstance(alpha_outputs, dict) else {},
            "edit_type": item.get("edit_type"),
            "category": category_dir_name(item.get("edit_type")) if suite_name == "basic" else None,
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
        runner = build_alpha_vis_model(
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
        heatmap_blend=worker_cfg["heatmap_blend"],
        overlay_blend=worker_cfg["overlay_blend"],
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
        "heatmap_blend": HEATMAP_BLEND,
        "overlay_blend": OVERLAY_BLEND,
    }
    if len(buckets) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        device = args.device
        if args.role == "teacher":
            runner = build_teacher_pipeline(args.model_path, device, args.cpu_offload)
        elif args.role == "klein":
            runner = build_klein_pipeline(args.model_path, device, args.cpu_offload)
        else:
            runner = build_alpha_vis_model(args.config, args.ckpt, device)
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
            desc=f"ImgEdit {args.role}/{args.suite} (alpha+v6)",
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

    complete, report = basic_suite_complete(output_dir, tasks)
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
            runner = build_alpha_vis_model(args.config, args.ckpt, device)
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
            desc=f"ImgEdit {args.role}/{args.suite} (alpha+v6)",
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
        "alpha_vis": "v6_in_case_dir",
        "heatmap_blend": HEATMAP_BLEND,
        "overlay_blend": OVERLAY_BLEND,
        "flat_basic_score_png": False,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(manifest)} task records to {output_dir}")
    print("Basic layout: basic/<Category>/<key>/{src,edit,pred,prompt,step{1,2}_*}")
    print("(no flat basic/{key}.png duplicates)")


if __name__ == "__main__":
    main()
