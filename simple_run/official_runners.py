"""GEdit / ImgEdit loops for official Qwen and FLUX.2-klein-base simple_run."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from tqdm import tqdm

from official_common import (
    EDITFLOW_ROOT,
    configure_hf_env,
    slice_split,
    merge_manifest_parts,
    manifest_name,
    resolve_gpu_ids,
    split_tasks_round_robin,
)

GEDIT_ROOT = EDITFLOW_ROOT / "evaluation" / "gedit_v2"
IMGEDIT_ROOT = EDITFLOW_ROOT / "evaluation" / "imgedit_bench"
DEFAULT_GEDIT_META = Path("/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json")
DEFAULT_IMGEDIT_BENCH = Path(
    os.environ.get(
        "IMGEDIT_BENCH_ROOT",
        "/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark",
    )
)


def _ensure_eval_paths() -> None:
    for path in (str(GEDIT_ROOT), str(IMGEDIT_ROOT), str(EDITFLOW_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _import_model(role: str):
    if role == "qwen":
        from qwen_common import build_pipeline, run_one
    elif role == "klein":
        from klein_common import build_pipeline, run_one
    else:
        raise ValueError(f"Unknown role={role!r}")
    return build_pipeline, run_one


def add_common_args(parser: argparse.ArgumentParser, *, default_output: Path) -> None:
    parser.add_argument("--output_dir", type=Path, default=default_output)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--split_id",
        type=int,
        default=int(os.environ.get("SPLIT_ID", "0")),
        help="Shard index. Tasks are metadata[split_id::split_num].",
    )
    parser.add_argument(
        "--split_num",
        type=int,
        default=int(os.environ.get("SPLIT_NUM", "1")),
        help="Number of shards. Tasks are metadata[split_id::split_num].",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=int(os.environ.get("NUM_GPUS", "1")),
        help="GPUs used inside this process after the split slice.",
    )
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")


def apply_split(tasks: list, args: argparse.Namespace, label: str) -> list:
    n_all = len(tasks)
    tasks = slice_split(tasks, args.split_id, args.split_num)
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]
    print(
        f"[{label}] {n_all} total -> split {args.split_id}/{args.split_num} "
        f"keeps {len(tasks)}  (metadata[split_id::split_num])"
    )
    return tasks


def write_run_config(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    run_root = output_dir.parent if output_dir.name == "student" else output_dir
    run_root.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {payload[key]}" for key in payload]
    (run_root / "run_config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_gedit_tasks(
    run_one,
    pipe,
    tasks: List[Tuple[str, Dict]],
    output_dir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    skip_existing: bool,
    run_extra: Dict[str, Any],
    desc: str,
) -> Dict[str, Dict]:
    from run_editflow_gedit_infer import case_dir, pred_path, write_case_bundle

    manifest: Dict[str, Dict] = {}
    for stem, item in tqdm(tasks, desc=desc):
        out_path = pred_path(output_dir, item)
        bundle = case_dir(output_dir, item, stem)
        if skip_existing and out_path.is_file() and (bundle / "pred.png").is_file():
            manifest[stem] = {
                "index": item["index"],
                "source": item["src_image_path"],
                "prompt": item["prompt"],
                "edit_type": item["edit_type"],
                "output": str(out_path),
                "case_dir": str(bundle),
                "skipped": True,
            }
            continue
        src_path = Path(item["src_image_path"])
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_path}")
        image = Image.open(src_path).convert("RGB")
        result = run_one(
            pipe,
            image=image,
            prompt=item["prompt"],
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed + int(item["index"]),
            **run_extra,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path, quality=95)
        write_case_bundle(bundle, image, result, item["prompt"])
        manifest[stem] = {
            "index": item["index"],
            "source": str(src_path),
            "prompt": item["prompt"],
            "edit_type": item["edit_type"],
            "output": str(out_path),
            "case_dir": str(bundle),
            "skipped": False,
        }
    return manifest


def process_imgedit_tasks(
    run_one,
    pipe,
    tasks: List[Tuple[str, Dict]],
    bench_root: Path,
    output_dir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    skip_existing: bool,
    run_extra: Dict[str, Any],
    desc: str,
) -> Dict[str, Dict]:
    from run_editflow_imgedit_infer import (
        basic_case_dir,
        case_bundle_ready,
        expected_outputs,
        resolve_source_path,
        save_png_atomic,
        try_open_rgb_image,
        write_basic_case_bundle,
    )

    # Same layout as evaluation/imgedit_bench/outputs/runs/flux_kontext_official:
    #   basic/{key}.png
    #   basic/{Category}/{key}/{src.png,pred.png,prompt.txt}
    #   uge/{key}.png
    manifest: Dict[str, Dict] = {}
    for task_key, item in tqdm(tasks, desc=desc):
        suite_name, sample_key = task_key.split(":", 1)
        abs_paths = [output_dir / rel for rel in expected_outputs(task_key, item)]
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

        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            print(f"[skip] missing prompt: {task_key}", flush=True)
            continue

        tail = sample_key.split(":")[-1]
        task_seed = seed + int(tail) if tail.isdigit() else seed
        need_infer = (not score_ready) or (
            case_dir is not None and not case_bundle_ready(case_dir)
        )
        if need_infer:
            result = run_one(
                pipe,
                image=image,
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=task_seed,
                **run_extra,
            )
            save_png_atomic(result, abs_paths[0])
        else:
            result = Image.open(abs_paths[0]).convert("RGB")

        if case_dir is not None:
            write_basic_case_bundle(case_dir, image, result, prompt)

        manifest[task_key] = {
            "suite": suite_name,
            "key": sample_key,
            "source": str(src_path),
            "prompt": prompt,
            "outputs": [str(p) for p in abs_paths],
            "case_dir": str(case_dir) if case_dir is not None else None,
            "edit_type": item.get("edit_type"),
        }
    return manifest


def _worker(gpu_id: int, tasks: list, cfg: dict) -> None:
    configure_hf_env()
    _ensure_eval_paths()
    for path in cfg["extra_sys_path"]:
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    build_pipeline, run_one = _import_model(cfg["role"])
    pipe = build_pipeline(cfg["model_path"], "cuda", cfg["cpu_offload"])
    if cfg["bench"] == "gedit":
        manifest = process_gedit_tasks(
            run_one,
            pipe,
            tasks,
            Path(cfg["output_dir"]),
            cfg["num_steps"],
            cfg["guidance_scale"],
            cfg["seed"],
            cfg["skip_existing"],
            cfg["run_extra"],
            desc=f"GPU {gpu_id}",
        )
    else:
        manifest = process_imgedit_tasks(
            run_one,
            pipe,
            tasks,
            Path(cfg["bench_root"]),
            Path(cfg["output_dir"]),
            cfg["num_steps"],
            cfg["guidance_scale"],
            cfg["seed"],
            cfg["skip_existing"],
            cfg["run_extra"],
            desc=f"GPU {gpu_id}",
        )
    part = Path(cfg["output_dir"]) / f"manifest.gpu{gpu_id}.json"
    part.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    del pipe


def launch(tasks: list, args: argparse.Namespace, cfg: dict) -> None:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_name = manifest_name(args.split_id, args.split_num)
    gpu_ids = resolve_gpu_ids(args.num_gpus)
    worker_cfg = dict(cfg)
    worker_cfg["cpu_offload"] = args.cpu_offload
    worker_cfg["skip_existing"] = args.skip_existing
    worker_cfg["seed"] = args.seed

    if len(gpu_ids) > 1 and not args.cpu_offload:
        print(f"[infer] parallel on GPUs: {gpu_ids}")
        buckets = split_tasks_round_robin(tasks, len(gpu_ids))
        ctx = mp.get_context("spawn")
        procs = []
        for gpu_id, bucket in zip(gpu_ids, buckets):
            p = ctx.Process(target=_worker, args=(gpu_id, bucket, worker_cfg))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker failed with exit code {p.exitcode}")
        merge_manifest_parts(
            output_dir, glob_pat="manifest.gpu*.json", dest_name=dest_name
        )
        return

    if len(gpu_ids) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    build_pipeline, run_one = _import_model(cfg["role"])
    pipe = build_pipeline(args.model_path, "cuda", args.cpu_offload)
    if cfg["bench"] == "gedit":
        manifest = process_gedit_tasks(
            run_one,
            pipe,
            tasks,
            output_dir,
            cfg["num_steps"],
            cfg["guidance_scale"],
            args.seed,
            args.skip_existing,
            cfg["run_extra"],
            desc=cfg["bench"],
        )
    else:
        manifest = process_imgedit_tasks(
            run_one,
            pipe,
            tasks,
            Path(cfg["bench_root"]),
            output_dir,
            cfg["num_steps"],
            cfg["guidance_scale"],
            args.seed,
            args.skip_existing,
            cfg["run_extra"],
            desc=cfg["bench"],
        )
    (output_dir / dest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_gedit(role: str, defaults, extra_sys_path: List[str]) -> None:
    configure_hf_env()
    _ensure_eval_paths()
    for path in extra_sys_path:
        if path not in sys.path:
            sys.path.insert(0, path)

    from run_editflow_gedit_infer import load_gedit_tasks

    parser = argparse.ArgumentParser(
        description=f"Official {role} GEdit-v2 inference (generation only)."
    )
    parser.add_argument("--meta_json", type=Path, default=DEFAULT_GEDIT_META)
    add_common_args(
        parser,
        default_output=GEDIT_ROOT / "outputs" / "runs" / defaults.RUN_TAG_GEDIT,
    )
    args = parser.parse_args()
    args.model_path = args.model_path or defaults.DEFAULT_MODEL
    args.num_inference_steps = args.num_inference_steps or defaults.DEFAULT_STEPS
    args.guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else defaults.DEFAULT_GUIDANCE
    )
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.meta_json.is_file():
        raise FileNotFoundError(f"GEdit meta not found: {args.meta_json}")

    tasks = apply_split(load_gedit_tasks(args.meta_json), args, "gedit")
    run_extra: Dict[str, Any] = {}
    if role == "qwen":
        run_extra = {
            "true_cfg_scale": defaults.DEFAULT_TRUE_CFG,
            "negative_prompt": defaults.DEFAULT_NEGATIVE,
        }
    print(
        f"[gedit] role={role} resize={defaults.RESIZE_MODE} "
        f"steps={args.num_inference_steps} guidance={args.guidance_scale} "
        f"model={args.model_path}"
    )
    print(f"[gedit] output -> {args.output_dir}")
    write_run_config(
        args.output_dir,
        {
            "bench": "gedit",
            "role": role,
            "hf_id": defaults.HF_ID,
            "model_path": args.model_path,
            "resize_mode": defaults.RESIZE_MODE,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "split_id": args.split_id,
            "split_num": args.split_num,
            "num_tasks": len(tasks),
            "meta_json": str(args.meta_json),
        },
    )
    launch(
        tasks,
        args,
        {
            "role": role,
            "bench": "gedit",
            "model_path": args.model_path,
            "output_dir": str(args.output_dir),
            "num_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "run_extra": run_extra,
            "extra_sys_path": extra_sys_path,
            "bench_root": "",
        },
    )
    print(f"[gedit] done -> {args.output_dir}")


def run_imgedit(role: str, defaults, extra_sys_path: List[str]) -> None:
    configure_hf_env()
    _ensure_eval_paths()
    for path in extra_sys_path:
        if path not in sys.path:
            sys.path.insert(0, path)

    from run_editflow_imgedit_infer import (
        BASIC_CATEGORY_DIRS,
        load_basic_tasks,
        load_uge_tasks,
    )

    parser = argparse.ArgumentParser(
        description=(
            f"Official {role} ImgEdit-Bench inference (generation only). "
            "Suite 'all' is basic + uge only; no multiturn."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("basic", "uge", "all"),
        default="all",
        help="all = basic + uge (same as flux_kontext_official). No multiturn.",
    )
    parser.add_argument("--bench_root", type=Path, default=DEFAULT_IMGEDIT_BENCH)
    parser.add_argument(
        "--annotations_dir",
        type=Path,
        default=IMGEDIT_ROOT / "annotations",
    )
    add_common_args(
        parser,
        default_output=IMGEDIT_ROOT / "outputs" / "runs" / defaults.RUN_TAG_IMGEDIT,
    )
    args = parser.parse_args()
    args.model_path = args.model_path or defaults.DEFAULT_MODEL
    args.num_inference_steps = args.num_inference_steps or defaults.DEFAULT_STEPS
    args.guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else defaults.DEFAULT_GUIDANCE
    )
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.bench_root.is_dir():
        raise FileNotFoundError(f"Bench root not found: {args.bench_root}")

    tasks: list = []
    if args.suite in ("basic", "all"):
        tasks.extend(load_basic_tasks(args.annotations_dir))
    if args.suite in ("uge", "all"):
        tasks.extend(load_uge_tasks(args.annotations_dir))
    tasks = apply_split(tasks, args, "imgedit")
    run_extra: Dict[str, Any] = {}
    if role == "qwen":
        run_extra = {
            "true_cfg_scale": defaults.DEFAULT_TRUE_CFG,
            "negative_prompt": defaults.DEFAULT_NEGATIVE,
        }
    print(
        f"[imgedit] role={role} resize={defaults.RESIZE_MODE} "
        f"suite={args.suite} (basic+uge, no multiturn) "
        f"steps={args.num_inference_steps} "
        f"guidance={args.guidance_scale} model={args.model_path}"
    )
    print(f"[imgedit] output -> {args.output_dir}")
    print(f"[imgedit] basic categories: {', '.join(BASIC_CATEGORY_DIRS)}")
    write_run_config(
        args.output_dir,
        {
            "bench": "imgedit",
            "role": role,
            "hf_id": defaults.HF_ID,
            "model_path": args.model_path,
            "resize_mode": defaults.RESIZE_MODE,
            "suite": args.suite,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "split_id": args.split_id,
            "split_num": args.split_num,
            "num_tasks": len(tasks),
            "bench_root": str(args.bench_root),
        },
    )
    launch(
        tasks,
        args,
        {
            "role": role,
            "bench": "imgedit",
            "model_path": args.model_path,
            "output_dir": str(args.output_dir),
            "num_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "run_extra": run_extra,
            "extra_sys_path": extra_sys_path,
            "bench_root": str(args.bench_root),
        },
    )
    print(f"[imgedit] done -> {args.output_dir}")
