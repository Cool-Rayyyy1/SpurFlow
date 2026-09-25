#!/usr/bin/env python3
"""GEdit-v2 inference for official image-editing model baselines.

Writes:
  {output_dir}/{tgt_image_path}             — matches meta JSON for GPT scoring
  {output_dir}/cases/{edit_type}/{stem}/
    src.png  pred.png  prompt.txt
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parents[1]
IMGEDIT_ROOT = EVAL_ROOT / "imgedit_bench"
PROJECT_ROOT = EVAL_ROOT.parent
for path in (str(IMGEDIT_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_spurflow_imgedit_infer import (  # noqa: E402
    build_teacher_pipeline,
    build_klein_pipeline,
    build_qwen_pipeline,
    merge_manifest_parts,
    resolve_gpu_ids,
    run_one_teacher,
    run_one_klein,
    run_one_qwen,
    split_tasks_round_robin,
    try_open_rgb_image,
)

DEFAULT_META = Path("/path/to/data/benchmark/GEdit_v2/gedit_v2_meta.json")
DEFAULT_KONTEXT = os.environ.get(
    "KONTEXT_MODEL_PATH", "/path/to/checkpoints/FLUX.1-Kontext-dev"
)
DEFAULT_QWEN = os.environ.get(
    "QWEN_MODEL_PATH", "/path/to/checkpoints/Qwen-Image-Edit-2511"
)
DEFAULT_KLEIN = os.environ.get(
    "KLEIN_MODEL_PATH", "/path/to/checkpoints/FLUX.2-klein-base-9B"
)

# Official pipeline defaults.
KONTEXT_DEFAULT_STEPS = 28
KONTEXT_DEFAULT_GUIDANCE = 2.5
QWEN_DEFAULT_STEPS = 40
QWEN_DEFAULT_GUIDANCE = 1.0
KLEIN_BASE_DEFAULT_STEPS = 50
KLEIN_BASE_DEFAULT_GUIDANCE = 4.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GEdit-v2 official Kontext / Qwen / FLUX.2-klein-base inference.")
    p.add_argument("--role", choices=("kontext", "qwen", "klein"), required=True)
    p.add_argument("--meta_json", type=Path, default=DEFAULT_META)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_gpus", type=int, default=int(os.environ.get("NUM_GPUS", "8")))
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--cpu_offload", action="store_true")
    return p.parse_args()


def default_model_path(role: str) -> str:
    if role == "kontext":
        return DEFAULT_KONTEXT
    return DEFAULT_QWEN if role == "qwen" else DEFAULT_KLEIN


def resolve_settings(role: str, steps: int | None, guidance: float | None) -> Tuple[int, float]:
    if role == "kontext":
        nfe = steps if steps is not None else int(
            os.environ.get("KONTEXT_STEPS", KONTEXT_DEFAULT_STEPS))
        cfg = guidance if guidance is not None else float(
            os.environ.get("KONTEXT_GUIDANCE", KONTEXT_DEFAULT_GUIDANCE))
        return nfe, cfg
    if role == "qwen":
        nfe = steps if steps is not None else int(os.environ.get("QWEN_STEPS", QWEN_DEFAULT_STEPS))
        cfg = guidance if guidance is not None else float(
            os.environ.get("QWEN_GUIDANCE", QWEN_DEFAULT_GUIDANCE))
        return nfe, cfg
    nfe = steps if steps is not None else int(
        os.environ.get("KLEIN_STEPS", KLEIN_BASE_DEFAULT_STEPS))
    cfg = guidance if guidance is not None else float(
        os.environ.get("KLEIN_GUIDANCE", KLEIN_BASE_DEFAULT_GUIDANCE))
    return nfe, cfg


def load_gedit_tasks(meta_json: Path) -> List[Tuple[str, Dict]]:
    records = json.loads(meta_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{meta_json} must be a JSON list")
    meta_root = meta_json.resolve().parent
    tasks: List[Tuple[str, Dict]] = []
    for index, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        src = str(rec.get("src_image_path") or "").strip()
        if src and not Path(src).is_absolute():
            src = str((meta_root / src).resolve())
        prompt = str(rec.get("instruction") or rec.get("prompt") or "").strip()
        tgt_rel = str(rec.get("tgt_image_path") or "").strip()
        edit_type = str(
            rec.get("edit_type")
            or rec.get("task")
            or rec.get("task_type")
            or "unknown"
        ).strip() or "unknown"
        stem = Path(tgt_rel).stem if tgt_rel else f"{edit_type}_{index:06d}"
        if not tgt_rel:
            tgt_rel = f"GEdit_v2/{stem}.jpg"
        tgt_path = Path(tgt_rel)
        if tgt_path.is_absolute() or ".." in tgt_path.parts:
            raise ValueError(
                f"tgt_image_path must be a safe relative path, got {tgt_rel!r}"
            )
        tasks.append((
            stem,
            {
                "index": index,
                "prompt": prompt,
                "src_image_path": src,
                "tgt_image_path": tgt_rel,
                "edit_type": edit_type,
            },
        ))
    return tasks


def pred_path(output_dir: Path, item: Dict) -> Path:
    return output_dir / item["tgt_image_path"]


def case_dir(output_dir: Path, item: Dict, stem: str) -> Path:
    return output_dir / "cases" / item["edit_type"] / stem


def write_case_bundle(folder: Path, src: Image.Image, pred: Image.Image, prompt: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    src.save(folder / "src.png")
    pred.save(folder / "pred.png")
    (folder / "prompt.txt").write_text((prompt or "").rstrip() + "\n", encoding="utf-8")


def build_pipeline(role: str, model_path: str, device: str, cpu_offload: bool):
    if role == "kontext":
        return build_teacher_pipeline(model_path, device, cpu_offload)
    if role == "qwen":
        return build_qwen_pipeline(model_path, device, cpu_offload)
    return build_klein_pipeline(model_path, device, cpu_offload)


def run_one(role: str, pipe, image: Image.Image, prompt: str, steps: int, guidance: float, seed: int):
    if role == "kontext":
        return run_one_teacher(pipe, image, prompt, steps, guidance, seed)
    if role == "qwen":
        return run_one_qwen(pipe, image, prompt, steps, guidance, seed)
    return run_one_klein(pipe, image, prompt, steps, guidance, seed)


def process_tasks(
    role: str,
    pipe,
    tasks: List[Tuple[str, Dict]],
    output_dir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    skip_existing: bool,
    desc: str,
) -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    for stem, item in tqdm(tasks, desc=desc):
        out_path = pred_path(output_dir, item)
        bundle = case_dir(output_dir, item, stem)
        existing = try_open_rgb_image(out_path) if skip_existing else None
        if existing is not None and (bundle / "pred.png").is_file():
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
        task_seed = seed + int(item["index"])
        result = run_one(role, pipe, image, item["prompt"], num_steps, guidance_scale, task_seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.convert("RGB").save(out_path, quality=95)
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


def _infer_worker(gpu_id: int, tasks: List[Tuple[str, Dict]], worker_cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"
    pipe = build_pipeline(
        worker_cfg["role"],
        worker_cfg["model_path"],
        device,
        worker_cfg["cpu_offload"],
    )
    manifest = process_tasks(
        worker_cfg["role"],
        pipe,
        tasks,
        Path(worker_cfg["output_dir"]),
        worker_cfg["num_steps"],
        worker_cfg["guidance_scale"],
        worker_cfg["seed"],
        worker_cfg["skip_existing"],
        desc=f"GPU {gpu_id}",
    )
    part_path = Path(worker_cfg["output_dir"]) / f"manifest.gpu{gpu_id}.json"
    part_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if not args.meta_json.is_file():
        raise FileNotFoundError(f"GEdit meta not found: {args.meta_json}")

    model_path = args.model_path or default_model_path(args.role)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    num_steps, guidance_scale = resolve_settings(
        args.role, args.num_inference_steps, args.guidance_scale)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_gedit_tasks(args.meta_json)
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]
    print(
        f"[gedit] role={args.role} n={len(tasks)} steps={num_steps} "
        f"guidance={guidance_scale} model={model_path}"
    )

    gpu_ids = resolve_gpu_ids(args.num_gpus)
    use_parallel = len(gpu_ids) > 1 and not args.cpu_offload
    worker_cfg = {
        "role": args.role,
        "model_path": model_path,
        "output_dir": str(output_dir),
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "seed": args.seed,
        "skip_existing": args.skip_existing,
        "cpu_offload": args.cpu_offload,
    }

    if use_parallel:
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
                raise RuntimeError(f"Inference worker failed with exit code {p.exitcode}")
        manifest = merge_manifest_parts(output_dir, {})
    else:
        if len(gpu_ids) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        pipe = build_pipeline(args.role, model_path, args.device, args.cpu_offload)
        manifest = process_tasks(
            args.role,
            pipe,
            tasks,
            output_dir,
            num_steps,
            guidance_scale,
            args.seed,
            args.skip_existing,
            desc=f"GEdit-v2 {args.role}",
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "meta_json": str(args.meta_json),
        "role": args.role,
        "model_path": model_path,
        "run_name": args.run_name,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": os.environ.get("QWEN_TRUE_CFG_SCALE", "4.0") if args.role == "qwen" else None,
        "lora_path": os.environ.get("QWEN_LORA_PATH", "").strip() or None,
        "num_tasks": len(manifest),
        "num_gpus": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "gedit_flat_dir": str(output_dir / "GEdit_v2"),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(manifest)} GEdit results -> {output_dir}")
    print(f"Flat preds (for GPT-4.1 scoring) -> {output_dir / 'GEdit_v2'}")


if __name__ == "__main__":
    main()
