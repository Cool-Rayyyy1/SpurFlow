#!/usr/bin/env python3
"""GEdit inference for EditFlow alpha / fixed-eps student models.

No scoring. Writes:
  {output_dir}/{tgt_image_path}             — matches meta JSON for later eval
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
EDITFLOW_ROOT = EVAL_ROOT.parent
for path in (str(IMGEDIT_ROOT), str(EDITFLOW_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from alpha_model_utils import build_alpha_vis_model  # noqa: E402
from run_editflow_imgedit_infer import (  # noqa: E402
    build_student_model,
    get_student_mixture_stats,
    install_mixture_stats_hook,
    merge_manifest_parts,
    resolve_gpu_ids,
    resolve_inference_settings,
    run_one_student,
    split_tasks_round_robin,
    write_mixture_stats_files,
)

DEFAULT_META = Path("/path/to/data/benchmark/GEdit_v2/gedit_v2_meta.json")
DEFAULT_KONTEXT = os.environ.get(
    "KONTEXT_MODEL_PATH", "/path/to/checkpoints/FLUX.1-Kontext-dev"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEdit-v2 EditFlow student inference (no scoring).")
    p.add_argument("--meta_json", type=Path, default=DEFAULT_META)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model_path", type=str, default=DEFAULT_KONTEXT)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_gpus", type=int, default=int(os.environ.get("NUM_GPUS", "8")))
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument(
        "--require_alpha",
        action="store_true",
        help="Require proj_out_alpha in the checkpoint (alpha student).",
    )
    p.add_argument(
        "--no_case_bundles",
        action="store_true",
        help="Only write GEdit_v2/{stem}.jpg for scoring; skip cases/{type}/{stem}/{src,pred,prompt}.",
    )
    p.add_argument(
        "--dump_mixture_stats",
        action="store_true",
        help="Record per-NFE-step mixture weights (pi) to mixture_stats/{stem}.json "
        "for dominant_k-vs-edit_type ARI analysis.",
    )
    return p.parse_args()


def _build_model(
    config_path: Path,
    ckpt_path: Path,
    device: str,
    require_alpha: bool,
    dump_mixture_stats: bool = False,
):
    if require_alpha:
        model = build_alpha_vis_model(config_path, ckpt_path, device)
    else:
        model = build_student_model(config_path, ckpt_path, device)
    if dump_mixture_stats:
        install_mixture_stats_hook(model)
    return model


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


def process_tasks(
    model,
    tasks: List[Tuple[str, Dict]],
    output_dir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    skip_existing: bool,
    device: str,
    desc: str,
    write_case_bundles: bool = True,
    dump_mixture_stats: bool = False,
) -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    for stem, item in tqdm(tasks, desc=desc):
        out_path = pred_path(output_dir, item)
        bundle = case_dir(output_dir, item, stem)
        case_ready = (not write_case_bundles) or (bundle / "pred.png").is_file()
        stats_path = output_dir / "mixture_stats" / f"{stem}.json"
        stats_ready = (not dump_mixture_stats) or stats_path.is_file()
        if skip_existing and out_path.is_file() and case_ready and stats_ready:
            rec = {
                "index": item["index"],
                "source": item["src_image_path"],
                "prompt": item["prompt"],
                "edit_type": item["edit_type"],
                "output": str(out_path),
                "skipped": True,
            }
            if write_case_bundles:
                rec["case_dir"] = str(bundle)
            manifest[stem] = rec
            continue

        src_path = Path(item["src_image_path"])
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_path}")
        image = Image.open(src_path).convert("RGB")
        task_seed = seed + int(item["index"])
        result = run_one_student(
            model,
            image=image,
            prompt=item["prompt"],
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=task_seed,
            device=device,
            dump_mixture_stats=dump_mixture_stats,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.convert("RGB").save(out_path, quality=95)
        rec = {
            "index": item["index"],
            "source": str(src_path),
            "prompt": item["prompt"],
            "edit_type": item["edit_type"],
            "output": str(out_path),
            "skipped": False,
        }
        if dump_mixture_stats:
            steps = get_student_mixture_stats(model)
            if steps:
                write_mixture_stats_files(
                    output_dir, stem, steps,
                    case_dir=bundle if write_case_bundles else None)
                rec["mixture_dominant_k"] = [
                    int(s.get("dominant_k", -1)) for s in steps]
        if write_case_bundles:
            write_case_bundle(bundle, image, result, item["prompt"])
            rec["case_dir"] = str(bundle)
        manifest[stem] = rec
    return manifest


def _infer_worker(gpu_id: int, tasks: List[Tuple[str, Dict]], worker_cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"
    model = _build_model(
        Path(worker_cfg["config"]),
        Path(worker_cfg["ckpt"]),
        device,
        worker_cfg.get("require_alpha", False),
        dump_mixture_stats=worker_cfg.get("dump_mixture_stats", False),
    )
    manifest = process_tasks(
        model,
        tasks,
        Path(worker_cfg["output_dir"]),
        worker_cfg["num_steps"],
        worker_cfg["guidance_scale"],
        worker_cfg["seed"],
        worker_cfg["skip_existing"],
        device,
        desc=f"GPU {gpu_id}",
        write_case_bundles=worker_cfg.get("write_case_bundles", True),
        dump_mixture_stats=worker_cfg.get("dump_mixture_stats", False),
    )
    part_path = Path(worker_cfg["output_dir"]) / f"manifest.gpu{gpu_id}.json"
    part_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.meta_json.is_file():
        raise FileNotFoundError(f"GEdit meta not found: {args.meta_json}")

    class _Settings:
        role = "student"
        run_name = args.run_name
        num_inference_steps = args.num_inference_steps
        guidance_scale = args.guidance_scale

    num_steps, guidance_scale = resolve_inference_settings(_Settings())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_gedit_tasks(args.meta_json)
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]
    print(f"[gedit] {len(tasks)} samples from {args.meta_json}")

    gpu_ids = resolve_gpu_ids(args.num_gpus)
    use_parallel = len(gpu_ids) > 1 and not args.cpu_offload
    worker_cfg = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "output_dir": str(output_dir),
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "seed": args.seed,
        "skip_existing": args.skip_existing,
        "require_alpha": args.require_alpha,
        "write_case_bundles": not args.no_case_bundles,
        "dump_mixture_stats": args.dump_mixture_stats,
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
        model = _build_model(
            args.config, args.ckpt, args.device, args.require_alpha,
            dump_mixture_stats=args.dump_mixture_stats)
        manifest = process_tasks(
            model,
            tasks,
            output_dir,
            num_steps,
            guidance_scale,
            args.seed,
            args.skip_existing,
            args.device,
            desc="GEdit-v2 student",
            write_case_bundles=not args.no_case_bundles,
            dump_mixture_stats=args.dump_mixture_stats,
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "meta_json": str(args.meta_json),
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "run_name": args.run_name,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "num_tasks": len(manifest),
        "num_gpus": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "student_resize_mode": os.environ.get("STUDENT_RESIZE_MODE", "kontext"),
        "output_dir": str(output_dir),
        "gedit_flat_dir": str(output_dir / "GEdit_v2"),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(manifest)} GEdit results -> {output_dir}")
    print(f"Flat preds (for later scoring) -> {output_dir / 'GEdit_v2'}")


if __name__ == "__main__":
    main()
