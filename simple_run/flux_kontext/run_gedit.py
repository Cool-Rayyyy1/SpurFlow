#!/usr/bin/env python3
"""Official FLUX.1-Kontext-dev baseline on GEdit-v2 (generation only).

No student/teacher split. Writes directly under the run folder:

  {output_dir}/GEdit_v2/{stem}.jpg
  {output_dir}/cases/{edit_type}/{stem}/{src.png,pred.png,prompt.txt}

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
GEDIT_ROOT = EDITFLOW_ROOT / "evaluation" / "gedit_v2"
for path in (str(HERE), str(GEDIT_ROOT), str(EDITFLOW_ROOT)):
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
from run_editflow_gedit_infer import (  # noqa: E402
    DEFAULT_META,
    case_dir,
    load_gedit_tasks,
    pred_path,
    write_case_bundle,
)

DEFAULT_RUN_TAG = os.environ.get("RUN_TAG", "flux_kontext_official")
DEFAULT_OUTPUT = GEDIT_ROOT / "outputs" / "runs" / DEFAULT_RUN_TAG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Official FLUX.1-Kontext-dev GEdit-v2 inference (no scoring)."
    )
    p.add_argument("--meta_json", type=Path, default=DEFAULT_META)
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


def _infer_worker(gpu_id: int, tasks: List[Tuple[str, Dict]], cfg: dict) -> None:
    configure_hf_env()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pipe = build_pipeline(cfg["model_path"], "cuda", cfg["cpu_offload"])
    manifest = process_tasks(
        pipe,
        tasks,
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
    if not args.meta_json.is_file():
        raise FileNotFoundError(f"GEdit meta not found: {args.meta_json}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_gedit_tasks(args.meta_json)
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]
    print(
        f"[gedit] {len(tasks)} samples from {args.meta_json}  "
        f"steps={args.num_inference_steps}  guidance={args.guidance_scale}  "
        f"resize=kontext-buckets  model={args.model_path}"
    )
    print(f"[gedit] flat preds -> {output_dir / 'GEdit_v2'}")
    print(f"[gedit] cases     -> {output_dir / 'cases'}")

    gpu_ids = resolve_gpu_ids(args.num_gpus)
    worker_cfg = {
        "model_path": args.model_path,
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
            output_dir,
            args.num_inference_steps,
            args.guidance_scale,
            args.seed,
            args.skip_existing,
            desc="gedit",
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"[gedit] done -> {output_dir}")


if __name__ == "__main__":
    main()
