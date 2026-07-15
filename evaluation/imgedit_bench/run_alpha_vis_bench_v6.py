#!/usr/bin/env python3
"""Alpha v6: per-case folder with src/edit/prompt + step1/step2 continuous alpha maps.

Layout (same as v5):
  <output_dir>/<category>/<case_key>/
    prompt.txt
    src.png
    edit.png
    step1_alpha.png      # blank grid + per-patch alpha values (no src)
    step1_heatmap.png
    step1_overlay.png
    step2_alpha.png
    step2_heatmap.png
    step2_overlay.png

Unlike v5 (binary α∈{0,1} red/green), v6 visualizes continuous sigmoid alpha with
diverging colormap heatmap/overlay on source.

Default: all cases in each of the 9 basic categories (737 total).
Use --samples_per_category N for a quick preview (seeded random subset per category).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
EDITFLOW_ROOT = EVAL_ROOT.parents[1]
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))

from alpha_vis import (  # noqa: E402
    capture_student_alphas,
    render_alpha_grid_blank,
    render_continuous_alpha_on_src,
)
from alpha_model_utils import build_alpha_vis_model  # noqa: E402
from run_editflow_imgedit_infer import (  # noqa: E402
    DEFAULT_BENCH_ROOT,
    merge_manifest_parts,
    pil_to_tensor,
    preprocess_image_for_student,
    resolve_gpu_ids,
    split_tasks_round_robin,
    tensor_to_pil,
)

DEFAULT_CATEGORIES = (
    "action",
    "add",
    "adjust",
    "background",
    "compose",
    "extract",
    "remove",
    "replace",
    "style",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Alpha v6: 9 categories, all imedit_bench cases, continuous alpha maps.")
    p.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT)
    p.add_argument(
        "--annotations_path",
        type=Path,
        default=EVAL_ROOT / "annotations" / "basic_edit.json",
    )
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--num_inference_steps", type=int, default=2)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--samples_per_category",
        type=int,
        default=None,
        help="If set, randomly sample this many cases per category (preview mode). "
        "If omitted, run every case in each category.",
    )
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_gpus", type=int, default=int(os.environ.get("NUM_GPUS", "1")))
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument(
        "--heatmap_blend",
        type=float,
        default=float(os.environ.get("HEATMAP_BLEND", "0.36")),
        help="Continuous colormap tint strength for heatmap (src still visible).",
    )
    p.add_argument(
        "--overlay_blend",
        type=float,
        default=float(os.environ.get("OVERLAY_BLEND", "0.52")),
        help="Slightly stronger continuous colormap tint for overlay.",
    )
    return p.parse_args()


def load_category_examples(
    annotations_path: Path,
    bench_root: Path,
    samples_per_category: Optional[int],
    seed: int,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
) -> List[Dict]:
    raw = json.loads(annotations_path.read_text(encoding="utf-8"))
    by_cat: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
    for key, item in raw.items():
        cat = str(item.get("edit_type", "")).lower()
        if cat in categories:
            by_cat[cat].append((key, item))

    rng = random.Random(seed)
    samples: List[Dict] = []
    for cat in categories:
        pool = sorted(by_cat.get(cat, []), key=lambda x: x[0])
        if not pool:
            continue
        if samples_per_category is None or len(pool) <= samples_per_category:
            chosen = pool
        else:
            chosen = sorted(rng.sample(pool, samples_per_category), key=lambda x: x[0])
        for key, item in chosen:
            src_rel = item["id"]
            samples.append(dict(
                task_key=f"basic:{key}",
                key=key,
                category=cat,
                example_name=str(key),
                prompt=item["prompt"],
                source_path=str(bench_root / "singleturn" / src_rel),
                source_rel=src_rel,
            ))
    return samples


def v6_case_dir(output_dir: Path, category: str, example_name: str) -> Path:
    return output_dir / category / example_name


def v6_output_paths(case_dir: Path) -> Dict[str, Path]:
    return {
        "prompt": case_dir / "prompt.txt",
        "src": case_dir / "src.png",
        "edit": case_dir / "edit.png",
        "step1_alpha": case_dir / "step1_alpha.png",
        "step1_heatmap": case_dir / "step1_heatmap.png",
        "step1_overlay": case_dir / "step1_overlay.png",
        "step2_alpha": case_dir / "step2_alpha.png",
        "step2_heatmap": case_dir / "step2_heatmap.png",
        "step2_overlay": case_dir / "step2_overlay.png",
    }


def process_samples(
    model,
    samples: List[Dict],
    output_dir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    skip_existing: bool,
    device: str,
    desc: str,
    heatmap_blend: float,
    overlay_blend: float,
) -> Dict[str, Dict]:
    if num_steps != 2:
        raise ValueError(f"Alpha v6 expects nfe=2 so both steps are visualized, got {num_steps}.")

    manifest: Dict[str, Dict] = {}
    for idx, sample in enumerate(tqdm(samples, desc=desc)):
        case_dir = v6_case_dir(output_dir, sample["category"], sample["example_name"])
        paths = v6_output_paths(case_dir)
        image_paths = {k: v for k, v in paths.items() if k != "prompt"}
        if skip_existing and paths["prompt"].is_file() and all(p.is_file() for p in image_paths.values()):
            continue

        src_path = Path(sample["source_path"])
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_path}")

        task_seed = seed + idx
        image = Image.open(src_path).convert("RGB")
        src_pil, alphas, edit_pil = capture_student_alphas(
            model,
            image,
            sample["prompt"],
            num_steps,
            guidance_scale,
            task_seed,
            device,
            preprocess_image_for_student,
            pil_to_tensor,
            return_edited=True,
            tensor_to_pil_fn=tensor_to_pil,
        )
        if len(alphas) < 2:
            raise RuntimeError(
                f"Expected 2 alpha maps for nfe=2, got {len(alphas)} for {sample['task_key']}")

        case_dir.mkdir(parents=True, exist_ok=True)
        paths["prompt"].write_text(sample["prompt"].rstrip() + "\n", encoding="utf-8")
        src_pil.save(paths["src"])
        edit_pil.save(paths["edit"])

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

        manifest[sample["task_key"]] = {
            "category": sample["category"],
            "example_name": sample["example_name"],
            "key": sample["key"],
            "source": str(src_path),
            "prompt": sample["prompt"],
            "case_dir": str(case_dir),
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    return manifest


def _worker(gpu_id: int, samples: List[Dict], worker_cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model = build_alpha_vis_model(
        Path(worker_cfg["config"]),
        Path(worker_cfg["ckpt"]),
        "cuda",
    )
    manifest = process_samples(
        model,
        samples,
        Path(worker_cfg["output_dir"]),
        worker_cfg["num_steps"],
        worker_cfg["guidance_scale"],
        worker_cfg["seed"],
        worker_cfg["skip_existing"],
        "cuda",
        desc=f"GPU {gpu_id}",
        heatmap_blend=worker_cfg["heatmap_blend"],
        overlay_blend=worker_cfg["overlay_blend"],
    )
    part = Path(worker_cfg["output_dir"]) / f"manifest.gpu{gpu_id}.json"
    part.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if not args.annotations_path.is_file():
        raise FileNotFoundError(args.annotations_path)

    num_steps = int(args.num_inference_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_category_examples(
        args.annotations_path,
        args.bench_root,
        args.samples_per_category,
        args.seed,
    )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    gpu_ids = resolve_gpu_ids(args.num_gpus)
    wrapped = [(s["task_key"], s) for s in samples]
    buckets_wrapped = split_tasks_round_robin(wrapped, len(gpu_ids))
    buckets = [[item for _, item in bucket] for bucket in buckets_wrapped]

    worker_cfg = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "output_dir": str(args.output_dir),
        "num_steps": num_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "skip_existing": args.skip_existing,
        "heatmap_blend": args.heatmap_blend,
        "overlay_blend": args.overlay_blend,
    }

    if len(buckets) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        model = build_alpha_vis_model(args.config, args.ckpt, args.device)
        manifest = process_samples(
            model,
            buckets[0],
            args.output_dir,
            num_steps,
            args.guidance_scale,
            args.seed,
            args.skip_existing,
            args.device,
            desc="alpha v6",
            heatmap_blend=args.heatmap_blend,
            overlay_blend=args.overlay_blend,
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
    print(f"Saved v6 packs for {len(manifest)} cases -> {args.output_dir}")
    print("Layout: <category>/<case_key>/{prompt.txt,src,edit,step{1,2}_{alpha,heatmap,overlay}}")


if __name__ == "__main__":
    main()
