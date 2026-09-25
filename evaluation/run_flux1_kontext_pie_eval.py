#!/usr/bin/env python3
"""Run the official FLUX.1-Kontext-dev pipeline on PIE-Bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import FluxKontextPipeline
from PIL import Image
from tqdm import tqdm


DEFAULT_MODEL = Path("/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev")
DEFAULT_DATA = Path(
    "/mnt/afs_gaochengmin/projects/zhangyunzhe/mask_data/PIE-Bench_unique_mask"
)
DEFAULT_OUTPUT = Path(
    "evaluation/pie_bench/outputs/runs/"
    "flux1_kontext_dev_28step_cfg2.5_official"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    mapping_path = args.data_root / "mapping_file.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    tasks = list(mapping.items())
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]

    edit_dir = args.output_dir / "edit_png"
    edit_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to("cuda")

    for index, (sample_id, item) in enumerate(tqdm(tasks, desc="PIE FLUX.1-Kontext")):
        source_path = args.data_root / "annotation_images" / item["image_path"]
        output_path = edit_dir / f"{sample_id}.png"
        if args.skip_existing and valid_image(output_path):
            continue

        source = Image.open(source_path).convert("RGB")
        prompt = item["editing_instruction"]
        result = pipe(
            image=source,
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed + index),
        ).images[0]
        result.convert("RGB").save(output_path)
        manifest[sample_id] = {
            "index": index,
            "source": str(source_path),
            "prompt": prompt,
            "edit_type_id": str(item.get("editing_type_id", "")),
            "output": str(output_path.resolve()),
            "seed": args.seed + index,
            "size": list(result.size),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    run_meta = {
        "model_path": str(args.model_path),
        "data_root": str(args.data_root),
        "num_tasks": len(tasks),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "scheduler": "native FLUX.1-Kontext scheduler with dynamic shifting",
        "seed": args.seed,
        "output_dir": str(edit_dir.resolve()),
        "completed": sum(valid_image(edit_dir / f"{sample_id}.png") for sample_id, _ in tasks),
    }
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done: {run_meta['completed']}/{len(tasks)} -> {edit_dir.resolve()}")


if __name__ == "__main__":
    main()
