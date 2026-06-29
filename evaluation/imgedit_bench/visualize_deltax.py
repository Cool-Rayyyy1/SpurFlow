#!/usr/bin/env python3
"""Visualize the per-step predicted delta_x of the 2-NFE gmkontext_uedit_fixedeps student.

For each ImgEdit-Bench basic edit_type (action, add, adjust, background, compose,
extract, remove, replace, style) we pick the first N examples (default 2) and, for
every example, dump 4 images into the category folder:

    <ex>_0_src.png            preprocessed source image (model input)
    <ex>_1_edited.png         final edited image (val_step output)
    <ex>_2_deltax_step1.png   model deltax at sampling step 1
    <ex>_3_deltax_step2.png   model deltax at sampling step 2

delta_x visualization follows: take the model's mixture pred_delta (~ x0_tgt - x_ref)
at the step, clip it to [-1, 1] in latent space, VAE-decode, then map ((x+1)/2)*255.

The edit prompt of every example is written to <category>/prompts.txt.

This reuses the official sampling loop unchanged: it only wraps
``diffusion.momentum_integration`` (called once per NFE step) to record the policy's
``compute_pred_delta`` at each step.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Match training-time ImageEdit preprocessing before importing the infer module
# (it snapshots STUDENT_RESIZE_MODE at import time).
os.environ.setdefault("STUDENT_RESIZE_MODE", "kontext")

import numpy as np
import torch
from PIL import Image

EVAL_ROOT = Path(__file__).resolve().parent
EDITFLOW_ROOT = EVAL_ROOT.parents[1]
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))

# Reuse the exact preprocessing / tensor helpers used by the normal inference path.
from run_editflow_imgedit_infer import (  # noqa: E402
    build_student_model,
    pil_to_tensor,
    preprocess_image_for_student,
    tensor_to_pil,
)

CATEGORIES = (
    "action", "add", "adjust", "background", "compose",
    "extract", "remove", "replace", "style",
)

DEFAULT_BENCH_ROOT = Path(os.environ.get(
    "IMGEDIT_BENCH_ROOT",
    "/mnt/afs_zhangyunzhe/dataset/imgedit/benchmark/Benchmark"))
DEFAULT_CONFIG = EDITFLOW_ROOT / "configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data.py"
DEFAULT_CKPT = (EDITFLOW_ROOT
                / "checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k"
                / "20260618_055623/iter_20000.pth")
DEFAULT_OUTPUT = EVAL_ROOT / "outputs/deltax_vis/gmkontext_uedit_fixedeps_iter20000"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT)
    p.add_argument("--annotations_dir", type=Path, default=EVAL_ROOT / "annotations")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--num_per_cat", type=int, default=5)
    p.add_argument("--max_tasks", type=int, default=0,
                   help="If >0, cap total tasks (smoke test).")
    p.add_argument("--nfe", type=int, default=2)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=str, default="0,1",
                   help="Comma-separated GPU ids, e.g. '0,1'.")
    return p.parse_args()


def select_tasks(annotations_dir: Path, num_per_cat: int) -> List[Dict]:
    """Pick the first `num_per_cat` examples (sorted by key) for each category."""
    basic = json.loads((annotations_dir / "basic_edit.json").read_text(encoding="utf-8"))
    by_cat: Dict[str, List[Tuple[str, Dict]]] = {c: [] for c in CATEGORIES}
    for key in sorted(basic, key=lambda k: (len(k), k)):
        item = basic[key]
        cat = item.get("edit_type")
        if cat in by_cat and len(by_cat[cat]) < num_per_cat:
            by_cat[cat].append((key, item))

    tasks: List[Dict] = []
    for cat in CATEGORIES:
        for ex_idx, (key, item) in enumerate(by_cat[cat], start=1):
            tasks.append(dict(
                category=cat,
                ex_idx=ex_idx,
                key=key,
                src_id=item["id"],
                prompt=item["prompt"],
            ))
    return tasks


@torch.inference_mode()
def run_one_with_delta(model, image: Image.Image, prompt: str, nfe: int,
                       guidance: float, seed: int, device: str):
    """Return (src_pil, edited_pil, [delta_step1_pil, delta_step2_pil])."""
    src_pil = preprocess_image_for_student(image)
    source = pil_to_tensor(src_pil).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)

    vae_dtype = (model.vae.dtype if hasattr(model.vae, "dtype")
                 else next(model.vae.parameters()).dtype)
    latents = model.vae.encode((source * 2 - 1).to(vae_dtype)).float()
    noise = torch.randn(latents.shape, generator=gen, device=device, dtype=latents.dtype)

    data = {
        "prompt_kwargs": {"prompt": [prompt]},
        "source_images": source,
        "noise": noise,
    }
    test_cfg_override = {
        "nfe": nfe,
        "distilled_guidance_scale": guidance,
        "guidance_scale": 1.0,
    }

    diffusion = model.diffusion_ema if model.diffusion_use_ema else model.diffusion

    captured: List[torch.Tensor] = []
    orig_mi = diffusion.momentum_integration

    def capturing_mi(sigma_t_src, x_t_start, sigma_t_start, raw_t_end, policy,
                     eps=1e-4, seq_len=None):
        # Instantaneous model deltax at the current step (dt_past = 0).
        pred_delta = policy.compute_pred_delta(sigma_t_src, sigma_t_src)
        captured.append(pred_delta.detach().clone())
        return orig_mi(sigma_t_src, x_t_start, sigma_t_start, raw_t_end, policy,
                       eps=eps, seq_len=seq_len)

    diffusion.momentum_integration = capturing_mi
    try:
        outputs = model.val_step(data, test_cfg_override=test_cfg_override)
    finally:
        diffusion.momentum_integration = orig_mi

    edited_pil = tensor_to_pil(outputs["pred_imgs"][0])

    delta_pils: List[Image.Image] = []
    for pred_delta in captured:
        # pred_delta is patchified latent (B, 64, h, w). Unpatchify -> (B, 16, H, W).
        delta_lat = model.unpatchify(pred_delta.to(device))
        delta_lat = delta_lat.clamp(-1.0, 1.0).to(vae_dtype)
        dec = model.vae.decode(delta_lat).float()  # ~[-1, 1]
        delta_pils.append(tensor_to_pil(dec[0] / 2 + 0.5))  # ((x+1)/2) then *255 in helper

    return src_pil, edited_pil, delta_pils


def process_shard(gpu_id: int, tasks: List[Dict], cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"
    model = build_student_model(Path(cfg["config"]), Path(cfg["ckpt"]), device)
    bench_root = Path(cfg["bench_root"])
    output_dir = Path(cfg["output_dir"])

    for task in tasks:
        cat = task["category"]
        ex_idx = task["ex_idx"]
        key = task["key"]
        src_path = bench_root / "singleturn" / task["src_id"]
        if not src_path.is_file():
            print(f"[gpu{gpu_id}][warn] missing source: {src_path}", flush=True)
            continue

        cat_dir = output_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"ex{ex_idx}_{key}"

        task_seed = cfg["seed"] + (int(key) if str(key).isdigit() else 0)
        image = Image.open(src_path).convert("RGB")
        src_pil, edited_pil, delta_pils = run_one_with_delta(
            model, image, task["prompt"], cfg["nfe"], cfg["guidance"],
            task_seed, device)

        src_pil.save(cat_dir / f"{prefix}_0_src.png")
        edited_pil.save(cat_dir / f"{prefix}_1_edited.png")
        for step_idx, dp in enumerate(delta_pils, start=1):
            dp.save(cat_dir / f"{prefix}_{step_idx + 1}_deltax_step{step_idx}.png")

        print(f"[gpu{gpu_id}] {cat} ex{ex_idx} ({key}) done "
              f"-> {len(delta_pils)} delta steps", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def split_round_robin(tasks: List[Dict], n: int) -> List[List[Dict]]:
    buckets: List[List[Dict]] = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        buckets[i % n].append(t)
    return [b for b in buckets if b]


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = select_tasks(args.annotations_dir, args.num_per_cat)
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    print(f"[info] {len(tasks)} tasks across {len(CATEGORIES)} categories", flush=True)

    # Write prompts.txt up front (per category) from the selection, so prompts are
    # always available even if inference later fails/OOMs.
    by_cat: Dict[str, List[Dict]] = {}
    for t in tasks:
        by_cat.setdefault(t["category"], []).append(t)
    for cat, cat_tasks in by_cat.items():
        cat_dir = args.output_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for t in cat_tasks:
            lines.append(f"example {t['ex_idx']} | key={t['key']} | src={t['src_id']}")
            lines.append(f"  prompt: {t['prompt']}")
        (cat_dir / "prompts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpu_ids:
        gpu_ids = [0]

    cfg = dict(
        config=str(args.config),
        ckpt=str(args.ckpt),
        bench_root=str(args.bench_root),
        output_dir=str(args.output_dir),
        nfe=args.nfe,
        guidance=args.guidance,
        seed=args.seed,
    )

    buckets = split_round_robin(tasks, len(gpu_ids))
    if len(buckets) == 1:
        process_shard(gpu_ids[0], buckets[0], cfg)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for gpu_id, bucket in zip(gpu_ids, buckets):
            p = ctx.Process(target=process_shard, args=(gpu_id, bucket, cfg))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker (exit {p.exitcode}) failed")

    print(f"\nDone. Outputs under: {args.output_dir}", flush=True)
    for cat in CATEGORIES:
        print(f"  {args.output_dir / cat}")


if __name__ == "__main__":
    main()
