#!/usr/bin/env python3
"""Load the student once and dump deltax viz for multiple seeds."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("STUDENT_RESIZE_MODE", "kontext")

import torch

EDITFLOW_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = EDITFLOW_ROOT / "evaluation" / "imgedit_bench"
sys.path.insert(0, str(EDITFLOW_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from run_editflow_imgedit_infer import build_student_model  # noqa: E402
from visualize_deltax_single import (  # noqa: E402
    load_image_on_bg,
    normalize_latent_to_rgb,
    parse_bg_color,
    run_full,
)


def save_one(out_dir: Path, out: dict, meta: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out["src"].save(out_dir / "0_src.png")
    out["edited"].save(out_dir / "1_edited.png")
    for i, dp in enumerate(out["deltas"], start=1):
        dp.save(out_dir / f"{i + 1}_deltax_step{i}.png")
    out["x0_step1"].save(out_dir / "4_x0_step1.png")
    out["src_noised_step1"].save(out_dir / "5_src_noised_step1.png")
    if out["xt_after_step1"] is not None:
        out["xt_after_step1"].save(out_dir / "6_xt_after_step1.png")
    if out.get("step2_start") is not None:
        out["step2_start"].save(out_dir / "7_step2_start.png")
    for i, vp in enumerate(out.get("vs", []), start=1):
        vp.save(out_dir / f"{7 + i}_v_step{i}.png")
    torch.save(out["noise"], out_dir / "noise.pt")
    normalize_latent_to_rgb(out["noise"]).save(out_dir / "noise_vis.png")
    torch.save(out["tensors"], out_dir / "tensors.pt")
    (out_dir / "prompt.txt").write_text(meta, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--nfe", type=int, default=2)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bg_color", type=str, default="white")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"
    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"[multiseed] loading model once...", flush=True)
    model = build_student_model(args.config, args.ckpt, device)
    bg = parse_bg_color(args.bg_color)
    image = load_image_on_bg(args.image, bg)

    for seed in args.seeds:
        out_dir = args.output_root / f"seed_{seed:04d}"
        print(f"[multiseed] seed={seed} -> {out_dir}", flush=True)
        out = run_full(model, image, args.prompt, args.nfe, args.guidance, seed, device)
        meta = (
            f"image: {args.image}\nprompt: {args.prompt}\n"
            f"nfe: {args.nfe}\nguidance: {args.guidance}\nseed: {seed}\n"
            f"bg_color: {args.bg_color}\n"
            f"ckpt: {args.ckpt}\n"
            f"sigma_step1_entry: {out['sigma_step1_entry']:.6f}\n"
            f"sigma_step1_end: {out['sigma_step1_end']:.6f}\n"
        )
        save_one(out_dir, out, meta)
        # also drop a quick preview name at root for easy browsing
        out["edited"].save(args.output_root / f"preview_seed_{seed:04d}_edited.png")
        print(f"[multiseed] done seed={seed}", flush=True)

    print(f"[multiseed] all done -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
