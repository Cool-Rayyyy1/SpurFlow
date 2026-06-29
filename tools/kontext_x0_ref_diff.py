#!/usr/bin/env python3
# Copyright (c) 2026 ArcFlow contributors
"""
FLUX Kontext editing: per-step x0_hat vs source and vs edited target.

At each denoise step we predict x0_hat = x_t - sigma * u and report:
  - diff vs source  (reference / input image)
  - diff vs edit    (ground-truth edited image)

Two metrics:
  - latent: L1/MSE in VAE latent space
  - image:  L1/MSE after VAE decode (pixel space)

25 steps -> 50 primary values per sample (L1_vs_src and L1_vs_edit per step).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _patch_transformers_version_check() -> None:
    try:
        import transformers.utils.versions as tv

        def _noop(*_args, **_kwargs):
            return None

        tv.require_version_core = _noop
    except Exception:
        pass


_patch_transformers_version_check()

from diffusers import FluxKontextPipeline  # noqa: E402
from diffusers.pipelines.flux.pipeline_flux_kontext import (  # noqa: E402
    PREFERRED_KONTEXT_RESOLUTIONS,
    calculate_shift,
    retrieve_timesteps,
)


@dataclass
class EditSample:
    sample_id: str
    reference_path: str
    prompt: str
    target_path: Optional[str] = None


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def discover_manifest(dataset_dir: Path) -> Optional[Path]:
    names = [
        "manifest.jsonl",
        "kontext_x0_diff_10.jsonl",
        "sft_with_local_source_image_path.jsonl",
        "metadata.jsonl",
        "meta.jsonl",
        "train.jsonl",
        "editing.jsonl",
        "prompts.jsonl",
    ]
    for name in names:
        p = dataset_dir / name
        if p.is_file():
            return p
    jsonls = sorted(dataset_dir.glob("*.jsonl"))
    return jsonls[0] if jsonls else None


def _resolve_path(base: Path, raw: str) -> str:
    p = Path(raw)
    if p.is_absolute() and p.is_file():
        return str(p)
    for cand in [base / raw, base / "images" / raw, base / "source" / raw, base / "reference" / raw]:
        if cand.is_file():
            return str(cand)
    return str(base / raw)


def _resolve_target_path(
    dataset_dir: Path,
    edited_images_dir: Optional[Path],
    raw: str,
) -> str:
    p = Path(raw)
    if p.is_absolute() and p.is_file():
        return str(p)
    if edited_images_dir is not None:
        for cand in [edited_images_dir / raw, edited_images_dir / Path(raw).name]:
            if cand.is_file():
                return str(cand)
    return _resolve_path(dataset_dir, raw)


def load_samples_from_manifest(
    manifest: Path,
    dataset_dir: Path,
    edited_images_dir: Optional[Path],
    num_samples: int,
    seed: int,
) -> List[EditSample]:
    rows: List[Dict[str, Any]] = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rng = random.Random(seed)
    if len(rows) > num_samples:
        rows = rng.sample(rows, num_samples)
    else:
        rows = rows[:num_samples]

    samples: List[EditSample] = []
    for i, row in enumerate(rows):
        ref = (
            row.get("local_input_image")
            or row.get("reference")
            or row.get("reference_image")
            or row.get("source_image")
            or row.get("source")
            or row.get("image")
            or row.get("input_image")
        )
        prompt = row.get("text") or row.get("prompt") or row.get("instruction") or row.get("edit_prompt") or row.get("caption")
        if ref is None or prompt is None:
            raise ValueError(f"manifest row missing reference/prompt: keys={list(row.keys())}")
        sid = str(row.get("id", row.get("sample_id", i)))
        tgt_raw = row.get("target") or row.get("target_image") or row.get("edited_image") or row.get("output_image")
        tgt = _resolve_target_path(dataset_dir, edited_images_dir, tgt_raw) if tgt_raw else None
        samples.append(
            EditSample(
                sample_id=sid,
                reference_path=_resolve_path(dataset_dir, ref) if not Path(ref).is_absolute() else str(ref),
                prompt=str(prompt),
                target_path=tgt,
            )
        )
    return samples


def load_samples_from_dirs(dataset_dir: Path, num_samples: int, seed: int) -> List[EditSample]:
    img_dirs = []
    for name in ("reference", "source", "src", "images", "input"):
        d = dataset_dir / name
        if d.is_dir():
            img_dirs.append(d)
    if not img_dirs:
        raise FileNotFoundError(f"No image subdir under {dataset_dir}")

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    pairs: List[Tuple[Path, Path]] = []
    for d in img_dirs:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in exts:
                txt = p.with_suffix(".txt")
                if txt.is_file():
                    pairs.append((p, txt))

    if not pairs:
        raise FileNotFoundError(f"No image+.txt pairs under {dataset_dir}")

    rng = random.Random(seed)
    pairs = rng.sample(pairs, min(num_samples, len(pairs)))
    return [
        EditSample(sample_id=p.stem, reference_path=str(p), prompt=t.read_text(encoding="utf-8").strip())
        for p, t in pairs
    ]


def load_samples(
    dataset_dir: Optional[str],
    manifest: Optional[str],
    edited_images_dir: Optional[str],
    num_samples: int,
    seed: int,
) -> List[EditSample]:
    if dataset_dir is None:
        return []

    root = Path(dataset_dir)
    edited_root = Path(edited_images_dir) if edited_images_dir else None
    if manifest:
        mpath = Path(manifest)
    else:
        mpath = discover_manifest(root)

    if mpath is not None and mpath.is_file():
        return load_samples_from_manifest(mpath, root, edited_root, num_samples, seed)
    return load_samples_from_dirs(root, num_samples, seed)


def demo_samples(num_samples: int) -> List[EditSample]:
    teaser = Path("/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev/teaser.png")
    if not teaser.is_file():
        raise FileNotFoundError("No dataset and no teaser.png for --demo")
    prompt = "Add a red hat to the subject, keep the rest unchanged."
    return [
        EditSample(sample_id=f"demo_{i}", reference_path=str(teaser), prompt=prompt)
        for i in range(num_samples)
    ]


def _pixel_to_latent_hw(pipe: FluxKontextPipeline, pixel_height: int, pixel_width: int) -> Tuple[int, int]:
    """Match FluxKontextPipeline.prepare_latents rounding (pixel -> latent grid)."""
    h = 2 * (int(pixel_height) // (pipe.vae_scale_factor * 2))
    w = 2 * (int(pixel_width) // (pipe.vae_scale_factor * 2))
    return h, w


@torch.no_grad()
def encode_image_latent(
    pipe: FluxKontextPipeline,
    image: Image.Image,
    pixel_height: int,
    pixel_width: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Encode RGB image to unpacked VAE latent (B,C,H,W) at generation pixel resolution."""
    multiple_of = pipe.vae_scale_factor * 2
    pixel_height = pixel_height // multiple_of * multiple_of
    pixel_width = pixel_width // multiple_of * multiple_of
    image = pipe.image_processor.resize(image, pixel_height, pixel_width)
    image_t = pipe.image_processor.preprocess(image, pixel_height, pixel_width)
    image_t = image_t.to(device=pipe._execution_device, dtype=pipe.vae.dtype)
    lat = pipe._encode_vae_image(image_t, generator=generator)
    return lat.float()


@torch.no_grad()
def compute_dual_step_metrics(
    pipe: FluxKontextPipeline,
    x0_hat_packed: torch.Tensor,
    ref_latent: torch.Tensor,
    edit_latent: torch.Tensor,
    height: int,
    width: int,
    metric: str,
) -> Dict[str, float]:
    x0 = pipe._unpack_latents(x0_hat_packed, height, width, pipe.vae_scale_factor)

    if metric == "latent":
        diff_src = x0 - ref_latent
        diff_edit = x0 - edit_latent
        return {
            "l1_vs_src": diff_src.abs().mean().item(),
            "mse_vs_src": (diff_src ** 2).mean().item(),
            "l1_vs_edit": diff_edit.abs().mean().item(),
            "mse_vs_edit": (diff_edit ** 2).mean().item(),
        }

    x0_dec = (x0 / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(x0_dec.to(pipe.vae.dtype), return_dict=False)[0].float()
    ref_dec = (ref_latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    ref_img = pipe.vae.decode(ref_dec.to(pipe.vae.dtype), return_dict=False)[0].float()
    edit_dec = (edit_latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    edit_img = pipe.vae.decode(edit_dec.to(pipe.vae.dtype), return_dict=False)[0].float()

    diff_src = img - ref_img
    diff_edit = img - edit_img
    return {
        "l1_vs_src": diff_src.abs().mean().item(),
        "mse_vs_src": (diff_src ** 2).mean().item(),
        "l1_vs_edit": diff_edit.abs().mean().item(),
        "mse_vs_edit": (diff_edit ** 2).mean().item(),
    }


@torch.no_grad()
def run_kontext_x0_diff(
    pipe: FluxKontextPipeline,
    image: Image.Image,
    edit_image: Image.Image,
    prompt: str,
    metric: str,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    max_area: int = 1024 ** 2,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    height = width = pipe.default_sample_size * pipe.vae_scale_factor
    multiple_of = pipe.vae_scale_factor * 2

    img = image
    image_height, image_width = pipe.image_processor.get_default_height_width(img)
    aspect_ratio = image_width / image_height
    _, image_width, image_height = min(
        (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
    )
    image_width = image_width // multiple_of * multiple_of
    image_height = image_height // multiple_of * multiple_of

    width = round((max_area * aspect_ratio) ** 0.5)
    height = round((max_area / aspect_ratio) ** 0.5)
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    image = pipe.image_processor.resize(img, image_height, image_width)
    image = pipe.image_processor.preprocess(image, image_height, image_width)

    device = pipe._execution_device
    batch_size = 1

    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
    )

    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = pipe.prepare_latents(
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )
    if image_ids is not None:
        latent_ids = torch.cat([latent_ids, image_ids], dim=0)

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
    )

    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    ref_latent = encode_image_latent(pipe, img, height, width, generator)
    edit_latent = encode_image_latent(pipe, edit_image, height, width, generator)

    pipe.scheduler.set_begin_index(0)
    step_metrics: List[Dict[str, float]] = []

    for i, t in enumerate(timesteps):
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, : latents.size(1)]

        sigma_idx = pipe.scheduler.step_index
        if sigma_idx is None:
            pipe.scheduler._init_step_index(t)
            sigma_idx = pipe.scheduler.step_index
        sigma = pipe.scheduler.sigmas[sigma_idx].to(latents.device, latents.dtype)

        x0_hat_packed = latents - sigma * noise_pred
        m = compute_dual_step_metrics(pipe, x0_hat_packed, ref_latent, edit_latent, height, width, metric)
        m["step"] = float(i + 1)
        m["sigma"] = float(sigma.item())
        m["timestep"] = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
        step_metrics.append(m)

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    keys = ["l1_vs_src", "l1_vs_edit", "mse_vs_src", "mse_vs_edit"]
    avg = {k: float(np.mean([s[k] for s in step_metrics])) for k in keys}
    return step_metrics, avg


def print_sample_report(
    idx: int,
    total: int,
    sample: EditSample,
    metric: str,
    num_steps: int,
    step_metrics: List[Dict[str, float]],
    sample_avg: Dict[str, float],
) -> None:
    tag = "PIXEL (decoded)" if metric == "image" else "LATENT"
    print("")
    print("=" * 88)
    print(f" SAMPLE {idx + 1} / {total}  |  metric={tag}  |  steps={num_steps}  |  id={sample.sample_id}")
    print("=" * 88)
    print(f"  source (x_src) : {sample.reference_path}")
    print(f"  target (x_edit): {sample.target_path}")
    print(f"  instruction    : {sample.prompt}")
    print("-" * 88)
    print("  Per-step x0_hat differences (50 values = 25 x L1_vs_src + 25 x L1_vs_edit):")
    print(f"  {'step':>4}  {'sigma':>9}  {'L1(x0-src)':>14}  {'L1(x0-edit)':>14}  {'MSE(x0-src)':>14}  {'MSE(x0-edit)':>14}")
    for s in step_metrics:
        print(
            f"  {int(s['step']):>4}  {s['sigma']:>9.5f}  "
            f"{s['l1_vs_src']:>14.8f}  {s['l1_vs_edit']:>14.8f}  "
            f"{s['mse_vs_src']:>14.8f}  {s['mse_vs_edit']:>14.8f}"
        )
    print("-" * 88)
    print("  Sample averages over 25 steps:")
    print(f"    mean L1(x0 - x_src)  = {sample_avg['l1_vs_src']:.8f}")
    print(f"    mean L1(x0 - x_edit) = {sample_avg['l1_vs_edit']:.8f}")
    print(f"    mean MSE(x0 - x_src)  = {sample_avg['mse_vs_src']:.8f}")
    print(f"    mean MSE(x0 - x_edit) = {sample_avg['mse_vs_edit']:.8f}")
    print("=" * 88)


def print_global_report(
    metric: str,
    all_step_metrics: List[List[Dict[str, float]]],
    all_avgs: List[Dict[str, float]],
    num_steps: int,
) -> None:
    tag = "PIXEL (decoded)" if metric == "image" else "LATENT"
    n = len(all_avgs)
    print("")
    print("#" * 88)
    print(f"# GLOBAL SUMMARY  |  metric={tag}  |  n_samples={n}  |  steps={num_steps}")
    print("#" * 88)

    for label, key in [("L1(x0 - x_src)", "l1_vs_src"), ("L1(x0 - x_edit)", "l1_vs_edit")]:
        print(f"\n  Per-step mean {label} across {n} samples:")
        for step_i in range(num_steps):
            vals = [sm[step_i][key] for sm in all_step_metrics]
            print(f"    step {step_i + 1:>2}: {float(np.mean(vals)):.8f}  (std={float(np.std(vals)):.8f})")

    print("\n  Overall sample-mean statistics:")
    for key, label in [
        ("l1_vs_src", "L1(x0 - x_src)"),
        ("l1_vs_edit", "L1(x0 - x_edit)"),
        ("mse_vs_src", "MSE(x0 - x_src)"),
        ("mse_vs_edit", "MSE(x0 - x_edit)"),
    ]:
        vals = [a[key] for a in all_avgs]
        print(f"    {label:18s}  mean={float(np.mean(vals)):.8f}  std={float(np.std(vals)):.8f}  "
              f"min={float(np.min(vals)):.8f}  max={float(np.max(vals)):.8f}")
    print("#" * 88)
    print("")


def parse_args():
    p = argparse.ArgumentParser(description="Kontext 25-step x0_hat vs src/edit diff")
    p.add_argument("--metric", choices=("latent", "image"), required=True)
    p.add_argument("--model-path", default="/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev")
    p.add_argument("--dataset-dir", default="/mnt/afs_zhangyunzhe/dataset/pico-banana-400k")
    p.add_argument("--manifest", default=None, help="jsonl manifest (auto-detect if omitted)")
    p.add_argument(
        "--edited-images-dir",
        default="/mnt/afs_zhangyunzhe/dataset/pico-banana-400k/edited_images",
        help="Root dir for output_image paths in pico-banana jsonl",
    )
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--num-inference-steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo", action="store_true", help="Use teaser.png when dataset missing")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--output-json", default=None, help="Optional path to save all metrics as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    samples = load_samples(
        args.dataset_dir,
        args.manifest,
        args.edited_images_dir,
        args.num_samples,
        args.seed,
    )
    if not samples:
        if args.demo or not Path(args.dataset_dir).exists():
            print(f"[warn] No dataset at {args.dataset_dir}; using --demo teaser.", file=sys.stderr)
            samples = demo_samples(args.num_samples)
        else:
            raise FileNotFoundError(
                f"No samples under {args.dataset_dir}. Add manifest jsonl or pass --demo."
            )

    dtype = getattr(torch, args.dtype)
    print(f"Loading Kontext from {args.model_path} ...")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.to("cuda")

    metric = args.metric
    all_avgs: List[Dict[str, float]] = []
    all_step_metrics: List[List[Dict[str, float]]] = []
    all_records: List[Dict[str, Any]] = []

    tag = "PIXEL" if metric == "image" else "LATENT"
    print(
        f"\n>>> metric={tag}  steps={args.num_inference_steps}  "
        f"samples={len(samples)}  (50 values/sample = 25 L1_vs_src + 25 L1_vs_edit)\n"
    )

    for i, sample in enumerate(samples):
        if not Path(sample.reference_path).is_file():
            print(f"[skip] missing source: {sample.reference_path}", file=sys.stderr)
            continue
        if sample.target_path is None or not Path(sample.target_path).is_file():
            print(f"[skip] missing edit target: {sample.target_path}", file=sys.stderr)
            continue

        gen = torch.Generator(device="cuda").manual_seed(args.seed + i)
        src_image = _load_image(sample.reference_path)
        edit_image = _load_image(sample.target_path)
        step_metrics, sample_avg = run_kontext_x0_diff(
            pipe,
            src_image,
            edit_image,
            sample.prompt,
            metric=metric,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
        )
        print_sample_report(
            i, len(samples), sample, metric, args.num_inference_steps, step_metrics, sample_avg
        )
        all_avgs.append(sample_avg)
        all_step_metrics.append(step_metrics)
        all_records.append(
            {
                "sample_id": sample.sample_id,
                "reference": sample.reference_path,
                "target": sample.target_path,
                "prompt": sample.prompt,
                "steps": step_metrics,
                "sample_avg": sample_avg,
            }
        )

    if not all_avgs:
        raise RuntimeError("No samples were processed successfully.")

    print_global_report(metric, all_step_metrics, all_avgs, args.num_inference_steps)

    if args.output_json:
        out = {
            "metric": metric,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "samples": all_records,
            "global_avg": {
                k: float(np.mean([a[k] for a in all_avgs]))
                for k in ("l1_vs_src", "l1_vs_edit", "mse_vs_src", "mse_vs_edit")
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
