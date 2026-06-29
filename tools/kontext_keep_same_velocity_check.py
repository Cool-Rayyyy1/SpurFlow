#!/usr/bin/env python3
"""
Random pico-banana samples + FLUX Kontext with prompt "keep the same" (28 NFE).

Records per-step velocity u (= transformer noise_pred) and writes a txt report
to check whether the model predicts ~zero velocity under this instruction.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

EDITFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))

from tools.kontext_x0_ref_diff import (  # noqa: E402
    EditSample,
    _load_image,
    discover_manifest,
    load_samples_from_manifest,
)


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

DEFAULT_MODEL = "/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev"
DEFAULT_DATASET = "/mnt/afs_zhangyunzhe/dataset/pico-banana-400k"
DEFAULT_PROMPT = "keep the same"
ZERO_EPS = 1e-3


@dataclass
class VelocityStats:
    step: int
    sigma: float
    timestep: float
    mean_abs: float
    max_abs: float
    l2_norm: float
    is_near_zero: bool


def velocity_stats(noise_pred: torch.Tensor, eps: float) -> Dict[str, float | bool]:
    u = noise_pred.float()
    mean_abs = u.abs().mean().item()
    max_abs = u.abs().max().item()
    l2_norm = u.norm().item()
    return {
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "l2_norm": l2_norm,
        "is_near_zero": max_abs < eps,
    }


@torch.no_grad()
def run_velocity_probe(
    pipe: FluxKontextPipeline,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    zero_eps: float,
    max_area: int = 1024 ** 2,
) -> Tuple[List[VelocityStats], torch.Tensor]:
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

    pipe.scheduler.set_begin_index(0)
    step_records: List[VelocityStats] = []

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

        stats = velocity_stats(noise_pred, zero_eps)
        step_records.append(
            VelocityStats(
                step=i + 1,
                sigma=float(sigma.item()),
                timestep=float(t.item()) if isinstance(t, torch.Tensor) else float(t),
                mean_abs=float(stats["mean_abs"]),
                max_abs=float(stats["max_abs"]),
                l2_norm=float(stats["l2_norm"]),
                is_near_zero=bool(stats["is_near_zero"]),
            )
        )

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return step_records, latents


@torch.no_grad()
def decode_latents_to_image(
    pipe: FluxKontextPipeline,
    latents: torch.Tensor,
    height: int,
    width: int,
) -> Image.Image:
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image


def pixel_diff(a: Image.Image, b: Image.Image) -> Dict[str, float]:
    a_np = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    b_np = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    h = min(a_np.shape[0], b_np.shape[0])
    w = min(a_np.shape[1], b_np.shape[1])
    diff = a_np[:h, :w] - b_np[:h, :w]
    return {
        "l1": float(np.abs(diff).mean()),
        "mse": float((diff ** 2).mean()),
        "max_abs": float(np.abs(diff).max()),
    }


def load_random_samples(
    dataset_dir: Path,
    manifest: Optional[Path],
    num_samples: int,
    seed: int,
) -> List[EditSample]:
    if manifest is None:
        manifest = discover_manifest(dataset_dir)
    if manifest is None:
        raise FileNotFoundError(f"No jsonl manifest found under {dataset_dir}")

    jsonl_path = manifest
    if manifest.suffix == ".txt":
        first = manifest.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        jsonl_path = Path(first) if Path(first).is_absolute() else dataset_dir / first

    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(seed)
    picked = rng.sample(rows, min(num_samples, len(rows)))

    samples: List[EditSample] = []
    for i, row in enumerate(picked):
        ref = row.get("local_input_image")
        if ref is None:
            raise ValueError(f"Row {i} missing local_input_image: keys={list(row.keys())}")
        sid = str(row.get("id", row.get("sample_id", Path(ref).stem)))
        samples.append(
            EditSample(
                sample_id=sid,
                reference_path=str(ref),
                prompt=DEFAULT_PROMPT,
                target_path=None,
            )
        )
    return samples


def format_report(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("FLUX Kontext velocity check — prompt: keep the same")
    lines.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"model: {args.model_path}")
    lines.append(f"num_inference_steps: {args.num_inference_steps}")
    lines.append(f"guidance_scale: {args.guidance_scale}")
    lines.append(f"prompt: {args.prompt!r}")
    lines.append(f"zero_eps (max_abs < eps => near-zero velocity): {args.zero_eps}")
    lines.append(f"num_samples: {len(records)}")
    lines.append("")

    all_near_zero = True
    for rec in records:
        lines.append("=" * 88)
        lines.append(f"sample_id: {rec['sample_id']}")
        lines.append(f"source: {rec['source']}")
        lines.append(f"output: {rec['output_image']}")
        lines.append("-" * 88)
        lines.append(
            f"{'step':>4}  {'sigma':>9}  {'mean|u|':>12}  {'max|u|':>12}  "
            f"{'||u||_2':>12}  {'near_zero':>9}"
        )
        sample_all_zero = True
        for s in rec["velocity_steps"]:
            lines.append(
                f"{s['step']:>4}  {s['sigma']:>9.5f}  {s['mean_abs']:>12.6e}  "
                f"{s['max_abs']:>12.6e}  {s['l2_norm']:>12.6e}  "
                f"{'YES' if s['is_near_zero'] else 'NO':>9}"
            )
            sample_all_zero = sample_all_zero and s["is_near_zero"]
        lines.append("-" * 88)
        lines.append(
            f"all_steps_near_zero: {'YES' if sample_all_zero else 'NO'}  "
            f"(max|u| across steps = {rec['max_velocity']:.6e})"
        )
        pd = rec["pixel_diff_vs_source"]
        lines.append(
            f"output vs source pixel diff: L1={pd['l1']:.8f}  "
            f"MSE={pd['mse']:.8f}  max_abs={pd['max_abs']:.8f}"
        )
        lines.append("")
        all_near_zero = all_near_zero and sample_all_zero

    lines.append("#" * 88)
    lines.append("GLOBAL CONCLUSION")
    lines.append("#" * 88)
    if all_near_zero:
        lines.append(
            f"All {len(records)} samples: velocity u is near-zero at every step "
            f"(max|u| < {args.zero_eps})."
        )
    else:
        lines.append(
            f"Velocity u is NOT zero for prompt {args.prompt!r}: at least one step "
            f"has max|u| >= {args.zero_eps}."
        )
        worst = max(records, key=lambda r: r["max_velocity"])
        lines.append(
            f"Largest velocity: sample_id={worst['sample_id']}  "
            f"max|u|={worst['max_velocity']:.6e}"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kontext 28NFE with 'keep the same' — check if velocity u is zero."
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    p.add_argument(
        "--manifest",
        default=None,
        help="jsonl manifest with local_input_image (default: auto-detect under dataset-dir/jsonl/)",
    )
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--num-inference-steps", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--zero-eps", type=float, default=ZERO_EPS)
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument(
        "--out-dir",
        default=str(EDITFLOW_ROOT / "outputs" / "kontext_keep_same_velocity"),
    )
    p.add_argument(
        "--report-txt",
        default=None,
        help="Path for txt report (default: <out-dir>/velocity_report.txt)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset_dir = Path(args.dataset_dir)
    manifest = Path(args.manifest) if args.manifest else dataset_dir / "jsonl" / "sft_with_local_source_image_path.jsonl"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_txt) if args.report_txt else out_dir / "velocity_report.txt"

    samples = load_random_samples(dataset_dir, manifest, args.num_samples, args.seed)
    print(f"Loaded {len(samples)} random samples from {manifest}")

    dtype = getattr(torch, args.dtype)
    print(f"Loading Kontext from {args.model_path} ...")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.to("cuda")

    all_records: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples):
        if not Path(sample.reference_path).is_file():
            print(f"[skip] missing source: {sample.reference_path}", file=sys.stderr)
            continue

        gen = torch.Generator(device="cuda").manual_seed(args.seed + i)
        src_image = _load_image(sample.reference_path)

        step_records, final_latents = run_velocity_probe(
            pipe,
            src_image,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
            zero_eps=args.zero_eps,
        )

        sample_dir = out_dir / f"{i:03d}_{sample.sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        src_path = sample_dir / "source.png"
        out_path = sample_dir / f"kontext_{args.num_inference_steps}nfe.png"
        src_image.save(src_path)

        # Decode final latents for visual check
        multiple_of = pipe.vae_scale_factor * 2
        image_height, image_width = pipe.image_processor.get_default_height_width(src_image)
        aspect_ratio = image_width / image_height
        max_area = 1024 ** 2
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        out_image = decode_latents_to_image(pipe, final_latents, height, width)
        out_image.save(out_path)

        step_dicts = [
            {
                "step": s.step,
                "sigma": s.sigma,
                "timestep": s.timestep,
                "mean_abs": s.mean_abs,
                "max_abs": s.max_abs,
                "l2_norm": s.l2_norm,
                "is_near_zero": s.is_near_zero,
            }
            for s in step_records
        ]
        max_velocity = max(s.max_abs for s in step_records)
        rec = {
            "sample_id": sample.sample_id,
            "source": str(src_path),
            "output_image": str(out_path),
            "velocity_steps": step_dicts,
            "max_velocity": max_velocity,
            "pixel_diff_vs_source": pixel_diff(src_image, out_image),
        }
        all_records.append(rec)
        print(
            f"[{i + 1}/{len(samples)}] {sample.sample_id}: "
            f"max|u|={max_velocity:.6e}  "
            f"pixel_L1={rec['pixel_diff_vs_source']['l1']:.6f}"
        )

    if not all_records:
        raise RuntimeError("No samples processed.")

    report = format_report(args, all_records)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {report_path}")
    print(report)


if __name__ == "__main__":
    main()
