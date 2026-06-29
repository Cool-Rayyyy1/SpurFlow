#!/usr/bin/env python3
"""Single-image delta_x visualization for the 2-NFE gmkontext_uedit_fixedeps student.

Runs one local image + one prompt through the official 2-step sampler and dumps, into
a single folder:

    0_src.png            preprocessed source image (model input)
    1_edited.png         final edited image (val_step output, decode of final latent)
    2_deltax_step1.png   model deltax at sampling step 1 (clip[-1,1] -> decode -> (x+1)/2*255)
    3_deltax_step2.png   model deltax at sampling step 2 (same mapping)
    4_x0_step1.png       predicted clean edited image at step 1: decode(x_ref + pred_delta_1)
    5_src_noised_step1.png  img2img-style forward noised src at step1 entry sigma:
                            decode((1-σ)*x_ref + σ*noise) via sample_forward_diffusion
    6_xt_after_step1.png model x_t after step1 integration (actual sampler trajectory)
    7_step2_start.png    step2 input: decode(epsilon - v_step1), v_step1 = step1 displacement
    noise.pt             the sampled latent noise (== path_epsilon), shape (1, C, H, W)
    noise_vis.png        per-channel min-max normalized RGB view of the first 3 latent channels
    tensors.pt           dict of raw latents: noise / pred_delta_step{1,2} / x_ref /
                         x0_step1_latent / final_latent
    prompt.txt           the edit prompt

delta_x = the model's mixture pred_delta (~ x0_tgt - x_ref). The official sampling loop is
left unchanged; we only wrap ``momentum_integration`` (called once per NFE step) to record
the policy's ``compute_pred_delta`` and ``x_ref`` at each step.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("STUDENT_RESIZE_MODE", "kontext")

import numpy as np
import torch
from PIL import Image

EVAL_ROOT = Path(__file__).resolve().parent
EDITFLOW_ROOT = EVAL_ROOT.parents[1]
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))

from run_editflow_imgedit_infer import (  # noqa: E402
    build_student_model,
    pil_to_tensor,
    preprocess_image_for_student,
    tensor_to_pil,
)

DEFAULT_CONFIG = EDITFLOW_ROOT / "configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data.py"
DEFAULT_CKPT = (EDITFLOW_ROOT
                / "checkpoints/gmkontext_uedit_fixedeps_k16_2nfe_pico400k"
                / "20260618_055623/iter_20000.pth")
DEFAULT_IMAGE = EDITFLOW_ROOT / "pikachu.png"
DEFAULT_PROMPT = (
    "Put an Ash Ketchum-style Pokemon Trainer cap on Pikachu, with a red-and-white "
    "design and a green emblem on the front.")
DEFAULT_OUTPUT = EVAL_ROOT / "outputs/deltax_vis/pikachu_single"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--nfe", type=int, default=2)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bg_color", type=str, default="white",
                   help="Background to composite transparent (RGBA) inputs onto: "
                        "'white', 'black', or 'r,g,b'.")
    return p.parse_args()


def parse_bg_color(spec: str) -> tuple:
    spec = spec.strip().lower()
    named = {"white": (255, 255, 255), "black": (0, 0, 0)}
    if spec in named:
        return named[spec]
    parts = [int(x) for x in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid --bg_color={spec!r}; use 'white', 'black' or 'r,g,b'.")
    return tuple(parts)


def load_image_on_bg(path: Path, bg_color: tuple) -> Image.Image:
    """Load an image, compositing any transparency onto a solid bg_color -> RGB."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, bg_color + (255,))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def normalize_latent_to_rgb(latent: torch.Tensor) -> Image.Image:
    """Per-channel min-max normalize the first 3 latent channels -> RGB image."""
    x = latent[0, :3].float()
    chans = []
    for c in range(x.size(0)):
        ch = x[c]
        lo, hi = ch.min(), ch.max()
        chans.append((ch - lo) / (hi - lo + 1e-8))
    rgb = torch.stack(chans, dim=0)  # (3, H, W)
    return tensor_to_pil(rgb)


@torch.inference_mode()
def run_full(model, image: Image.Image, prompt: str, nfe: int, guidance: float,
             seed: int, device: str) -> dict:
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
    test_cfg_override = {"nfe": nfe, "distilled_guidance_scale": guidance, "guidance_scale": 1.0}

    noise_patched = model.patchify(noise.to(device))

    diffusion = model.diffusion_ema if model.diffusion_use_ema else model.diffusion
    captured = []
    xt_outputs = []  # x_t produced by each step (last == final latent)
    v_steps = []     # per-step displacement: x_t_start - x_t_end
    sigmas = []      # sigma_t_start at each step (for forward-noise visualization)
    holder = {}
    orig_mi = diffusion.momentum_integration

    def capturing_mi(sigma_t_src, x_t_start, sigma_t_start, raw_t_end, policy,
                     eps=1e-4, seq_len=None):
        pred_delta = policy.compute_pred_delta(sigma_t_src, sigma_t_src)
        captured.append(pred_delta.detach().clone())
        if "x_ref" not in holder:
            holder["x_ref"] = policy.x_ref.detach().clone()
        sigmas.append(sigma_t_start.detach().clone())
        result = orig_mi(sigma_t_src, x_t_start, sigma_t_start, raw_t_end, policy,
                         eps=eps, seq_len=seq_len)
        x_t_end = result[0]
        xt_outputs.append(x_t_end.detach().clone())
        v_step = (x_t_start - x_t_end).detach().clone()
        v_steps.append(v_step)
        if len(captured) == 1:
            holder["v_step1"] = v_step
            holder["u_step1"] = policy.velocity(sigma_t_src, sigma_t_src).detach().clone()
            if policy.path_epsilon is not None:
                holder["path_epsilon"] = policy.path_epsilon.detach().clone()
        return result

    diffusion.momentum_integration = capturing_mi
    try:
        outputs = model.val_step(data, test_cfg_override=test_cfg_override)
    finally:
        diffusion.momentum_integration = orig_mi

    edited_pil = tensor_to_pil(outputs["pred_imgs"][0])

    # delta_x visualization: clip[-1,1] in latent space -> decode -> ((x+1)/2)*255
    delta_pils = []
    for pred_delta in captured:
        delta_lat = model.unpatchify(pred_delta.to(device)).clamp(-1.0, 1.0).to(vae_dtype)
        dec = model.vae.decode(delta_lat).float()
        delta_pils.append(tensor_to_pil(dec[0] / 2 + 0.5))

    def decode_latent(patched: torch.Tensor) -> Image.Image:
        lat = model.unpatchify(patched.to(device)).to(vae_dtype)
        dec = model.vae.decode(lat).float()
        return tensor_to_pil(dec[0] / 2 + 0.5)

    def decode_v_patched(v_patched: torch.Tensor) -> Image.Image:
        """Visualize displacement v: clip[-1,1] in latent -> decode -> (x+1)/2*255."""
        v_lat = model.unpatchify(v_patched.to(device)).clamp(-1.0, 1.0).to(vae_dtype)
        dec = model.vae.decode(v_lat).float()
        return tensor_to_pil(dec[0] / 2 + 0.5)

    # Predicted clean edited latent at step 1: x0 = x_ref + pred_delta_step1 (normal decode).
    x_ref = holder["x_ref"]
    x0_step1_patched = x_ref + captured[0]
    x0_step1_pil = decode_latent(x0_step1_patched)

    # img2img-style: forward-noise the source latent to step1-end sigma (raw_t after
    # the first NFE segment). Step1-entry sigma is ~1.0 and would decode as pure noise;
    # step1-end sigma keeps src structure visible for PPT.
    if len(sigmas) >= 2:
        sigma_for_noised_src = sigmas[1]  # step2 input == step1 output sigma
    else:
        eps = 1e-4
        timestep_ratio = max(test_cfg_override.get("timestep_ratio", 1.0), eps)
        base_segment_size = 1 / (nfe - 1 + timestep_ratio)
        raw_t_end = 1.0 - base_segment_size
        seq_len = x_ref.shape[2:].numel()
        sigma_for_noised_src = diffusion.timestep_sampler.warp_t(
            torch.tensor([raw_t_end], device=device, dtype=torch.float32),
            seq_len=seq_len,
        ).reshape(1, *([1] * (x_ref.dim() - 1)))

    t_noised = sigma_for_noised_src.flatten() * diffusion.num_timesteps
    src_noised_patched, _, std_noised = diffusion.sample_forward_diffusion(
        x_ref, t_noised, noise_patched)
    src_noised_step1_pil = decode_latent(src_noised_patched)

    xt_after_step1_pil = decode_latent(xt_outputs[0]) if len(xt_outputs) >= 1 else None

    # Step2 start: epsilon - v_step1 (still noisy; NOT the clean x0 prediction).
    path_epsilon = holder.get("path_epsilon", noise_patched)
    v_step1 = holder["v_step1"]
    step2_start_patched = path_epsilon - v_step1
    # Should match xt_outputs[0] (step1 integration output == step2 input).
    if len(xt_outputs) >= 1:
        max_diff = (step2_start_patched - xt_outputs[0]).abs().max().item()
        if max_diff > 1e-3:
            print(f"[warn] step2_start != xt_after_step1, max_diff={max_diff:.6f}", flush=True)
    step2_start_pil = decode_latent(step2_start_patched)

    v_pils = [decode_v_patched(v) for v in v_steps]

    tensors = {
        "noise": noise.detach().cpu(),
        "noise_patched": noise_patched.detach().cpu(),
        "path_epsilon": path_epsilon.detach().cpu(),
        "x_ref": x_ref.detach().cpu(),
        "v_step1": v_step1.detach().cpu(),
        "u_step1": holder.get("u_step1", torch.empty(0)).detach().cpu(),
        "step2_start_latent": step2_start_patched.detach().cpu(),
        "x0_step1_latent": x0_step1_patched.detach().cpu(),
        "src_noised_step1_latent": src_noised_patched.detach().cpu(),
        "sigma_step1_entry": sigmas[0].detach().cpu(),
        "sigma_step1_end": sigma_for_noised_src.detach().cpu(),
        "std_step1_end": std_noised.detach().cpu(),
    }
    for i, pd in enumerate(captured, start=1):
        tensors[f"pred_delta_step{i}"] = pd.detach().cpu()
    for i, xt in enumerate(xt_outputs, start=1):
        tensors[f"xt_step{i}_output_latent"] = xt.detach().cpu()
    for i, sig in enumerate(sigmas, start=1):
        tensors[f"sigma_step{i}"] = sig.detach().cpu()
    for i, v in enumerate(v_steps, start=1):
        tensors[f"v_step{i}"] = v.detach().cpu()

    return dict(
        src=src_pil,
        edited=edited_pil,
        deltas=delta_pils,
        vs=v_pils,
        x0_step1=x0_step1_pil,
        src_noised_step1=src_noised_step1_pil,
        xt_after_step1=xt_after_step1_pil,
        step2_start=step2_start_pil,
        noise=noise.detach().cpu(),
        sigma_step1_entry=float(sigmas[0].mean().item()) if sigmas else 0.0,
        sigma_step1_end=float(sigma_for_noised_src.mean().item()),
        std_step1_end=float(std_noised.mean().item()),
        tensors=tensors,
    )


def main() -> None:
    args = parse_args()
    for f in (args.config, args.ckpt, args.image):
        if not Path(f).is_file():
            raise FileNotFoundError(f"Not found: {f}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[deltax-single] image={args.image}")
    print(f"[deltax-single] prompt={args.prompt}")
    print(f"[deltax-single] ckpt={args.ckpt}")
    print(f"[deltax-single] output={args.output_dir}", flush=True)

    model = build_student_model(args.config, args.ckpt, device)
    bg_color = parse_bg_color(args.bg_color)
    print(f"[deltax-single] bg_color={bg_color} (transparent areas composited onto this)",
          flush=True)
    image = load_image_on_bg(args.image, bg_color)
    out = run_full(model, image, args.prompt, args.nfe, args.guidance, args.seed, device)

    out["src"].save(args.output_dir / "0_src.png")
    out["edited"].save(args.output_dir / "1_edited.png")
    for i, dp in enumerate(out["deltas"], start=1):
        dp.save(args.output_dir / f"{i + 1}_deltax_step{i}.png")
    out["x0_step1"].save(args.output_dir / "4_x0_step1.png")
    out["src_noised_step1"].save(args.output_dir / "5_src_noised_step1.png")
    if out["xt_after_step1"] is not None:
        out["xt_after_step1"].save(args.output_dir / "6_xt_after_step1.png")
    if out.get("step2_start") is not None:
        out["step2_start"].save(args.output_dir / "7_step2_start.png")
    for i, vp in enumerate(out.get("vs", []), start=1):
        vp.save(args.output_dir / f"{7 + i}_v_step{i}.png")

    torch.save(out["noise"], args.output_dir / "noise.pt")
    normalize_latent_to_rgb(out["noise"]).save(args.output_dir / "noise_vis.png")
    torch.save(out["tensors"], args.output_dir / "tensors.pt")
    (args.output_dir / "prompt.txt").write_text(
        f"image: {args.image}\nprompt: {args.prompt}\n"
        f"nfe: {args.nfe}\nguidance: {args.guidance}\nseed: {args.seed}\n"
        f"bg_color: {args.bg_color}\n"
        f"sigma_step1_entry: {out['sigma_step1_entry']:.6f}\n"
        f"sigma_step1_end: {out['sigma_step1_end']:.6f}\n"
        f"std_step1_end: {out['std_step1_end']:.6f}\n"
        f"5_src_noised_step1.png: decode((1-std)*x_ref + std*noise) at step1-end sigma\n"
        f"6_xt_after_step1.png: actual model x_t after step1 integration\n"
        f"7_step2_start.png: decode(epsilon - v_step1), step2 input (still noisy)\n"
        f"8_v_step1.png / 9_v_step2.png: decode(clip(v_step{i},[-1,1])), v = x_t_start - x_t_end\n"
        f"  x0_step1 (4_*) = x_ref + pred_delta_1 is the clean prediction, not step2 start\n",
        encoding="utf-8")

    print(f"\n[deltax-single] Done -> {args.output_dir}")
    for p in sorted(args.output_dir.iterdir()):
        print(f"   {p.name}")


if __name__ == "__main__":
    main()
