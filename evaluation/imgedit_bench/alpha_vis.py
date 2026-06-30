"""Render per-patch alpha annotations for ArcFlowEditNewAlpha inference."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


VAE_SCALE = 8
PATCH_LATENT = 2
PATCH_IMAGE = VAE_SCALE * PATCH_LATENT  # 16 px per transformer patch on image


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fmt_alpha_value(value: float) -> str:
    """Compact label: ~3 significant digits, no long tail decimals."""
    if math.isnan(value):
        return "-"
    av = abs(value)
    if av >= 100:
        return f"{value:.0f}"
    if av >= 10:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if av >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if av >= 0.1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if av < 1e-3:
        return "0"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _normalize_alpha_tensor(alpha: torch.Tensor) -> torch.Tensor:
    """Return (H_lat, W_lat) float alpha map."""
    x = alpha.detach().float().cpu()
    while x.dim() > 2:
        x = x.squeeze(0)
    if x.dim() != 2:
        raise ValueError(f"Expected 2D alpha map after squeeze, got shape {tuple(alpha.shape)}")
    return x


def patch_alpha_means(
    alpha_latent: torch.Tensor,
    img_height: int,
    img_width: int,
    patch_image: int = PATCH_IMAGE,
    vae_scale: int = VAE_SCALE,
) -> Tuple[np.ndarray, int, int]:
    """Average alpha over each transformer patch aligned to the source image grid."""
    alpha = _normalize_alpha_tensor(alpha_latent)
    h_lat, w_lat = alpha.shape
    patch_latent = patch_image // vae_scale

    n_ph = img_height // patch_image
    n_pw = img_width // patch_image
    if n_ph <= 0 or n_pw <= 0:
        raise ValueError(
            f"Image too small for patch grid: ({img_width}, {img_height}), patch={patch_image}")

    grid = np.zeros((n_ph, n_pw), dtype=np.float32)
    for pi in range(n_ph):
        for pj in range(n_pw):
            y0 = pi * patch_latent
            x0 = pj * patch_latent
            y1 = min(y0 + patch_latent, h_lat)
            x1 = min(x0 + patch_latent, w_lat)
            if y0 >= h_lat or x0 >= w_lat:
                grid[pi, pj] = float("nan")
                continue
            grid[pi, pj] = alpha[y0:y1, x0:x1].mean().item()
    return grid, n_ph, n_pw


def render_alpha_grid_blank(
    img_width: int,
    img_height: int,
    alpha_latent: torch.Tensor,
    *,
    step_label: Optional[str] = None,
    patch_image: int = PATCH_IMAGE,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Blank canvas + grid + per-patch alpha labels (1+head, same as integration)."""
    grid, n_ph, n_pw = patch_alpha_means(alpha_latent, img_height, img_width, patch_image=patch_image)

    valid = grid[~np.isnan(grid)]
    if valid.size == 0:
        vmin, vmax = 0.0, 2.0
    else:
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))

    canvas = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    grid_color = (180, 180, 180)
    for pi in range(n_ph + 1):
        y = pi * patch_image
        draw.line([(0, y), (img_width - 1, y)], fill=grid_color, width=1)
    for pj in range(n_pw + 1):
        x = pj * patch_image
        draw.line([(x, 0), (x, img_height - 1)], fill=grid_color, width=1)

    font_size = max(10, min(20, patch_image // 3))
    font = _load_font(font_size)

    for pi in range(n_ph):
        for pj in range(n_pw):
            val = grid[pi, pj]
            label = fmt_alpha_value(val)
            x0 = pj * patch_image
            y0 = pi * patch_image
            cx = x0 + patch_image // 2
            cy = y0 + patch_image // 2
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = cx - tw // 2
            ty = cy - th // 2
            draw.text((tx, ty), label, fill=(20, 20, 20), font=font)

    header_h = 32
    out = Image.new("RGB", (img_width, img_height + header_h), bg_color)
    out.paste(canvas, (0, header_h))
    header = ImageDraw.Draw(out)
    title = f"alpha (1+head) per {patch_image}px cell"
    if step_label:
        title = f"{title}  {step_label}"
    header.text((8, 6), title, fill=(0, 0, 0), font=_load_font(14))
    legend = (
        f"grid {n_ph}x{n_pw}   "
        f"min {fmt_alpha_value(vmin)}  max {fmt_alpha_value(vmax)}  "
        f"mean {fmt_alpha_value(float(np.nanmean(grid)))}   init=1"
    )
    header.text((8, 20), legend, fill=(90, 90, 90), font=_load_font(10))
    return out


@torch.inference_mode()
def capture_student_alphas(
    model,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
    preprocess_fn,
    pil_to_tensor_fn,
    return_edited: bool = False,
    tensor_to_pil_fn=None,
):
    """Run val_step and return (preprocessed source PIL, alpha tensors per NFE step).

    If ``return_edited`` is True, also returns the decoded edited PIL image.
    """
    src_pil = preprocess_fn(image)
    source = pil_to_tensor_fn(src_pil).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    if hasattr(model.vae, "dtype"):
        vae_dtype = model.vae.dtype
    else:
        vae_dtype = next(model.vae.parameters()).dtype
    with torch.no_grad():
        latents = model.vae.encode((source * 2 - 1).to(vae_dtype)).float()
        noise = torch.randn(latents.shape, generator=gen, device=device, dtype=latents.dtype)

    data = {
        "prompt_kwargs": {"prompt": [prompt]},
        "source_images": source,
        "noise": noise,
    }
    test_cfg_override = {
        "nfe": num_inference_steps,
        "distilled_guidance_scale": guidance_scale,
        "guidance_scale": 1.0,
    }

    diffusion = model.diffusion_ema if model.diffusion_use_ema else model.diffusion
    captured_alphas: List[torch.Tensor] = []
    orig_mi = diffusion.momentum_integration

    def capturing_mi(
        sigma_t_src,
        x_t_start,
        sigma_t_start,
        raw_t_end,
        policy,
        eps=1e-4,
        seq_len=None,
    ):
        if hasattr(policy, "alpha"):
            alpha = policy.alpha
            if alpha.dim() == 5:
                alpha = alpha.squeeze(2)
            captured_alphas.append(alpha.detach().clone())
        return orig_mi(
            sigma_t_src,
            x_t_start,
            sigma_t_start,
            raw_t_end,
            policy,
            eps=eps,
            seq_len=seq_len,
        )

    diffusion.momentum_integration = capturing_mi
    try:
        outputs = model.val_step(data, test_cfg_override=test_cfg_override)
    finally:
        diffusion.momentum_integration = orig_mi

    if return_edited:
        if tensor_to_pil_fn is None:
            raise ValueError("tensor_to_pil_fn is required when return_edited=True")
        edited_pil = tensor_to_pil_fn(outputs["pred_imgs"][0])
        return src_pil, captured_alphas, edited_pil
    return src_pil, captured_alphas
