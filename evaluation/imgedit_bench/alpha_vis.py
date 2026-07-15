"""Render per-patch alpha annotations for ArcFlowEditNewAlpha inference."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


VAE_SCALE = 8
PATCH_LATENT = 2
PATCH_IMAGE = VAE_SCALE * PATCH_LATENT  # 16 px per transformer patch on image

# Panel / typography
HEADER_H = 52
PANEL_BG = (18, 22, 28)
PANEL_LINE = (45, 52, 64)
TEXT_PRIMARY = (241, 245, 249)
TEXT_MUTED = (148, 163, 184)
TEXT_ACCENT = (125, 211, 252)
ALPHA_CENTER = 0.5


def _load_font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")
    )
    for path in (
        f"/usr/share/fonts/truetype/dejavu/{names[0]}",
        f"/usr/share/fonts/truetype/liberation/{names[1]}",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_header(
    canvas: Image.Image,
    *,
    title: str,
    subtitle: str,
    colorbar_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Draw a dark header band with optional horizontal colorbar."""
    w, _ = canvas.size
    header = ImageDraw.Draw(canvas)
    header.rectangle([(0, 0), (w, HEADER_H)], fill=PANEL_BG)
    header.line([(0, HEADER_H - 1), (w, HEADER_H - 1)], fill=PANEL_LINE, width=1)

    title_font = _load_font(15)
    sub_font = _load_font(11, bold=False)
    header.text((14, 10), title, fill=TEXT_PRIMARY, font=title_font)
    header.text((14, 30), subtitle, fill=TEXT_MUTED, font=sub_font)

    if colorbar_range is None:
        return

    vmin, vmax = colorbar_range
    bar_w = min(220, w - 28)
    bar_h = 10
    bar_x0 = w - bar_w - 14
    bar_y = 30
    header.text((bar_x0, 10), "α scale", fill=TEXT_ACCENT, font=sub_font)

    for i in range(bar_w):
        t = i / max(bar_w - 1, 1)
        val = vmin + t * (vmax - vmin)
        c = _alpha_to_rgb_scalar(val, vmin, vmax)
        header.rectangle(
            [(bar_x0 + i, bar_y), (bar_x0 + i, bar_y + bar_h - 1)],
            fill=c,
        )
    header.rectangle(
        [(bar_x0, bar_y), (bar_x0 + bar_w - 1, bar_y + bar_h - 1)],
        outline=PANEL_LINE,
        width=1,
    )
    tick_font = _load_font(9, bold=False)
    ticks = [vmin]
    if vmin <= ALPHA_CENTER <= vmax:
        ticks.append(ALPHA_CENTER)
    ticks.append(vmax)
    for tick_val in ticks:
        t = (tick_val - vmin) / max(vmax - vmin, 1e-6)
        anchor = bar_x0 + int(t * (bar_w - 1))
        label = fmt_alpha_value(tick_val)
        tw, _ = _text_size(header, label, tick_font)
        header.text((anchor - tw // 2, bar_y + bar_h + 2), label, fill=TEXT_MUTED, font=tick_font)


def _alpha_display_range(values: np.ndarray) -> Tuple[float, float]:
    """Data-driven range so local variation stays visible."""
    flat = values[np.isfinite(values)]
    if flat.size == 0:
        return 0.0, 2.0
    vmin = float(np.min(flat))
    vmax = float(np.max(flat))
    span = max(vmax - vmin, 0.04)
    pad = max(span * 0.18, 0.015)
    return vmin - pad, vmax + pad


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rgb(c0: Tuple[int, int, int], c1: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        int(_lerp(c0[0], c1[0], t)),
        int(_lerp(c0[1], c1[1], t)),
        int(_lerp(c0[2], c1[2], t)),
    )


def _alpha_to_rgb_scalar(val: float, vmin: float, vmax: float) -> Tuple[int, int, int]:
    """Diverging around α=1 when in range; otherwise sequential viridis-like."""
    if math.isnan(val):
        return (100, 100, 100)
    if vmax <= vmin:
        vmax = vmin + 1e-3

    if vmin <= ALPHA_CENTER <= vmax:
        half = max(ALPHA_CENTER - vmin, vmax - ALPHA_CENTER, 1e-6)
        t = 0.5 + (val - ALPHA_CENTER) / (2.0 * half)
        t = max(0.0, min(1.0, t))
        stops = (
            (0.00, (49, 54, 149)),
            (0.35, (69, 117, 180)),
            (0.50, (224, 243, 248)),
            (0.65, (253, 174, 97)),
            (1.00, (165, 15, 21)),
        )
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t <= t1:
                u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                return _lerp_rgb(c0, c1, u)
        return stops[-1][1]

    t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    stops = (
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    )
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            return _lerp_rgb(c0, c1, u)
    return stops[-1][1]


def _alpha_colormap_rgb(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    vectorized = np.vectorize(lambda v: _alpha_to_rgb_scalar(float(v), vmin, vmax))
    rgb = np.stack(vectorized(values), axis=-1).astype(np.uint8)
    return rgb


def _compose_with_header(body: Image.Image, **header_kwargs) -> Image.Image:
    w, h = body.size
    out = Image.new("RGB", (w, h + HEADER_H), PANEL_BG)
    out.paste(body, (0, HEADER_H))
    _draw_header(out, **header_kwargs)
    return out


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
    continuous: bool = False,
) -> Image.Image:
    """Blank canvas + grid + per-patch alpha labels (binary {0,1} or continuous sigmoid)."""
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
    alpha_kind = "continuous sigmoid" if continuous else "0/1"
    title = f"alpha ({alpha_kind}) per {patch_image}px cell"
    if step_label:
        title = f"{title}  {step_label}"
    header.text((8, 6), title, fill=(0, 0, 0), font=_load_font(14))
    init_note = "init=0.5" if continuous else "init=1 (on)"
    legend = (
        f"grid {n_ph}x{n_pw}   "
        f"min {fmt_alpha_value(vmin)}  max {fmt_alpha_value(vmax)}  "
        f"mean {fmt_alpha_value(float(np.nanmean(grid)))}   {init_note}"
    )
    header.text((8, 20), legend, fill=(90, 90, 90), font=_load_font(10))
    return out


def upsample_alpha_to_image(
    alpha_latent: torch.Tensor,
    img_height: int,
    img_width: int,
    *,
    smooth: bool = True,
) -> np.ndarray:
    """Upsample latent alpha (binary gate) to image resolution."""
    alpha = _normalize_alpha_tensor(alpha_latent)
    mode = "bilinear" if smooth else "nearest"
    up = F.interpolate(
        alpha.unsqueeze(0).unsqueeze(0),
        size=(img_height, img_width),
        mode=mode,
        align_corners=False if smooth else None,
    )[0, 0]
    return up.numpy()


def _patch_grid_overlay(img: Image.Image) -> Image.Image:
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    n_ph = h // PATCH_IMAGE
    n_pw = w // PATCH_IMAGE
    for pi in range(n_ph + 1):
        y = pi * PATCH_IMAGE
        odraw.line([(0, y), (w - 1, y)], fill=(255, 255, 255, 28), width=1)
    for pj in range(n_pw + 1):
        x = pj * PATCH_IMAGE
        odraw.line([(x, 0), (x, h - 1)], fill=(255, 255, 255, 28), width=1)
    base = img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_alpha_heatmap(
    img_width: int,
    img_height: int,
    alpha_latent: torch.Tensor,
    *,
    step_label: Optional[str] = None,
) -> Image.Image:
    """Smooth alpha heatmap with patch grid and colorbar."""
    alpha_up = upsample_alpha_to_image(alpha_latent, img_height, img_width, smooth=True)
    vmin, vmax = _alpha_display_range(alpha_up)
    rgb = _alpha_colormap_rgb(alpha_up, vmin, vmax)
    canvas = Image.fromarray(rgb, mode="RGB")
    canvas = _patch_grid_overlay(canvas)

    title = "Alpha heatmap  (softmax binary)"
    if step_label:
        title = f"{title}  ·  {step_label}"
    subtitle = (
        f"smooth upsample · init=1 (on) · "
        f"min {fmt_alpha_value(float(np.min(alpha_up)))}  "
        f"max {fmt_alpha_value(float(np.max(alpha_up)))}  "
        f"mean {fmt_alpha_value(float(np.mean(alpha_up)))}"
    )
    return _compose_with_header(canvas, title=title, subtitle=subtitle, colorbar_range=(vmin, vmax))


# Soft binary colors for alpha∈{0,1} visualizations (v5).
ALPHA0_RGB = (220, 60, 60)    # red  = drop x_ref (edit freely)
ALPHA1_RGB = (40, 170, 90)    # green = keep x_ref


def render_continuous_alpha_on_src(
    src_pil: Image.Image,
    alpha_latent: torch.Tensor,
    *,
    blend: float = 0.42,
    step_label: Optional[str] = None,
    title_prefix: str = "Alpha heatmap",
    draw_grid: bool = True,
) -> Image.Image:
    """Continuous sigmoid α colormap tint over source so the image stays visible."""
    src = np.array(src_pil.convert("RGB"), dtype=np.float32)
    img_h, img_w = src.shape[:2]
    alpha_up = upsample_alpha_to_image(alpha_latent, img_h, img_w, smooth=True)
    vmin, vmax = _alpha_display_range(alpha_up)
    heat_rgb = _alpha_colormap_rgb(alpha_up, vmin, vmax).astype(np.float32)

    out = src * (1.0 - blend) + heat_rgb * blend
    out = np.clip(out, 0, 255).astype(np.uint8)
    body = Image.fromarray(out, mode="RGB")
    if draw_grid:
        body = _patch_grid_overlay(body)

    title = f"{title_prefix}  (continuous α)"
    if step_label:
        title = f"{title}  ·  {step_label}"
    subtitle = (
        f"min {fmt_alpha_value(float(np.min(alpha_up)))}  "
        f"max {fmt_alpha_value(float(np.max(alpha_up)))}  "
        f"mean {fmt_alpha_value(float(np.mean(alpha_up)))}  ·  "
        f"blend={blend:.2f}"
    )
    return _compose_with_header(
        body, title=title, subtitle=subtitle, colorbar_range=(vmin, vmax))


def render_binary_alpha_on_src(
    src_pil: Image.Image,
    alpha_latent: torch.Tensor,
    *,
    blend: float = 0.38,
    step_label: Optional[str] = None,
    title_prefix: str = "Alpha heatmap",
    draw_grid: bool = True,
) -> Image.Image:
    """Soft red(α=0) / green(α=1) tint over source so the image stays visible."""
    src = np.array(src_pil.convert("RGB"), dtype=np.float32)
    img_h, img_w = src.shape[:2]
    alpha_up = upsample_alpha_to_image(alpha_latent, img_h, img_w, smooth=False)
    # Hard binary for display (model already outputs {0,1}; nearest upsample keeps edges).
    alpha_bin = (alpha_up >= 0.5).astype(np.float32)

    color = np.zeros_like(src)
    color[..., 0] = ALPHA0_RGB[0] * (1.0 - alpha_bin) + ALPHA1_RGB[0] * alpha_bin
    color[..., 1] = ALPHA0_RGB[1] * (1.0 - alpha_bin) + ALPHA1_RGB[1] * alpha_bin
    color[..., 2] = ALPHA0_RGB[2] * (1.0 - alpha_bin) + ALPHA1_RGB[2] * alpha_bin

    out = src * (1.0 - blend) + color * blend
    out = np.clip(out, 0, 255).astype(np.uint8)
    body = Image.fromarray(out, mode="RGB")
    if draw_grid:
        body = _patch_grid_overlay(body)

    frac1 = float(alpha_bin.mean())
    frac0 = 1.0 - frac1
    title = f"{title_prefix}  (α∈{{0,1}})"
    if step_label:
        title = f"{title}  ·  {step_label}"
    subtitle = (
        f"red=α0 (edit) {frac0 * 100:.1f}%  ·  green=α1 (keep ref) {frac1 * 100:.1f}%  ·  "
        f"blend={blend:.2f}"
    )
    return _compose_with_header(body, title=title, subtitle=subtitle, colorbar_range=None)


def render_src_alpha_low_overlay(
    src_pil: Image.Image,
    alpha_latent: torch.Tensor,
    *,
    threshold: float = 0.1,
    step_label: Optional[str] = None,
) -> Image.Image:
    """Source + semi-transparent alpha heatmap; highlight α below threshold."""
    src = np.array(src_pil.convert("RGB"), dtype=np.float32)
    img_h, img_w = src.shape[:2]
    alpha_up = upsample_alpha_to_image(alpha_latent, img_h, img_w, smooth=True)
    vmin, vmax = _alpha_display_range(alpha_up)
    heat_rgb = _alpha_colormap_rgb(alpha_up, vmin, vmax).astype(np.float32)

    # Base blend: heatmap over source
    heat_blend = 0.42
    out = src * (1.0 - heat_blend) + heat_rgb * heat_blend

    # Emphasize low-alpha regions with a soft crimson veil
    effective_thr = threshold
    mask = alpha_up < effective_thr
    if mask.mean() < 0.005:
        # adaptive: bottom 20% of values when fixed threshold hits nothing
        effective_thr = float(np.percentile(alpha_up, 20))
        mask = alpha_up <= effective_thr

    strength = np.zeros_like(alpha_up, dtype=np.float32)
    denom = max(effective_thr, 1e-6)
    strength[mask] = np.clip((effective_thr - alpha_up[mask]) / denom, 0.0, 1.0)
    strength = strength[..., None]
    accent = np.array([255.0, 70.0, 110.0], dtype=np.float32)
    accent_blend = 0.38
    out = out * (1.0 - strength * accent_blend) + accent * strength * accent_blend
    out = np.clip(out, 0, 255).astype(np.uint8)

    body = Image.fromarray(out, mode="RGB")
    body = _patch_grid_overlay(body)

    n_low = int(mask.sum())
    pct = 100.0 * n_low / max(mask.size, 1)
    title = f"Alpha on source  (softmax binary)"
    if step_label:
        title = f"{title}  ·  {step_label}"
    subtitle = (
        f"heatmap blend · pink = lowest α (≤ {fmt_alpha_value(effective_thr)}) · "
        f"highlight {pct:.1f}% pixels"
    )
    return _compose_with_header(body, title=title, subtitle=subtitle, colorbar_range=(vmin, vmax))


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
