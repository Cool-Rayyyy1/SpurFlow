"""Visualize step-2 alpha mask crops for DINO GAN (edit_mass heatmap + crop boxes)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lakonlab.models.architecture.dino_feature_discriminator import (
    sample_dino_alpha_mask_crop_specs,
    upsample_alpha_to_image,
)

from alpha_vis import (  # type: ignore
    _compose_with_header,
    _patch_grid_overlay,
)


CROP_COLORS = {
    'global': (80, 220, 120),
    'random_local': (70, 150, 255),
    'mask_local': (255, 90, 70),
}


def compute_edit_mass_map(
        alpha_map: torch.Tensor,
        *,
        edit_is_low_alpha: bool = True,
        mass_threshold_percentile: float = 30.0) -> np.ndarray:
    """Return edit mass (H, W) used by mask-guided crop (low alpha = edit when True)."""
    alpha_np = alpha_map.detach().float().cpu().numpy()
    if edit_is_low_alpha:
        ref = float(np.percentile(alpha_np, mass_threshold_percentile))
        edit_mass = np.clip(ref - alpha_np, 0.0, None)
    else:
        ref = float(np.percentile(alpha_np, 100.0 - mass_threshold_percentile))
        edit_mass = np.clip(alpha_np - ref, 0.0, None)
    return edit_mass.astype(np.float32)


def _edit_mass_to_rgb(values: np.ndarray) -> np.ndarray:
    """Warm heatmap: dark blue -> yellow -> red for rising edit mass."""
    flat = values[np.isfinite(values)]
    vmax = float(flat.max()) if flat.size else 1.0
    vmax = max(vmax, 1e-8)
    t = np.clip(values / vmax, 0.0, 1.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(255.0 * np.maximum(0.0, (t - 0.35) / 0.65), 0, 255)
    rgb[..., 1] = np.clip(255.0 * np.minimum(1.0, t * 1.4), 0, 255)
    rgb[..., 2] = np.clip(255.0 * (1.0 - t) * 0.85, 0, 255)
    return rgb.astype(np.uint8)


def render_edit_mass_heatmap(
        src_pil: Image.Image,
        alpha_latent: torch.Tensor,
        *,
        edit_is_low_alpha: bool = True,
        mass_threshold_percentile: float = 30.0,
        blend: float = 0.45,
        step_label: str = 'step2',
) -> Image.Image:
    """Edit-mass heatmap overlay (high = predicted edit region)."""
    src = np.array(src_pil.convert('RGB'), dtype=np.float32)
    img_h, img_w = src.shape[:2]
    alpha_img = upsample_alpha_to_image(
        alpha_latent, img_h, img_w, smooth=True, smooth_sigma=2.0)
    edit_mass = compute_edit_mass_map(
        alpha_img[0] if alpha_img.dim() == 3 else alpha_img,
        edit_is_low_alpha=edit_is_low_alpha,
        mass_threshold_percentile=mass_threshold_percentile,
    )
    heat_rgb = _edit_mass_to_rgb(edit_mass).astype(np.float32)
    out = src * (1.0 - blend) + heat_rgb * blend
    out = np.clip(out, 0, 255).astype(np.uint8)
    body = Image.fromarray(out, mode='RGB')
    body = _patch_grid_overlay(body)
    total = float(edit_mass.sum())
    frac = 100.0 * float((edit_mass > edit_mass.max() * 0.05).mean()) if edit_mass.size else 0.0
    title = f'Edit-mass heatmap  ·  {step_label}'
    subtitle = (
        f'low α → high edit mass · ref=p{mass_threshold_percentile:.0f} · '
        f'sum={total:.4f} · active≈{frac:.1f}% pixels'
    )
    return _compose_with_header(
        body, title=title, subtitle=subtitle, colorbar_range=(0.0, float(edit_mass.max())))


def _spec_to_pixel_box(
        spec: Sequence[float],
        width: int,
        height: int) -> Tuple[int, int, int, int]:
    top, left, crop_h, crop_w = [float(v) for v in spec]
    y0 = min(max(int(round(top * height)), 0), height - 1)
    x0 = min(max(int(round(left * width)), 0), width - 1)
    y1 = min(max(int(round((top + crop_h) * height)), y0 + 1), height)
    x1 = min(max(int(round((left + crop_w) * width)), x0 + 1), width)
    return x0, y0, x1, y1


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_crop_boxes_overlay(
        src_pil: Image.Image,
        crop_specs: List[torch.Tensor],
        *,
        batch_idx: int = 0,
        mask_fallback: bool = False,
        local_enabled: bool = True,
        title: str = 'GAN crop boxes (shared real/fake)',
) -> Image.Image:
    """Draw global + second crop (random-local or mask-local per sample mode)."""
    canvas = src_pil.convert('RGB').copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    font = _load_font(13)

    if len(crop_specs) >= 1:
        x0, y0, x1, y1 = _spec_to_pixel_box(crop_specs[0][batch_idx].tolist(), width, height)
        color = CROP_COLORS['global']
        for offset in (0, 1):
            draw.rectangle(
                [(x0 - offset, y0 - offset), (x1 + offset, y1 + offset)],
                outline=color, width=3)
        draw.text((x0 + 4, max(y0 - 18, 2)), 'global', fill=color, font=font)

    if len(crop_specs) >= 2:
        x0, y0, x1, y1 = _spec_to_pixel_box(crop_specs[1][batch_idx].tolist(), width, height)
        if local_enabled:
            color = CROP_COLORS['mask_local']
            label = 'mask_local' if not mask_fallback else 'mask_local (fallback→global)'
        else:
            color = CROP_COLORS['random_local']
            label = 'random_local'
        for offset in (0, 1):
            draw.rectangle(
                [(x0 - offset, y0 - offset), (x1 + offset, y1 + offset)],
                outline=color, width=3)
        draw.text((x0 + 4, max(y0 - 18, 2)), label, fill=color, font=font)

    mode = 'B: mask+paired' if local_enabled else 'A: random+unpaired'
    subtitle = (
        f'{mode} · green=global · '
        f'{"red=mask local" if local_enabled else "blue=random local"} · '
        f'mask_fallback={mask_fallback}'
    )
    return _compose_with_header(canvas, title=title, subtitle=subtitle, colorbar_range=None)


def crop_region_from_spec(
        image_pil: Image.Image,
        spec: Sequence[float]) -> Image.Image:
    width, height = image_pil.size
    x0, y0, x1, y1 = _spec_to_pixel_box(spec, width, height)
    return image_pil.crop((x0, y0, x1, y1))


def make_crop_panel(
        src_pil: Image.Image,
        alpha_latent: torch.Tensor,
        crop_specs: List[torch.Tensor],
        meta: Dict[str, torch.Tensor],
        *,
        batch_idx: int = 0,
        edit_is_low_alpha: bool = True,
        mass_threshold_percentile: float = 30.0,
) -> Image.Image:
    """2×2 panel: src | edit-mass heatmap / crop overlay | mask crop zoom."""
    local_enabled = bool(meta['local_enabled'][batch_idx].item())
    mask_fallback = bool(meta['mask_fallback'][batch_idx].item())
    alpha_heat = render_edit_mass_heatmap(
        src_pil,
        alpha_latent,
        edit_is_low_alpha=edit_is_low_alpha,
        mass_threshold_percentile=mass_threshold_percentile,
        step_label='step2',
    )
    alpha_heat_body = alpha_heat.crop((0, 52, alpha_heat.width, alpha_heat.height))
    overlay = render_crop_boxes_overlay(
        src_pil,
        crop_specs,
        batch_idx=batch_idx,
        mask_fallback=mask_fallback,
        local_enabled=local_enabled,
    )
    overlay_body = overlay.crop((0, 52, overlay.width, overlay.height))
    if local_enabled and len(crop_specs) >= 2:
        mask_spec = crop_specs[1][batch_idx].tolist()
        mask_zoom = crop_region_from_spec(src_pil, mask_spec).resize(
            (src_pil.width, src_pil.height), Image.Resampling.BICUBIC)
    elif not local_enabled and len(crop_specs) >= 2:
        random_spec = crop_specs[1][batch_idx].tolist()
        mask_zoom = crop_region_from_spec(src_pil, random_spec).resize(
            (src_pil.width, src_pil.height), Image.Resampling.BICUBIC)
    else:
        mask_zoom = Image.new('RGB', src_pil.size, (24, 28, 36))
        draw = ImageDraw.Draw(mask_zoom)
        draw.text((16, 16), 'local disabled', fill=(200, 200, 200))
    row1 = _hstack([src_pil, alpha_heat_body])
    row2 = _hstack([overlay_body, mask_zoom])
    panel_h = max(row1.height, row2.height)
    row1 = _pad_to_height(row1, panel_h)
    row2 = _pad_to_height(row2, panel_h)
    return _vstack([row1, row2])


def _hstack(images: List[Image.Image]) -> Image.Image:
    height = max(im.height for im in images)
    width = sum(im.width for im in images)
    canvas = Image.new('RGB', (width, height), (18, 22, 28))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def _vstack(images: List[Image.Image]) -> Image.Image:
    width = max(im.width for im in images)
    height = sum(im.height for im in images)
    canvas = Image.new('RGB', (width, height), (18, 22, 28))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
    return canvas


def _pad_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    canvas = Image.new('RGB', (image.width, height), (18, 22, 28))
    canvas.paste(image, (0, 0))
    return canvas


def build_crop_specs_for_alpha(
        alpha_latent: torch.Tensor,
        image_height: int,
        image_width: int,
        *,
        p_disable_local: float = 0.0,
        edit_is_low_alpha: bool = True,
        alpha_smooth_sigma: float = 2.0,
        mass_threshold_percentile: float = 30.0,
        mass_coverage_min: float = 0.85,
        mass_coverage_max: float = 0.90,
        union_area_max_ratio: float = 0.55,
        bbox_expand_factor: float = 1.4,
        min_crop_area_ratio: float = 0.05,
        max_crop_area_ratio: float = 0.55,
        min_edit_mass_ratio: float = 0.002,
        min_component_pixels: int = 16,
        seed: int = 0) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    device = alpha_latent.device
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    return sample_dino_alpha_mask_crop_specs(
        1,
        device,
        alpha=alpha_latent.unsqueeze(0) if alpha_latent.dim() == 3 else alpha_latent,
        image_height=image_height,
        image_width=image_width,
        p_disable_local=p_disable_local,
        edit_is_low_alpha=edit_is_low_alpha,
        alpha_smooth_sigma=alpha_smooth_sigma,
        mass_threshold_percentile=mass_threshold_percentile,
        mass_coverage_min=mass_coverage_min,
        mass_coverage_max=mass_coverage_max,
        union_area_max_ratio=union_area_max_ratio,
        bbox_expand_factor=bbox_expand_factor,
        min_crop_area_ratio=min_crop_area_ratio,
        max_crop_area_ratio=max_crop_area_ratio,
        min_edit_mass_ratio=min_edit_mass_ratio,
        min_component_pixels=min_component_pixels,
        rng=rng,
    )
