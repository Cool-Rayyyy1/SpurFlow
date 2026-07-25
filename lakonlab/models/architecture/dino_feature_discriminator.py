# Copyright (c) 2026 EditFlow contributors
#
# TDM-style DINO feature discriminator for EditFlow step-2 GAN:
#   frozen DINOv3 intermediate features + trainable conv head(s)
#   shared global/local random crops for real & fake RGB images in [0, 1]
#   optional alpha-guided mask local crop (low alpha = edit region)
#   optional source-conditional fusion: cat([DINO(ref), DINO(target)], dim=1)
#     -> D(ref, x_edit)=1, D(ref, x_student)=0
#   logistic (softplus) D/G losses

from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmgen.models.builder import MODULES
from torch.utils.checkpoint import checkpoint

from .dinov3_discriminator import load_dinov3_vitl16_from_hf

try:
    from scipy import ndimage as scipy_ndimage
except ImportError:  # pragma: no cover - optional dependency
    scipy_ndimage = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sample_random_crop_spec(
        scale_range: Sequence[float],
        aspect_range: Sequence[float],
        device: torch.device) -> Tuple[float, float, float, float]:
    min_scale, max_scale = [float(v) for v in scale_range]
    min_aspect, max_aspect = [float(v) for v in aspect_range]
    for _ in range(10):
        area = torch.empty((), device=device).uniform_(min_scale, max_scale).item()
        aspect = math.exp(torch.empty((), device=device).uniform_(
            math.log(min_aspect), math.log(max_aspect)).item())
        crop_h = math.sqrt(area / aspect)
        crop_w = math.sqrt(area * aspect)
        if crop_h <= 1.0 and crop_w <= 1.0:
            top = torch.empty((), device=device).uniform_(0.0, 1.0 - crop_h).item()
            left = torch.empty((), device=device).uniform_(0.0, 1.0 - crop_w).item()
            return top, left, crop_h, crop_w
    side = min(1.0, math.sqrt(max_scale))
    return (1.0 - side) / 2.0, (1.0 - side) / 2.0, side, side


def sample_dino_crop_specs(
        batch_size: int,
        device: torch.device,
        *,
        num_global_crops: int = 1,
        num_local_crops: int = 1,
        global_crop_scale: Sequence[float] = (0.5, 1.0),
        local_crop_scale: Sequence[float] = (0.125, 0.5),
        crop_aspect_ratio: Sequence[float] = (0.75, 1.3333333333),
) -> List[torch.Tensor]:
    """Shared crop specs for real and fake.

    When ``num_global_crops >= 1``, index 0 is the full image and additional
    crops use ``global_crop_scale`` / ``local_crop_scale`` as before.
    When ``num_global_crops <= 0``, only local random crops are returned
    (refinement-only GAN; no full-frame semantic view).
    """
    num_global_crops = int(num_global_crops)
    num_local_crops = int(num_local_crops)
    crop_specs: List[torch.Tensor] = []
    if num_global_crops > 0:
        crop_specs.append(
            torch.tensor(
                [[0.0, 0.0, 1.0, 1.0]] * int(batch_size),
                device=device,
                dtype=torch.float32,
            )
        )
        start = 1
        total = num_global_crops + num_local_crops
    else:
        if num_local_crops <= 0:
            raise ValueError(
                'sample_dino_crop_specs requires num_local_crops >= 1 '
                'when num_global_crops <= 0')
        start = 0
        total = num_local_crops
    for crop_idx in range(start, total):
        if num_global_crops > 0 and crop_idx < num_global_crops:
            scale_range = global_crop_scale
        else:
            scale_range = local_crop_scale
        crop_specs.append(
            torch.tensor(
                [
                    sample_random_crop_spec(scale_range, crop_aspect_ratio, device)
                    for _ in range(int(batch_size))
                ],
                device=device,
                dtype=torch.float32,
            )
        )
    return crop_specs


def _normalize_alpha_map(alpha: torch.Tensor) -> torch.Tensor:
    """Return per-sample alpha map (B, H, W) in latent or image space."""
    x = alpha.detach().float()
    while x.dim() > 3:
        x = x.squeeze(1)
    if x.dim() == 4:
        x = x.mean(dim=1)
    if x.dim() != 3:
        raise ValueError(f'Expected alpha (B, H, W) after normalization, got {tuple(alpha.shape)}')
    return x


def upsample_alpha_to_image(
        alpha: torch.Tensor,
        image_height: int,
        image_width: int,
        *,
        smooth: bool = True,
        smooth_sigma: float = 2.0) -> torch.Tensor:
    """Upsample alpha to image resolution; optional separable Gaussian smooth."""
    alpha_map = _normalize_alpha_map(alpha)
    upsampled = F.interpolate(
        alpha_map.unsqueeze(1),
        size=(int(image_height), int(image_width)),
        mode='bicubic',
        align_corners=False,
        antialias=True,
    ).squeeze(1)
    if smooth and smooth_sigma > 0:
        kernel_size = max(3, int(round(smooth_sigma * 4)) | 1)
        coords = torch.arange(kernel_size, device=upsampled.device, dtype=torch.float32)
        coords = coords - (kernel_size - 1) * 0.5
        gauss_1d = torch.exp(-0.5 * (coords / smooth_sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_h = gauss_1d.view(1, 1, 1, -1)
        gauss_w = gauss_1d.view(1, 1, -1, 1)
        upsampled = F.conv2d(
            upsampled.unsqueeze(1),
            gauss_h.expand(1, 1, 1, -1),
            padding=(0, kernel_size // 2),
        )
        upsampled = F.conv2d(
            upsampled,
            gauss_w.expand(1, 1, -1, 1),
            padding=(kernel_size // 2, 0),
        ).squeeze(1)
    return upsampled


def _label_connected_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    if scipy_ndimage is not None:
        return scipy_ndimage.label(binary_mask)
    labeled = np.zeros_like(binary_mask, dtype=np.int32)
    current_label = 0
    height, width = binary_mask.shape
    for y in range(height):
        for x in range(width):
            if not binary_mask[y, x] or labeled[y, x]:
                continue
            current_label += 1
            stack = [(y, x)]
            labeled[y, x] = current_label
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary_mask[ny, nx] and labeled[ny, nx] == 0:
                        labeled[ny, nx] = current_label
                        stack.append((ny, nx))
    return labeled, current_label


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _expand_square_bbox(
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        height: int,
        width: int,
        *,
        expand_factor: float = 1.4,
        min_area_ratio: float = 0.05,
        max_area_ratio: float = 0.55) -> Optional[Tuple[float, float, float, float]]:
    box_h = max(y1 - y0, 1)
    box_w = max(x1 - x0, 1)
    cy = (y0 + y1) * 0.5
    cx = (x0 + x1) * 0.5
    side = max(box_h, box_w) * float(expand_factor)
    image_area = float(height * width)
    min_side = math.sqrt(max(min_area_ratio, 1e-6) * image_area)
    max_side = math.sqrt(max(max_area_ratio, min_area_ratio) * image_area)
    side = min(max(side, min_side), max_side, float(height), float(width))
    half = side * 0.5
    top = int(round(cy - half))
    left = int(round(cx - half))
    top = min(max(top, 0), max(height - int(round(side)), 0))
    left = min(max(left, 0), max(width - int(round(side)), 0))
    crop_h = min(int(round(side)), height - top)
    crop_w = min(int(round(side)), width - left)
    if crop_h <= 1 or crop_w <= 1:
        return None
    return (
        top / float(height),
        left / float(width),
        crop_h / float(height),
        crop_w / float(width),
    )


def compute_mask_guided_crop_spec(
        alpha_map: torch.Tensor,
        *,
        edit_is_low_alpha: bool = True,
        smooth_sigma: float = 2.0,
        mass_threshold_percentile: float = 30.0,
        mass_coverage_min: float = 0.85,
        mass_coverage_max: float = 0.90,
        union_area_max_ratio: float = 0.35,
        bbox_expand_factor: float = 1.1,
        min_crop_area_ratio: float = 0.01,
        max_crop_area_ratio: float = 0.35,
        min_edit_mass_ratio: float = 0.002,
        min_component_pixels: int = 16,
        hot_mass_frac: float = 0.40) -> Optional[Tuple[float, float, float, float]]:
    """Build a tight crop from the edit-mass heatmap hot core (red region).

    Matches the edit-mass vis colormap: red kicks in around ``t ≳ 0.35`` where
    ``t = edit_mass / max``. We threshold at ``hot_mass_frac * max`` (default 0.4),
    keep the largest connected hot blob, then pad with a small square expand.

    ``mass_coverage_*`` / ``union_area_max_ratio`` are kept for API compatibility
    but are no longer used to accumulate many weak components.
    """
    del mass_coverage_min, mass_coverage_max, union_area_max_ratio  # unused (legacy API)
    alpha_np = alpha_map.detach().float().cpu().numpy()
    height, width = alpha_np.shape
    if height <= 1 or width <= 1:
        return None

    if smooth_sigma > 0:
        if scipy_ndimage is not None:
            alpha_np = scipy_ndimage.gaussian_filter(alpha_np, sigma=smooth_sigma)
        else:
            alpha_t = torch.from_numpy(alpha_np).unsqueeze(0).unsqueeze(0)
            kernel_size = max(3, int(round(smooth_sigma * 4)) | 1)
            alpha_t = F.avg_pool2d(
                F.pad(alpha_t, [kernel_size // 2] * 4, mode='reflect'),
                kernel_size=kernel_size,
                stride=1,
            )
            alpha_np = alpha_t.squeeze().numpy()

    if edit_is_low_alpha:
        ref = float(np.percentile(alpha_np, mass_threshold_percentile))
        edit_mass = np.clip(ref - alpha_np, 0.0, None)
    else:
        ref = float(np.percentile(alpha_np, 100.0 - mass_threshold_percentile))
        edit_mass = np.clip(alpha_np - ref, 0.0, None)

    total_mass = float(edit_mass.sum())
    if total_mass <= min_edit_mass_ratio * height * width:
        return None

    vmax = float(edit_mass.max())
    if vmax <= 1e-12:
        return None

    # Hot core ≈ red region on edit-mass heatmap.
    hot_frac = float(np.clip(hot_mass_frac, 0.05, 0.95))
    hot_thr = max(vmax * hot_frac, 1e-8)
    binary = edit_mass >= hot_thr
    if int(binary.sum()) < int(min_component_pixels):
        # Mild relax toward orange/yellow mid-tones, still above weak blue halo.
        hot_thr = max(vmax * max(hot_frac * 0.7, 0.25), 1e-8)
        binary = edit_mass >= hot_thr
    if int(binary.sum()) < int(min_component_pixels):
        return None

    labeled, num_labels = _label_connected_components(binary)
    if num_labels <= 0:
        return None

    best_mask = None
    best_mass = -1.0
    for label_id in range(1, num_labels + 1):
        comp_mask = labeled == label_id
        pixel_count = int(comp_mask.sum())
        if pixel_count < int(min_component_pixels):
            continue
        mass = float(edit_mass[comp_mask].sum())
        if mass > best_mass:
            best_mass = mass
            best_mask = comp_mask
    if best_mask is None:
        return None

    bbox = _bbox_from_mask(best_mask)
    if bbox is None:
        return None
    uy0, uy1, ux0, ux1 = bbox
    return _expand_square_bbox(
        uy0, uy1, ux0, ux1, height, width,
        expand_factor=bbox_expand_factor,
        min_area_ratio=min_crop_area_ratio,
        max_area_ratio=max_crop_area_ratio,
    )


def sample_dino_alpha_mask_crop_specs(
        batch_size: int,
        device: torch.device,
        *,
        alpha: Optional[torch.Tensor] = None,
        image_height: int,
        image_width: int,
        p_disable_local: float = 0.0,
        num_global_crops: int = 1,
        global_crop_scale: Sequence[float] = (0.5, 1.0),
        local_crop_scale: Sequence[float] = (0.125, 0.5),
        crop_aspect_ratio: Sequence[float] = (0.75, 1.3333333333),
        edit_is_low_alpha: bool = True,
        alpha_smooth_sigma: float = 2.0,
        mass_threshold_percentile: float = 30.0,
        mass_coverage_min: float = 0.85,
        mass_coverage_max: float = 0.90,
        union_area_max_ratio: float = 0.35,
        bbox_expand_factor: float = 1.1,
        min_crop_area_ratio: float = 0.01,
        max_crop_area_ratio: float = 0.35,
        min_edit_mass_ratio: float = 0.002,
        min_component_pixels: int = 16,
        hot_mass_frac: float = 0.40,
        rng: Optional[torch.Generator] = None) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    """Shared crop specs with two modes (always global + one local slot).

    Case A (``local_enabled=False``, prob ``p_disable_local``):
        crops = [full global, random local]; use dataset-sampled unpaired
        edited images as GAN reals (see ``ImageEdit.load_unpaired_edited``).
    Case B (``local_enabled=True``):
        crops = [full global, alpha-mask local]; use paired ref/edit real images.
        Mask failure falls back to random local (not a duplicate of global).
    """
    batch_size = int(batch_size)
    global_spec = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0]] * batch_size,
        device=device,
        dtype=torch.float32,
    )
    # local_enabled=True -> mask-local mode (Case B); False -> random-local (Case A).
    local_enabled = torch.rand(batch_size, device=device, generator=rng) >= float(p_disable_local)
    mask_fallback = torch.zeros(batch_size, device=device, dtype=torch.bool)

    random_local = torch.tensor(
        [
            sample_random_crop_spec(local_crop_scale, crop_aspect_ratio, device)
            for _ in range(batch_size)
        ],
        device=device,
        dtype=torch.float32,
    )
    local_crop = random_local.clone()
    if alpha is not None:
        alpha_img = upsample_alpha_to_image(
            alpha, image_height, image_width, smooth=False, smooth_sigma=alpha_smooth_sigma)
        for batch_idx in range(batch_size):
            if not bool(local_enabled[batch_idx]):
                continue
            spec = compute_mask_guided_crop_spec(
                alpha_img[batch_idx],
                edit_is_low_alpha=edit_is_low_alpha,
                smooth_sigma=alpha_smooth_sigma,
                mass_threshold_percentile=mass_threshold_percentile,
                mass_coverage_min=mass_coverage_min,
                mass_coverage_max=mass_coverage_max,
                union_area_max_ratio=union_area_max_ratio,
                bbox_expand_factor=bbox_expand_factor,
                min_crop_area_ratio=min_crop_area_ratio,
                max_crop_area_ratio=max_crop_area_ratio,
                min_edit_mass_ratio=min_edit_mass_ratio,
                min_component_pixels=min_component_pixels,
                hot_mass_frac=hot_mass_frac,
            )
            if spec is None:
                # Keep a local crop via random fallback; do not duplicate global.
                mask_fallback[batch_idx] = True
                local_crop[batch_idx] = random_local[batch_idx]
            else:
                local_crop[batch_idx] = torch.tensor(spec, device=device, dtype=torch.float32)
    else:
        mask_fallback.fill_(True)
        local_crop = random_local.clone()

    # Per-sample: Case A keeps random local; Case B uses mask (or random fallback).
    local_crop = torch.where(
        local_enabled.unsqueeze(1),
        local_crop,
        random_local,
    )
    crop_specs = [global_spec, local_crop]
    meta = dict(
        local_enabled=local_enabled,
        mask_fallback=mask_fallback,
        num_active_crops=torch.full(
            (batch_size,), 2, device=device, dtype=torch.int64),
    )
    return crop_specs, meta


def apply_dino_crop_specs(
        image_pixels: torch.Tensor,
        crop_specs: Sequence[torch.Tensor],
        *,
        num_global_crops: int = 1,
        global_input_size: int = 224,
        local_input_size: int = 224,
        clamp_pixels: bool = True,
) -> List[torch.Tensor]:
    batch_size, _, height, width = image_pixels.shape
    crops = []
    for crop_idx, specs in enumerate(crop_specs):
        output_size = global_input_size if crop_idx < int(num_global_crops) else local_input_size
        per_sample = []
        for batch_idx in range(batch_size):
            top, left, crop_h, crop_w = specs[batch_idx].tolist()
            y0 = min(max(int(round(top * height)), 0), height - 1)
            x0 = min(max(int(round(left * width)), 0), width - 1)
            y1 = min(max(int(round((top + crop_h) * height)), y0 + 1), height)
            x1 = min(max(int(round((left + crop_w) * width)), x0 + 1), width)
            per_sample.append(image_pixels[batch_idx:batch_idx + 1, :, y0:y1, x0:x1])

        resized = []
        for crop in per_sample:
            crop = F.interpolate(
                crop.float(),
                size=(output_size, output_size),
                mode='bicubic',
                align_corners=False,
                antialias=True,
            )
            if clamp_pixels:
                crop = crop.clamp(0.0, 1.0)
            resized.append(crop)
        crops.append(torch.cat(resized, dim=0))
    return crops


class DinoConvBlock(nn.Module):

    def __init__(
            self,
            channels: int,
            *,
            stride: int = 1,
            post_avgpool: bool = False,
            norm_groups: int = 32):
        super().__init__()
        groups = min(int(norm_groups), int(channels))
        while int(channels) % groups != 0:
            groups -= 1
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.act = nn.SiLU()
        self.post_avgpool = bool(post_avgpool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.conv(x)))
        if self.post_avgpool and min(x.shape[-2:]) > 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class DinoConvClsHead(nn.Module):

    def __init__(
            self,
            channels: int,
            *,
            num_blocks: int = 3,
            use_avgpool: bool = False,
            dense_output: bool = False,
            gradient_checkpointing: bool = True,
            norm_groups: int = 32):
        super().__init__()
        self.dense_output = bool(dense_output)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        stride = 1 if use_avgpool else 2
        self.blocks = nn.ModuleList([
            DinoConvBlock(
                channels,
                stride=stride,
                post_avgpool=use_avgpool,
                norm_groups=norm_groups,
            )
            for _ in range(int(num_blocks))
        ])

    def forward(self, crops: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        checkpoint_blocks = (
            self.gradient_checkpointing
            and torch.is_grad_enabled()
            and any(p.requires_grad for p in self.parameters()))
        for crop in crops:
            x = crop
            for block in self.blocks:
                if checkpoint_blocks:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            if not self.dense_output:
                x = x.mean(dim=1, keepdim=True)
            outputs.append(x)
        return outputs


class DinoDiscriminatorHead(nn.Module):

    def __init__(
            self,
            *,
            channels: int,
            num_discriminators: int,
            num_steps: int,
            num_blocks: int = 3,
            dense_output: bool = False,
            use_avgpool: bool = False,
            step_conditioning: bool = False,
            gradient_checkpointing: bool = True,
            norm_groups: int = 32):
        super().__init__()
        self.step_conditioning = bool(step_conditioning)
        if self.step_conditioning:
            self.step_embedding = nn.Embedding(int(num_steps), int(channels))
        self.heads = nn.ModuleList([
            DinoConvClsHead(
                channels=channels,
                num_blocks=num_blocks,
                use_avgpool=use_avgpool,
                dense_output=dense_output,
                gradient_checkpointing=gradient_checkpointing,
                norm_groups=norm_groups,
            )
            for _ in range(int(num_discriminators))
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(int(num_discriminators))
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(channels, 1) for _ in range(int(num_discriminators))
        ])

    def forward(
            self,
            intermediate_features: Sequence[Sequence[torch.Tensor]],
            step_indices: torch.Tensor) -> torch.Tensor:
        logits = []
        step_indices = step_indices.long()
        for head, norm, classifier, crops in zip(
                self.heads, self.norms, self.classifiers, intermediate_features):
            if self.step_conditioning:
                step_embedding = self.step_embedding(step_indices)[:, :, None, None]
                crops = [crop + step_embedding.to(crop.dtype) for crop in crops]
            features = torch.cat(head(crops), dim=0)
            logits.append(classifier(norm(features)).squeeze(dim=2))
        return torch.cat(logits, dim=1)


@MODULES.register_module()
class DinoFeatureDiscriminator(nn.Module):
    """Frozen DINOv3 ViT-L/16 features + TDM-style conv discriminator head.

    Real/fake targets are RGB in ``[0, 1]``. With ``condition_on_source=True``
    the discriminator scores pairs:

      D(x_src, x_edit) -> 1
      D(x_src, x_student) -> 0

    Source and target share crop windows; backbone features are channel-concatenated
    before the head so D judges edit quality relative to the source, instead of
    unconditional photorealism (which can pull edits back toward the source).
    """

    def __init__(
            self,
            checkpoint_path: str,
            num_steps: int = 2,
            feature_layers: Sequence[int] = (23,),
            global_input_size: int = 224,
            local_input_size: int = 224,
            num_global_crops: int = 1,
            num_local_crops: int = 1,
            global_crop_scale: Sequence[float] = (0.5, 1.0),
            local_crop_scale: Sequence[float] = (0.125, 0.5),
            crop_aspect_ratio: Sequence[float] = (0.75, 1.3333333333),
            clamp_pixels: bool = True,
            step_conditioning: bool = False,
            head_num_blocks: int = 3,
            head_use_avgpool: bool = False,
            head_gradient_checkpointing: bool = True,
            head_norm_groups: int = 32,
            dense_output: bool = False,
            condition_on_source: bool = False,
            gan_global_weight: float = 1.0,
            gan_local_weight: float = 1.0,
            backbone_dtype: str = 'bf16',
            head_dtype: str = 'fp32',
            freeze_backbone: bool = True):
        super().__init__()
        self.num_steps = int(num_steps)
        self.feature_layers = tuple(int(v) for v in feature_layers)
        self.num_global_crops = int(num_global_crops)
        self.num_local_crops = int(num_local_crops)
        self.global_input_size = int(global_input_size)
        self.local_input_size = int(local_input_size)
        self.global_crop_scale = tuple(float(v) for v in global_crop_scale)
        self.local_crop_scale = tuple(float(v) for v in local_crop_scale)
        self.crop_aspect_ratio = tuple(float(v) for v in crop_aspect_ratio)
        self.clamp_pixels = bool(clamp_pixels)
        self.step_conditioning = bool(step_conditioning)
        self.condition_on_source = bool(condition_on_source)
        self.gan_global_weight = float(gan_global_weight)
        self.gan_local_weight = float(gan_local_weight)
        if self.num_global_crops <= 0 and self.num_local_crops <= 0:
            raise ValueError(
                'DinoFeatureDiscriminator needs num_global_crops > 0 or '
                'num_local_crops > 0')
        if self.gan_global_weight <= 0 and self.gan_local_weight <= 0:
            raise ValueError(
                'DinoFeatureDiscriminator needs gan_global_weight > 0 or '
                'gan_local_weight > 0')

        dtype_map = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
        self.backbone_dtype = dtype_map[str(backbone_dtype).lower()]
        self.head_dtype = dtype_map[str(head_dtype).lower()]

        self.backbone = load_dinov3_vitl16_from_hf(checkpoint_path)
        if self.backbone_dtype != torch.float32:
            self.backbone = self.backbone.to(dtype=self.backbone_dtype)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        channels = int(self.backbone.num_features)
        if self.condition_on_source:
            channels = channels * 2
        self.head = DinoDiscriminatorHead(
            channels=channels,
            num_discriminators=len(self.feature_layers),
            num_steps=self.num_steps,
            num_blocks=int(head_num_blocks),
            dense_output=bool(dense_output),
            use_avgpool=bool(head_use_avgpool),
            step_conditioning=bool(step_conditioning),
            gradient_checkpointing=bool(head_gradient_checkpointing),
            norm_groups=int(head_norm_groups),
        ).to(dtype=self.head_dtype)

        self.register_buffer(
            'mean',
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'std',
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def make_step_indices(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Step-2 supervision for 2-NFE: final rollout index = num_steps - 1."""
        device = device or self.device
        return torch.full(
            (int(batch_size),), self.num_steps - 1, device=device, dtype=torch.long)

    def sample_crop_specs(self, batch_size: int, device: Optional[torch.device] = None) -> List[torch.Tensor]:
        return sample_dino_crop_specs(
            batch_size,
            device or self.device,
            num_global_crops=self.num_global_crops,
            num_local_crops=self.num_local_crops,
            global_crop_scale=self.global_crop_scale,
            local_crop_scale=self.local_crop_scale,
            crop_aspect_ratio=self.crop_aspect_ratio,
        )

    def apply_crop_specs(
            self,
            image_pixels: torch.Tensor,
            crop_specs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return apply_dino_crop_specs(
            image_pixels,
            crop_specs,
            num_global_crops=self.num_global_crops,
            global_input_size=self.global_input_size,
            local_input_size=self.local_input_size,
            clamp_pixels=self.clamp_pixels,
        )

    def _extract_single_features(
            self,
            image_pixels: torch.Tensor,
            crop_specs: Sequence[torch.Tensor],
            *,
            requires_input_grad: bool) -> List[List[torch.Tensor]]:
        if self.clamp_pixels:
            image_pixels = image_pixels.clamp(0.0, 1.0)
        crops = self.apply_crop_specs(image_pixels.to(self.device), crop_specs)
        features_by_layer: List[List[torch.Tensor]] = [[] for _ in self.feature_layers]
        self.backbone.eval()

        grad_context = contextlib.nullcontext() if requires_input_grad else torch.no_grad()
        with grad_context:
            for crop in crops:
                normalized = ((crop.float() - self.mean) / self.std).to(self.backbone_dtype)
                with torch.autocast(device_type=normalized.device.type, enabled=False):
                    layer_feats = self.backbone.forward_intermediates(
                        normalized,
                        indices=list(self.feature_layers),
                        norm=False,
                        output_fmt='NCHW',
                        intermediates_only=True,
                    )
                if torch.is_tensor(layer_feats):
                    layer_feats = (layer_feats,)
                for layer_idx, feature in enumerate(layer_feats):
                    features_by_layer[layer_idx].append(feature.to(self.head_dtype))
        return features_by_layer

    def _fuse_cond_target_features(
            self,
            cond_features: List[List[torch.Tensor]],
            target_features: List[List[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        fused: List[List[torch.Tensor]] = []
        for cond_crops, target_crops in zip(cond_features, target_features):
            fused.append([
                torch.cat([cond_feat, target_feat], dim=1)
                for cond_feat, target_feat in zip(cond_crops, target_crops)
            ])
        return fused

    def _extract_features(
            self,
            image_pixels: torch.Tensor,
            crop_specs: Sequence[torch.Tensor],
            *,
            requires_input_grad: bool,
            cond_images: Optional[torch.Tensor] = None) -> List[List[torch.Tensor]]:
        target_features = self._extract_single_features(
            image_pixels, crop_specs, requires_input_grad=requires_input_grad)
        if cond_images is None:
            return target_features
        if cond_images.shape[-2:] != image_pixels.shape[-2:]:
            cond_images = F.interpolate(
                cond_images.float(),
                size=image_pixels.shape[-2:],
                mode='bilinear',
                align_corners=False)
            if image_pixels.dtype != torch.float32:
                cond_images = cond_images.to(dtype=image_pixels.dtype)
        # Source is a fixed conditioner; never backprop into it.
        cond_features = self._extract_single_features(
            cond_images, crop_specs, requires_input_grad=False)
        return self._fuse_cond_target_features(cond_features, target_features)

    def logits_from_pixels(
            self,
            image_pixels: torch.Tensor,
            step_indices: torch.Tensor,
            *,
            crop_specs: Optional[Sequence[torch.Tensor]] = None,
            requires_input_grad: bool = False,
            cond_images: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.condition_on_source and cond_images is None:
            raise ValueError(
                'DinoFeatureDiscriminator(condition_on_source=True) requires '
                'cond_images/source_images for D(x_src, x_target).')
        if (not self.condition_on_source) and cond_images is not None:
            cond_images = None
        if crop_specs is None:
            crop_specs = self.sample_crop_specs(image_pixels.shape[0], image_pixels.device)
        features = self._extract_features(
            image_pixels,
            crop_specs,
            requires_input_grad=requires_input_grad,
            cond_images=cond_images)
        return self.head(features, step_indices.to(self.device))

    @staticmethod
    def _discriminator_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
        return F.softplus(logits_fake.float()).mean() + F.softplus(-logits_real.float()).mean()

    @staticmethod
    def _generator_loss(logits_fake: torch.Tensor) -> torch.Tensor:
        return F.softplus(-logits_fake.float()).mean()

    def num_crop_views(self) -> int:
        if self.num_global_crops <= 0:
            return int(self.num_local_crops)
        return int(self.num_global_crops) + int(self.num_local_crops)

    def _crop_branch_weight(self, crop_idx: int) -> float:
        if self.num_global_crops <= 0:
            return self.gan_local_weight
        if crop_idx < self.num_global_crops:
            return self.gan_global_weight
        return self.gan_local_weight

    def _weighted_crop_gan_loss(
            self,
            logits_real: Optional[torch.Tensor],
            logits_fake: torch.Tensor,
            *,
            gan_mode: str) -> torch.Tensor:
        """Average D/G loss across crops with global/local weights.

        Logits are laid out as ``(num_crops * B, ...)`` from
        ``DinoDiscriminatorHead`` (crops concatenated on the batch dim).
        """
        num_crops = self.num_crop_views()
        if num_crops <= 0:
            raise RuntimeError('DinoFeatureDiscriminator has no crop views')
        if logits_fake.shape[0] % num_crops != 0:
            # Fallback for unexpected layouts (keeps old equal-weight behaviour).
            if gan_mode == 'discriminator':
                assert logits_real is not None
                return self._discriminator_loss(logits_real, logits_fake)
            return self._generator_loss(logits_fake)

        batch_size = logits_fake.shape[0] // num_crops
        fake_by_crop = logits_fake.view(num_crops, batch_size, *logits_fake.shape[1:])
        real_by_crop = None
        if gan_mode == 'discriminator':
            assert logits_real is not None
            real_by_crop = logits_real.view(num_crops, batch_size, *logits_real.shape[1:])

        total = logits_fake.new_zeros(())
        total_weight = 0.0
        for crop_idx in range(num_crops):
            weight = float(self._crop_branch_weight(crop_idx))
            if weight <= 0:
                continue
            lf = fake_by_crop[crop_idx]
            if gan_mode == 'discriminator':
                lr = real_by_crop[crop_idx]
                crop_loss = self._discriminator_loss(lr, lf)
            else:
                crop_loss = self._generator_loss(lf)
            total = total + weight * crop_loss
            total_weight += weight
        if total_weight <= 0:
            if gan_mode == 'discriminator':
                assert logits_real is not None
                return self._discriminator_loss(logits_real, logits_fake)
            return self._generator_loss(logits_fake)
        return total / total_weight

    @staticmethod
    def build_log_vars(real_logits: torch.Tensor, fake_logits: torch.Tensor, loss_d: torch.Tensor) -> Dict[str, float]:
        return dict(
            loss_d=float(loss_d.detach()),
            d_real=float(real_logits.detach().mean()),
            d_fake=float(fake_logits.detach().mean()),
            dino_gan_realism_real=float(torch.sigmoid(real_logits.float()).detach().mean()),
            dino_gan_realism_fake=float(torch.sigmoid(fake_logits.float()).detach().mean()),
            dino_gan_discriminator_accuracy=float(
                (real_logits.float() > fake_logits.float()).float().mean()),
        )

    @staticmethod
    def build_generator_log_vars(fake_logits: torch.Tensor, loss_g: torch.Tensor) -> Dict[str, float]:
        return dict(
            loss_g_gan=float(loss_g.detach()),
            g_fake_logits=float(fake_logits.detach().mean()),
            dino_gan_generator_realism=float(torch.sigmoid(fake_logits.float()).detach().mean()),
        )

    def forward(
            self,
            images: Optional[torch.Tensor] = None,
            real_images: Optional[torch.Tensor] = None,
            fake_images: Optional[torch.Tensor] = None,
            gan_mode: Optional[str] = None,
            step_indices: Optional[torch.Tensor] = None,
            crop_specs: Optional[Sequence[torch.Tensor]] = None,
            cond_images: Optional[torch.Tensor] = None,
            source_images: Optional[torch.Tensor] = None):
        if cond_images is None:
            cond_images = source_images
        if gan_mode == 'discriminator':
            assert real_images is not None and fake_images is not None
            if step_indices is None:
                step_indices = self.make_step_indices(fake_images.shape[0], fake_images.device)
            if crop_specs is None:
                crop_specs = self.sample_crop_specs(fake_images.shape[0], fake_images.device)
            logits_real = self.logits_from_pixels(
                real_images.detach(),
                step_indices,
                crop_specs=crop_specs,
                requires_input_grad=False,
                cond_images=None if cond_images is None else cond_images.detach())
            logits_fake = self.logits_from_pixels(
                fake_images.detach(),
                step_indices,
                crop_specs=crop_specs,
                requires_input_grad=False,
                cond_images=None if cond_images is None else cond_images.detach())
            return self._weighted_crop_gan_loss(
                logits_real, logits_fake, gan_mode='discriminator')

        if gan_mode == 'generator':
            assert fake_images is not None
            if step_indices is None:
                step_indices = self.make_step_indices(fake_images.shape[0], fake_images.device)
            if crop_specs is None:
                crop_specs = self.sample_crop_specs(fake_images.shape[0], fake_images.device)
            logits_fake = self.logits_from_pixels(
                fake_images,
                step_indices,
                crop_specs=crop_specs,
                requires_input_grad=True,
                cond_images=None if cond_images is None else cond_images.detach())
            return self._weighted_crop_gan_loss(
                None, logits_fake, gan_mode='generator')

        if images is None:
            raise ValueError('DinoFeatureDiscriminator requires `images` when gan_mode is None.')
        if step_indices is None:
            step_indices = self.make_step_indices(images.shape[0], images.device)
        return self.logits_from_pixels(
            images,
            step_indices,
            crop_specs=crop_specs,
            requires_input_grad=False,
            cond_images=None if cond_images is None else cond_images.detach())


@MODULES.register_module()
class DinoAlphaMaskFeatureDiscriminator(DinoFeatureDiscriminator):
    """DINO feature GAN with optional alpha-guided mask local crop.

    With ``condition_on_source=True`` (default), ref and target share crop windows,
    DINO features are channel-concatenated, and the head scores pairs:

      D(x_src, x_edit) -> 1
      D(x_src, x_student) -> 0
    """

    CROP_GLOBAL = 0
    CROP_LOCAL = 1

    def __init__(
            self,
            *args,
            use_mask_local_crop: bool = True,
            p_disable_local: float = 0.0,
            edit_is_low_alpha: bool = True,
            alpha_smooth_sigma: float = 2.0,
            mass_threshold_percentile: float = 30.0,
            mass_coverage_min: float = 0.85,
            mass_coverage_max: float = 0.90,
            union_area_max_ratio: float = 0.35,
            bbox_expand_factor: float = 1.1,
            min_crop_area_ratio: float = 0.01,
            max_crop_area_ratio: float = 0.35,
            min_edit_mass_ratio: float = 0.002,
            min_component_pixels: int = 16,
            hot_mass_frac: float = 0.40,
            gan_global_weight: float = 1.0,
            gan_random_local_weight: float = 1.0,
            gan_mask_local_weight: float = 1.0,
            **kwargs):
        if use_mask_local_crop:
            kwargs.setdefault('num_global_crops', 1)
            kwargs.setdefault('num_local_crops', 1)
        # Source-conditional by default for alpha edit GAN.
        kwargs.setdefault('condition_on_source', True)
        super().__init__(*args, **kwargs)
        self.use_mask_local_crop = bool(use_mask_local_crop)
        self.p_disable_local = float(p_disable_local)
        self.edit_is_low_alpha = bool(edit_is_low_alpha)
        self.alpha_smooth_sigma = float(alpha_smooth_sigma)
        self.mass_threshold_percentile = float(mass_threshold_percentile)
        self.mass_coverage_min = float(mass_coverage_min)
        self.mass_coverage_max = float(mass_coverage_max)
        self.union_area_max_ratio = float(union_area_max_ratio)
        self.bbox_expand_factor = float(bbox_expand_factor)
        self.min_crop_area_ratio = float(min_crop_area_ratio)
        self.max_crop_area_ratio = float(max_crop_area_ratio)
        self.min_edit_mass_ratio = float(min_edit_mass_ratio)
        self.min_component_pixels = int(min_component_pixels)
        self.hot_mass_frac = float(hot_mass_frac)
        self.gan_global_weight = float(gan_global_weight)
        self.gan_random_local_weight = float(gan_random_local_weight)
        self.gan_mask_local_weight = float(gan_mask_local_weight)
        self._last_crop_meta: Optional[Dict[str, torch.Tensor]] = None

    def sample_crop_specs(
            self,
            batch_size: int,
            device: Optional[torch.device] = None,
            *,
            alpha: Optional[torch.Tensor] = None,
            image_height: Optional[int] = None,
            image_width: Optional[int] = None) -> List[torch.Tensor]:
        device = device or self.device
        if not self.use_mask_local_crop:
            self._last_crop_meta = None
            return super().sample_crop_specs(batch_size, device)
        if image_height is None or image_width is None:
            raise ValueError(
                'DinoAlphaMaskFeatureDiscriminator requires image_height/image_width '
                'when use_mask_local_crop=True.')
        crop_specs, meta = sample_dino_alpha_mask_crop_specs(
            batch_size,
            device,
            alpha=alpha,
            image_height=int(image_height),
            image_width=int(image_width),
            p_disable_local=self.p_disable_local,
            num_global_crops=self.num_global_crops,
            global_crop_scale=self.global_crop_scale,
            local_crop_scale=self.local_crop_scale,
            crop_aspect_ratio=self.crop_aspect_ratio,
            edit_is_low_alpha=self.edit_is_low_alpha,
            alpha_smooth_sigma=self.alpha_smooth_sigma,
            mass_threshold_percentile=self.mass_threshold_percentile,
            mass_coverage_min=self.mass_coverage_min,
            mass_coverage_max=self.mass_coverage_max,
            union_area_max_ratio=self.union_area_max_ratio,
            bbox_expand_factor=self.bbox_expand_factor,
            min_crop_area_ratio=self.min_crop_area_ratio,
            max_crop_area_ratio=self.max_crop_area_ratio,
            min_edit_mass_ratio=self.min_edit_mass_ratio,
            min_component_pixels=self.min_component_pixels,
            hot_mass_frac=self.hot_mass_frac,
        )
        self._last_crop_meta = meta
        return crop_specs

    def _split_logits_by_crop(
            self,
            logits: torch.Tensor,
            batch_size: int,
            num_crops: int) -> torch.Tensor:
        num_layers = logits.shape[1]
        return logits.view(num_crops, batch_size, num_layers)

    def _crop_active_mask(
            self,
            crop_idx: int,
            batch_size: int,
            device: torch.device,
            *,
            branch: Optional[str] = None) -> torch.Tensor:
        if crop_idx == self.CROP_GLOBAL or self._last_crop_meta is None:
            return torch.ones(batch_size, device=device, dtype=torch.bool)
        local_enabled = self._last_crop_meta.get('local_enabled')
        if local_enabled is None:
            return torch.ones(batch_size, device=device, dtype=torch.bool)
        local_enabled = local_enabled.to(device=device, dtype=torch.bool)
        mask_fallback = self._last_crop_meta.get('mask_fallback')
        if mask_fallback is not None:
            mask_fallback = mask_fallback.to(device=device, dtype=torch.bool)
        else:
            mask_fallback = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if branch == 'random_local':
            # Case A, or Case B after mask-guided crop failed.
            return (~local_enabled) | mask_fallback
        if branch == 'mask_local':
            return local_enabled & (~mask_fallback)
        return torch.ones(batch_size, device=device, dtype=torch.bool)

    def _weighted_gan_loss(
            self,
            logits_real: torch.Tensor,
            logits_fake: torch.Tensor,
            *,
            gan_mode: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_crops = 2
        batch_size = logits_fake.shape[0] // num_crops
        if batch_size * num_crops != logits_fake.shape[0]:
            num_crops = 1
            batch_size = logits_fake.shape[0]
        logits_fake = self._split_logits_by_crop(logits_fake, batch_size, num_crops)
        if gan_mode == 'discriminator':
            logits_real = self._split_logits_by_crop(logits_real, batch_size, num_crops)
        else:
            logits_real = None

        total_loss = logits_fake.new_zeros(())
        total_weight = 0.0
        # Always write every branch key so TextLoggerHook does not keep stale
        # values from a previous iter where that branch was active.
        mode_tag = 'd' if gan_mode == 'discriminator' else 'g'
        log_vars: Dict[str, float] = {
            'dino_gan_global_logits_fake': 0.0,
            'dino_gan_random_local_logits_fake': 0.0,
            'dino_gan_mask_local_logits_fake': 0.0,
            f'loss_{mode_tag}_global': 0.0,
            f'loss_{mode_tag}_random_local': 0.0,
            f'loss_{mode_tag}_mask_local': 0.0,
        }
        if gan_mode == 'discriminator':
            log_vars.update({
                'dino_gan_global_logits_real': 0.0,
                'dino_gan_random_local_logits_real': 0.0,
                'dino_gan_mask_local_logits_real': 0.0,
            })
        branches = (
            (self.CROP_GLOBAL, None, self.gan_global_weight, 'global'),
            (self.CROP_LOCAL, 'random_local', self.gan_random_local_weight, 'random_local'),
            (self.CROP_LOCAL, 'mask_local', self.gan_mask_local_weight, 'mask_local'),
        )
        for crop_idx, branch, weight, name in branches:
            if weight <= 0:
                continue
            active = self._crop_active_mask(
                crop_idx, batch_size, logits_fake.device, branch=branch)
            if not bool(active.any()):
                continue
            lf = logits_fake[crop_idx, active]
            if gan_mode == 'discriminator':
                lr = logits_real[crop_idx, active]
                crop_loss = self._discriminator_loss(lr, lf)
            else:
                crop_loss = self._generator_loss(lf)
            active_frac = float(active.float().mean())
            total_loss = total_loss + (weight * crop_loss)
            total_weight += weight * active_frac
            with torch.no_grad():
                if gan_mode == 'discriminator':
                    log_vars[f'dino_gan_{name}_logits_real'] = float(lr.mean())
                log_vars[f'dino_gan_{name}_logits_fake'] = float(lf.mean())
                log_vars[f'loss_{mode_tag}_{name}'] = float(crop_loss.detach())

        if total_weight <= 0:
            if gan_mode == 'discriminator':
                return self._discriminator_loss(logits_real[0], logits_fake[0]), log_vars
            return self._generator_loss(logits_fake[0]), log_vars
        return total_loss / total_weight, log_vars

    def _crop_weights(self) -> Tuple[float, ...]:
        if not self.use_mask_local_crop:
            return (self.gan_global_weight, self.gan_random_local_weight)
        return (
            self.gan_global_weight,
            self.gan_random_local_weight,
            self.gan_mask_local_weight,
        )

    @staticmethod
    def build_log_vars(
            real_logits: torch.Tensor,
            fake_logits: torch.Tensor,
            loss_d: torch.Tensor,
            extra: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        log_vars = DinoFeatureDiscriminator.build_log_vars(real_logits, fake_logits, loss_d)
        if extra:
            log_vars.update(extra)
        if extra is not None and 'mask_fallback' in extra:
            log_vars['dino_gan_mask_fallback_rate'] = float(extra['mask_fallback'])
        return log_vars

    @staticmethod
    def build_generator_log_vars(
            fake_logits: torch.Tensor,
            loss_g: torch.Tensor,
            extra: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        log_vars = DinoFeatureDiscriminator.build_generator_log_vars(fake_logits, loss_g)
        if extra:
            log_vars.update(extra)
        return log_vars

    def forward(
            self,
            images: Optional[torch.Tensor] = None,
            real_images: Optional[torch.Tensor] = None,
            fake_images: Optional[torch.Tensor] = None,
            gan_mode: Optional[str] = None,
            step_indices: Optional[torch.Tensor] = None,
            crop_specs: Optional[Sequence[torch.Tensor]] = None,
            alpha: Optional[torch.Tensor] = None,
            cond_images: Optional[torch.Tensor] = None,
            source_images: Optional[torch.Tensor] = None):
        if cond_images is None:
            cond_images = source_images
        if self.condition_on_source and cond_images is None and gan_mode is not None:
            raise ValueError(
                'DinoAlphaMaskFeatureDiscriminator(condition_on_source=True) requires '
                'cond_images/source_images for D(x_src, x_target).')
        if (not self.condition_on_source) and cond_images is not None:
            cond_images = None
        cond_detached = None if cond_images is None else cond_images.detach()

        if gan_mode == 'discriminator':
            assert real_images is not None and fake_images is not None
            if step_indices is None:
                step_indices = self.make_step_indices(fake_images.shape[0], fake_images.device)
            if crop_specs is None:
                _, _, height, width = fake_images.shape
                crop_specs = self.sample_crop_specs(
                    fake_images.shape[0],
                    fake_images.device,
                    alpha=alpha,
                    image_height=height,
                    image_width=width)
            logits_real = self.logits_from_pixels(
                real_images.detach(),
                step_indices,
                crop_specs=crop_specs,
                requires_input_grad=False,
                cond_images=cond_detached)
            logits_fake = self.logits_from_pixels(
                fake_images.detach(),
                step_indices,
                crop_specs=crop_specs,
                requires_input_grad=False,
                cond_images=cond_detached)
            loss_d, extra = self._weighted_gan_loss(
                logits_real, logits_fake, gan_mode='discriminator')
            if self._last_crop_meta is not None:
                extra['mask_fallback'] = float(
                    self._last_crop_meta['mask_fallback'].float().mean())
                extra['local_enabled_rate'] = float(
                    self._last_crop_meta['local_enabled'].float().mean())
            self._last_gan_extra = extra
            return loss_d

        if gan_mode == 'generator':
            assert fake_images is not None
            if step_indices is None:
                step_indices = self.make_step_indices(fake_images.shape[0], fake_images.device)
            if crop_specs is None:
                _, _, height, width = fake_images.shape
                crop_specs = self.sample_crop_specs(
                    fake_images.shape[0],
                    fake_images.device,
                    alpha=alpha,
                    image_height=height,
                    image_width=width)
            logits_fake = self.logits_from_pixels(
                fake_images,
                step_indices,
                crop_specs=crop_specs,
                requires_input_grad=True,
                cond_images=cond_detached)
            loss_g, extra = self._weighted_gan_loss(
                logits_fake, logits_fake, gan_mode='generator')
            self._last_gan_extra = extra
            return loss_g

        if images is None:
            raise ValueError(
                'DinoAlphaMaskFeatureDiscriminator requires `images` when gan_mode is None.')
        if step_indices is None:
            step_indices = self.make_step_indices(images.shape[0], images.device)
        return self.logits_from_pixels(
            images,
            step_indices,
            crop_specs=crop_specs,
            requires_input_grad=False,
            cond_images=cond_detached)
