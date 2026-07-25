# Copyright (c) 2026 EditFlow contributors
"""Alpha PIID + alpha-mask local-crop latent HF loss (no RGB x0, no GAN)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from mmgen.models.builder import MODELS
from torchvision.transforms import functional as TVF

from .latent_diffusion_image_edit_gan import LatentDiffusionImageEditStep2GAN
from lakonlab.models.architecture.dino_feature_discriminator import (
    apply_dino_crop_specs,
    sample_dino_alpha_mask_crop_specs,
)


def _gaussian_high_freq(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """H(x) = x - GaussianBlur(x), differentiable (latent or RGB)."""
    sigma = float(max(sigma, 1e-4))
    k = int(2 * math.ceil(3.0 * sigma) + 1)
    if k % 2 == 0:
        k += 1
    blurred = TVF.gaussian_blur(x.float(), kernel_size=[k, k], sigma=[sigma, sigma])
    return x.float() - blurred


@MODELS.register_module()
class LatentDiffusionImageEditStep2AlphaX0HF(LatentDiffusionImageEditStep2GAN):
    """Alpha PIID + latent edit-HF on alpha local crop (no GAN, no RGB x0).

    ``L = L_PIID + λ_hf ||H(Δ_S) - H(Δ_GT)||_1``

    where latents are unpatchified, cropped by step-2 alpha, and

      Δ_S = ẑ0_S - z_src,  Δ_GT = z_edit - z_src,
      H(z) = z - GaussianBlur(z).
    """

    def __init__(self, *args, discriminator=None, **kwargs):
        super().__init__(*args, discriminator=None, **kwargs)

    def _sample_alpha_local_crop_spec(
            self,
            batch_size: int,
            device: torch.device,
            *,
            alpha: Optional[torch.Tensor],
            image_height: int,
            image_width: int,
    ) -> Tuple[torch.Tensor, dict]:
        cfg = self.train_cfg
        crop_specs, meta = sample_dino_alpha_mask_crop_specs(
            batch_size,
            device,
            alpha=alpha,
            image_height=int(image_height),
            image_width=int(image_width),
            p_disable_local=float(cfg.get('alpha_crop_p_disable_local', 0.0)),
            num_global_crops=1,
            global_crop_scale=tuple(cfg.get('alpha_crop_global_scale', (0.5, 1.0))),
            local_crop_scale=tuple(cfg.get('alpha_crop_local_scale', (0.08, 0.35))),
            crop_aspect_ratio=tuple(cfg.get('alpha_crop_aspect_ratio', (0.75, 1.3333333333))),
            edit_is_low_alpha=bool(cfg.get('edit_is_low_alpha', True)),
            alpha_smooth_sigma=float(cfg.get('alpha_smooth_sigma', 2.0)),
            mass_threshold_percentile=float(cfg.get('mass_threshold_percentile', 30.0)),
            mass_coverage_min=float(cfg.get('mass_coverage_min', 0.85)),
            mass_coverage_max=float(cfg.get('mass_coverage_max', 0.90)),
            union_area_max_ratio=float(cfg.get('union_area_max_ratio', 0.35)),
            bbox_expand_factor=float(cfg.get('bbox_expand_factor', 1.1)),
            min_crop_area_ratio=float(cfg.get('min_crop_area_ratio', 0.01)),
            max_crop_area_ratio=float(cfg.get('max_crop_area_ratio', 0.35)),
            min_edit_mass_ratio=float(cfg.get('min_edit_mass_ratio', 0.002)),
            min_component_pixels=int(cfg.get('min_component_pixels', 16)),
            hot_mass_frac=float(cfg.get('hot_mass_frac', 0.40)),
        )
        return crop_specs[1], meta

    def _crop_latents(
            self,
            latents: torch.Tensor,
            local_crop_spec: torch.Tensor,
            crop_size: int) -> torch.Tensor:
        """Crop spatial latents; never clamp (latents are not in [0, 1])."""
        crops = apply_dino_crop_specs(
            latents,
            [local_crop_spec],
            num_global_crops=0,
            global_input_size=int(crop_size),
            local_input_size=int(crop_size),
            clamp_pixels=False,
        )
        return crops[0]

    def train_minibatch(self, data, loss_scaler=None, running_status=None):
        bs, diffusion_args, diffusion_kwargs = self._prepare_train_minibatch_args(
            data, running_status)

        # Patchified GT edit / source latents (VAE encode is no-grad).
        z_edit_patch = diffusion_args[0]
        z_src_patch = diffusion_kwargs.get('x_ref')

        outputs = self.diffusion(
            *diffusion_args, return_loss=True, **diffusion_kwargs)
        if isinstance(outputs, tuple) and len(outputs) == 3:
            loss_diffusion, log_vars, extra = outputs
        else:
            loss_diffusion, log_vars = outputs
            extra = dict()

        step2_latent = extra.get('step2_latent')
        step2_alpha = extra.get('step2_alpha')

        # RGB x0 L1 removed; only latent HF remains.
        w_hf = float(self.train_cfg.get('hf_loss_weight', 0.3))
        crop_size = int(self.train_cfg.get('latent_crop_size',
                                           self.train_cfg.get('edit_crop_size', 64)))
        hf_sigma = float(self.train_cfg.get('hf_blur_sigma', 1.0))

        log_vars['loss_diffusion'] = float(loss_diffusion.detach())

        loss_hf = None
        if (
                w_hf > 0
                and step2_latent is not None
                and z_src_patch is not None
                and z_edit_patch is not None):
            z_s = self.unpatchify(step2_latent)
            z_e = self.unpatchify(z_edit_patch.detach())
            z_r = self.unpatchify(z_src_patch.detach())
            if z_e.shape[-2:] != z_s.shape[-2:]:
                z_e = F.interpolate(
                    z_e.float(), size=z_s.shape[-2:], mode='bilinear',
                    align_corners=False).to(dtype=z_s.dtype)
            if z_r.shape[-2:] != z_s.shape[-2:]:
                z_r = F.interpolate(
                    z_r.float(), size=z_s.shape[-2:], mode='bilinear',
                    align_corners=False).to(dtype=z_s.dtype)

            alpha_for_crop = None if step2_alpha is None else step2_alpha.detach()
            local_spec, crop_meta = self._sample_alpha_local_crop_spec(
                bs,
                z_s.device,
                alpha=alpha_for_crop,
                image_height=z_s.shape[-2],
                image_width=z_s.shape[-1],
            )
            if crop_meta is not None and 'local_enabled' in crop_meta:
                log_vars['alpha_local_enabled_rate'] = float(
                    crop_meta['local_enabled'].float().mean())

            z_s_c = self._crop_latents(z_s, local_spec, crop_size)
            z_e_c = self._crop_latents(z_e, local_spec, crop_size)
            z_r_c = self._crop_latents(z_r, local_spec, crop_size)

            delta_s = z_s_c - z_r_c
            delta_gt = z_e_c - z_r_c
            h_s = _gaussian_high_freq(delta_s, hf_sigma)
            h_gt = _gaussian_high_freq(delta_gt, hf_sigma)
            loss_hf = (h_s - h_gt).abs().mean()
            log_vars['loss_hf_edit'] = float(loss_hf.detach())

        loss_total = loss_diffusion
        if loss_hf is not None:
            loss_total = loss_total + w_hf * loss_hf

        if isinstance(loss_total, torch.Tensor) and loss_total.requires_grad:
            if loss_scaler is None:
                loss_total.backward()
            else:
                loss_scaler.scale(loss_total).backward()

        total = float(loss_diffusion.detach())
        if loss_hf is not None:
            total = total + float(w_hf * loss_hf.detach())
        log_vars['loss'] = total
        return log_vars, bs
