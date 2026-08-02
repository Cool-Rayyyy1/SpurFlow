# Copyright (c) 2026 EditFlow contributors
"""Qwen-Image-Edit fixed-eps split-stage PIID + DINO feature GAN (no alpha).

Same unroll sharing as ``alph_dino_gan`` / Qwen alpha split-stage GAN:
  - ``ArcFlowEditImitationSplitStageGAN``: step-1/step-2 PIID in one 2-NFE chain
  - GAN endpoint = integrate the same step-2 policy to t=0 (no extra independent rollout)
  - ``DinoFeatureDiscriminator`` with shared random global/local crops
  - ``condition_on_source=True``: ``cat([DINO(ref), DINO(target)])``
Unlike alpha version: no ``proj_out_alpha``, no alpha-mask local crop.
"""

from __future__ import annotations

from mmgen.models.builder import MODELS, build_module

from .latent_diffusion_qwen_image_edit import LatentDiffusionQwenImageEdit
from .latent_diffusion_image_edit_gan import (
    LatentDiffusionImageEditSplitStageGAN,
    LatentDiffusionImageEditStep2DinoFeatureGAN,
)


@MODELS.register_module()
class LatentDiffusionQwenImageEditSplitStageDinoFeatureGAN(LatentDiffusionQwenImageEdit):
    """Qwen fixed-eps edit + split-stage PIID + DINO feature GAN (no alpha)."""

    def __init__(self, *args, discriminator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = (
            build_module(discriminator) if discriminator is not None else None)

    _decode_rollout_latents_to_images = (
        LatentDiffusionImageEditSplitStageGAN._decode_rollout_latents_to_images)

    def _prepare_train_minibatch_args(self, data, running_status=None):
        bs, diffusion_args, diffusion_kwargs = super()._prepare_train_minibatch_args(
            data, running_status)
        diffusion_kwargs['return_step2_latent'] = True
        return bs, diffusion_args, diffusion_kwargs

    train_minibatch = LatentDiffusionImageEditStep2DinoFeatureGAN.train_minibatch
