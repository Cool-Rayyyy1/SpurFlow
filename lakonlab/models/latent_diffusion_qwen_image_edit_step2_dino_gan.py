# Copyright (c) 2026 SpurFlow contributors
"""Qwen-Image-Edit fixed-eps PIID + step-2 TDM-style DINO feature GAN.

Counterpart of Flux ``step2_dino_gan`` on Qwen (no alpha head):
  - ``ArcFlowEditImitationStep2GAN`` (standard 2-NFE PIID + step2 latent)
  - ``DinoFeatureDiscriminator`` with shared global/local crops
  - ``gan_grad_step2_only=True`` by default (GAN grads through final NFE only)

Keeps Qwen VL prompt encoding / CFG teacher path from
``LatentDiffusionQwenImageEdit``; reuses Flux GAN train_minibatch helpers.
"""

from __future__ import annotations

from mmgen.models.builder import MODELS, build_module

from .latent_diffusion_qwen_image_edit import LatentDiffusionQwenImageEdit
from .latent_diffusion_image_edit_gan import (
    LatentDiffusionImageEditSplitStageGAN,
    LatentDiffusionImageEditStep2DinoFeatureGAN,
)


@MODELS.register_module()
class LatentDiffusionQwenImageEditStep2DinoFeatureGAN(LatentDiffusionQwenImageEdit):
    """Qwen fixed-eps edit + step-2 DINO feature GAN (no alpha)."""

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
