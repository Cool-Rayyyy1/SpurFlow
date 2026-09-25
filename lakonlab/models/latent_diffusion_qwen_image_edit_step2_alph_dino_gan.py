# Copyright (c) 2026 EditFlow contributors
"""Qwen Softsign-01 alpha + step-2 PIID + source-cond DINO GAN.

Same GAN recipe as Kontext ``step2_alph_dino_gan``:
  - ``ArcFlowEditImitationStep2GAN``: standard random-segment PIID, then an
    *extra independent* 2-NFE rollout for the GAN fake endpoint + step-2 alpha
    (NOT split-stage shared unroll).
  - ``DinoAlphaMaskFeatureDiscriminator`` with condition_on_source
  - cat([DINO(ref), DINO(target)])
  - global + random local + alpha-mask local crops
"""

from __future__ import annotations

from mmgen.models.builder import MODELS, build_module

from .latent_diffusion_qwen_image_edit import LatentDiffusionQwenImageEdit
from .latent_diffusion_image_edit_gan import (
    LatentDiffusionImageEditSplitStageGAN,
    LatentDiffusionImageEditStep2AlphaDinoFeatureGAN,
)


@MODELS.register_module()
class LatentDiffusionQwenImageEditStep2AlphaDinoGAN(LatentDiffusionQwenImageEdit):
    """Qwen Softsign alpha edit + independent step-2 rollout + alpha-mask DINO GAN."""

    def __init__(self, *args, discriminator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = (
            build_module(discriminator) if discriminator is not None else None)

    _decode_rollout_latents_to_images = (
        LatentDiffusionImageEditSplitStageGAN._decode_rollout_latents_to_images)
    _match_spatial = staticmethod(
        LatentDiffusionImageEditStep2AlphaDinoFeatureGAN._match_spatial)
    _prepare_gan_real_images = staticmethod(
        LatentDiffusionImageEditStep2AlphaDinoFeatureGAN._prepare_gan_real_images)

    def _prepare_train_minibatch_args(self, data, running_status=None):
        bs, diffusion_args, diffusion_kwargs = super()._prepare_train_minibatch_args(
            data, running_status)
        diffusion_kwargs['return_step2_latent'] = True
        return bs, diffusion_args, diffusion_kwargs

    train_minibatch = LatentDiffusionImageEditStep2AlphaDinoFeatureGAN.train_minibatch
