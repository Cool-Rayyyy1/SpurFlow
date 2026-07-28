# Copyright (c) 2026 EditFlow contributors
"""Qwen-Image-Edit alpha + split-stage PIID + source-cond DINO GAN.

Same GAN recipe as Flux ``alph_dino_gan``:
  - DinoAlphaMaskFeatureDiscriminator with condition_on_source
  - cat([DINO(ref), DINO(target)])
  - alpha-mask local crop from step-2 alpha (30% random local)
but distillation is split-stage so the 2-NFE unroll is shared with GAN
(no extra independent rollout). ``gan_grad_step2_only=False`` keeps GAN
grads through both steps.
"""

from __future__ import annotations

from mmgen.models.builder import MODELS, build_module

from .latent_diffusion_qwen_image_edit import LatentDiffusionQwenImageEdit
from .latent_diffusion_image_edit_gan import (
    LatentDiffusionImageEditStep2AlphaDinoFeatureGAN,
    LatentDiffusionImageEditSplitStageGAN,
)


@MODELS.register_module()
class LatentDiffusionQwenImageEditAlphaSplitStageDinoGAN(LatentDiffusionQwenImageEdit):
    """Qwen alpha edit + split-stage PIID + source-cond alpha-mask DINO GAN."""

    def __init__(self, *args, discriminator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = (
            build_module(discriminator) if discriminator is not None else None)

    _decode_rollout_latents_to_images = (
        LatentDiffusionImageEditSplitStageGAN._decode_rollout_latents_to_images)
    _match_spatial = LatentDiffusionImageEditStep2AlphaDinoFeatureGAN._match_spatial
    _prepare_gan_real_images = (
        LatentDiffusionImageEditStep2AlphaDinoFeatureGAN._prepare_gan_real_images)

    def _prepare_train_minibatch_args(self, data, running_status=None):
        bs, diffusion_args, diffusion_kwargs = super()._prepare_train_minibatch_args(
            data, running_status)
        diffusion_kwargs['return_step2_latent'] = True
        return bs, diffusion_args, diffusion_kwargs

    train_minibatch = LatentDiffusionImageEditStep2AlphaDinoFeatureGAN.train_minibatch
