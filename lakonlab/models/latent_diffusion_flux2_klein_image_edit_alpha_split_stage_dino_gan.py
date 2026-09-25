# Copyright (c) 2026 EditFlow contributors
"""FLUX.2-klein Softsign-01 alpha + split-stage PIID + source-cond DINO GAN.

Same GAN recipe as Qwen ``alph_dino_gan`` / Flux ``alph_dino_gan``:
  - DinoAlphaMaskFeatureDiscriminator with condition_on_source
  - cat([DINO(ref), DINO(target)])
  - alpha-mask local crop from step-2 alpha
but distillation is split-stage so the 2-NFE unroll is shared with GAN
(no extra independent rollout). Teacher/student guidance stays Klein-specific
(teacher classic CFG=4.0, student cond-only) via ``LatentDiffusionFlux2KleinImageEdit``.
"""

from __future__ import annotations

from mmgen.models.builder import MODELS, build_module

from .latent_diffusion_flux2_klein_image_edit import LatentDiffusionFlux2KleinImageEdit
from .latent_diffusion_image_edit_gan import (
    LatentDiffusionImageEditStep2AlphaDinoFeatureGAN,
    LatentDiffusionImageEditSplitStageGAN,
)


@MODELS.register_module()
class LatentDiffusionFlux2KleinImageEditAlphaSplitStageDinoGAN(
        LatentDiffusionFlux2KleinImageEdit):
    """Klein Softsign alpha edit + split-stage PIID + source-cond alpha-mask DINO GAN."""

    def __init__(self, *args, discriminator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = (
            build_module(discriminator) if discriminator is not None else None)

    _decode_rollout_latents_to_images = (
        LatentDiffusionImageEditSplitStageGAN._decode_rollout_latents_to_images)
    # Must re-wrap: assigning a @staticmethod via Class.fn drops the descriptor
    # and Python would bind `self` as the first arg on instance call.
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
