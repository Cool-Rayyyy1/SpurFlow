# Copyright (c) 2026 EditFlow contributors
"""FLUX.2-klein Softsign-01 alpha + step-2 PIID + source-cond DINO GAN.

Same GAN recipe as Kontext ``step2_alph_dino_gan``:
  - ``ArcFlowEditImitationStep2GAN``: standard random-segment PIID, then an
    *extra independent* 2-NFE rollout (path_epsilon at t=1 -> t=0) for the GAN
    fake endpoint + step-2 alpha (NOT split-stage shared unroll).
  - ``DinoAlphaMaskFeatureDiscriminator`` with condition_on_source
  - cat([DINO(ref), DINO(target)])
  - alpha-mask local crop from the rolled-out step-2 alpha

Teacher/student guidance stays Klein-specific (teacher classic CFG=4.0,
student cond-only) via ``LatentDiffusionFlux2KleinImageEdit``.
"""

from __future__ import annotations

from mmgen.models.builder import MODELS, build_module

from .latent_diffusion_flux2_klein_image_edit import LatentDiffusionFlux2KleinImageEdit
from .latent_diffusion_image_edit_gan import (
    LatentDiffusionImageEditSplitStageGAN,
    LatentDiffusionImageEditStep2AlphaDinoFeatureGAN,
)


@MODELS.register_module()
class LatentDiffusionFlux2KleinImageEditStep2AlphaDinoGAN(
        LatentDiffusionFlux2KleinImageEdit):
    """Klein Softsign alpha edit + independent step-2 rollout + alpha-mask DINO GAN."""

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
