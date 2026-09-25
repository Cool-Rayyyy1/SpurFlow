# Copyright (c) 2026 EditFlow contributors

import torch
import torch.nn.functional as F

from typing import Dict, Optional

from mmgen.models.builder import MODELS, build_module

from .latent_diffusion_image_edit import LatentDiffusionImageEdit
from lakonlab.models.architecture.dinov3_discriminator import (
    DINOv3PatchDiscriminator,
    split_stage_gan_loss_scale,
)


def _set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def _discriminator_module(model):
    """Return the raw discriminator under DDP/FSDP wrappers."""
    discriminator = getattr(model, 'discriminator', None)
    if discriminator is None:
        return None
    while hasattr(discriminator, 'module'):
        inner = discriminator.module
        if inner is discriminator:
            break
        discriminator = inner
    return discriminator


@MODELS.register_module()
class LatentDiffusionImageEditSplitStageGAN(LatentDiffusionImageEdit):
    """Edit distillation with split-stage rollout and step-2 DINOv3 GAN."""

    def __init__(self, *args, discriminator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = build_module(discriminator) if discriminator is not None else None

    def _decode_rollout_latents_to_images(self, latents):
        """Match Kontext val/inference decode: unpatchify + VAE decode to RGB [0, 1]."""
        if self.vae is None:
            raise ValueError('VAE is required to decode latents for GAN training.')
        latents = self.unpatchify(latents)
        if hasattr(self.vae, 'dtype'):
            vae_dtype = self.vae.dtype
        else:
            vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(vae_dtype)
        return (self.vae.decode(latents).float() / 2 + 0.5).clamp(min=0, max=1)

    def _prepare_train_minibatch_args(self, data, running_status=None):
        bs, diffusion_args, diffusion_kwargs = super()._prepare_train_minibatch_args(
            data, running_status)
        diffusion_kwargs['return_step2_latent'] = True
        return bs, diffusion_args, diffusion_kwargs

    def train_minibatch(self, data, loss_scaler=None, running_status=None):
        bs, diffusion_args, diffusion_kwargs = self._prepare_train_minibatch_args(
            data, running_status)

        outputs = self.diffusion(
            *diffusion_args, return_loss=True, **diffusion_kwargs)
        if isinstance(outputs, tuple) and len(outputs) == 3:
            loss_diffusion, log_vars, extra = outputs
        else:
            loss_diffusion, log_vars = outputs
            extra = dict()

        step2_latent = extra.get('step2_latent')
        real_images = data.get('edited_images')
        w_gan = self.train_cfg.get('split_stage_gan_loss_weight', 1.0)
        gan_scale = split_stage_gan_loss_scale(running_status, self.train_cfg)
        log_vars['gan_loss_scale'] = gan_scale

        if (
                gan_scale > 0
                and self.discriminator is not None
                and step2_latent is not None
                and real_images is not None):
            fake_images = self._decode_rollout_latents_to_images(step2_latent)

            # D step: fake is detached, so no grad flows into diffusion.
            loss_d = self.discriminator(
                real_images=real_images,
                fake_images=fake_images.detach(),
                gan_mode='discriminator')
            loss_d = gan_scale * loss_d
            if loss_scaler is None:
                loss_d.backward()
            else:
                loss_scaler.scale(loss_d).backward()
            with torch.no_grad():
                real_logits = self.discriminator(real_images)
                fake_logits = self.discriminator(fake_images.detach())
                d_log_vars = DINOv3PatchDiscriminator.build_log_vars(
                    real_logits, fake_logits, loss_d)
            log_vars.update(d_log_vars)

            # G step: freeze D. Use one backward for PIID+GAN (required under FSDP;
            # split backward frees flat-param storage after the first pass). GAN grad
            # routing to step-2 only is handled by detach in the rollout forward.
            _set_requires_grad(self.discriminator, False)
            loss_g_gan = self.discriminator(fake_images=fake_images, gan_mode='generator')
            loss_generator = loss_diffusion + (w_gan * gan_scale) * loss_g_gan
            if loss_scaler is None:
                loss_generator.backward()
            else:
                loss_scaler.scale(loss_generator).backward()
            _set_requires_grad(self.discriminator, True)
            with torch.no_grad():
                fake_logits = self.discriminator(fake_images)
                g_log_vars = DINOv3PatchDiscriminator.build_generator_log_vars(
                    fake_logits, loss_g_gan)
            log_vars.update(g_log_vars)
            log_vars['loss'] = float((
                loss_diffusion.detach() + (w_gan * gan_scale) * loss_g_gan.detach()))
        elif isinstance(loss_diffusion, torch.Tensor) and loss_diffusion.requires_grad:
            if loss_scaler is None:
                loss_diffusion.backward()
            else:
                loss_scaler.scale(loss_diffusion).backward()

        return log_vars, bs


@MODELS.register_module()
class LatentDiffusionImageEditStep2GAN(LatentDiffusionImageEditSplitStageGAN):
    """Standard PIID edit distillation + step-2 DINOv3 GAN (non split-stage rollout).

    Validation uses the same 2-NFE ``forward_test`` path as the GAN rollout helper.
    """

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        cfg_override = dict(
            nfe=self.train_cfg.get('nfe', self.test_cfg.get('nfe', 2)),
            timestep_ratio=self.train_cfg.get(
                'timestep_ratio', self.test_cfg.get('timestep_ratio', 1.0)),
            total_substeps=self.train_cfg.get(
                'total_substeps', self.test_cfg.get('total_substeps', 128)),
            distilled_guidance_scale=self.train_cfg.get(
                'distilled_guidance_scale',
                self.test_cfg.get('distilled_guidance_scale', 3.5)),
        )
        cfg_override.update(test_cfg_override)
        return super().val_step(data, test_cfg_override=cfg_override, **kwargs)


@MODELS.register_module()
class LatentDiffusionImageEditStep2DinoFeatureGAN(LatentDiffusionImageEditStep2GAN):
    """Step-2 PIID + TDM-style DINO feature GAN with shared global/local crops."""

    def train_minibatch(self, data, loss_scaler=None, running_status=None):
        bs, diffusion_args, diffusion_kwargs = self._prepare_train_minibatch_args(
            data, running_status)

        outputs = self.diffusion(
            *diffusion_args, return_loss=True, **diffusion_kwargs)
        if isinstance(outputs, tuple) and len(outputs) == 3:
            loss_diffusion, log_vars, extra = outputs
        else:
            loss_diffusion, log_vars = outputs
            extra = dict()

        step2_latent = extra.get('step2_latent')
        real_images = data.get('edited_images')
        source_images = data.get('source_images')
        w_gan = self.train_cfg.get('split_stage_gan_loss_weight', 1.0)
        gan_scale = split_stage_gan_loss_scale(running_status, self.train_cfg)
        log_vars['gan_loss_scale'] = gan_scale

        if (
                gan_scale > 0
                and self.discriminator is not None
                and step2_latent is not None
                and real_images is not None):
            fake_images = self._decode_rollout_latents_to_images(step2_latent)
            device = fake_images.device
            disc = _discriminator_module(self)
            crop_specs = disc.sample_crop_specs(bs, device)
            step_indices = disc.make_step_indices(bs, device)
            cond_images = None
            if getattr(disc, 'condition_on_source', False):
                if source_images is None:
                    raise ValueError(
                        'condition_on_source=True requires data["source_images"] '
                        'for D(x_src, x_target).')
                cond_images = source_images
                if cond_images.shape[-2:] != fake_images.shape[-2:]:
                    cond_images = F.interpolate(
                        cond_images.float(),
                        size=fake_images.shape[-2:],
                        mode='bilinear',
                        align_corners=False).to(dtype=fake_images.dtype)

            loss_d = self.discriminator(
                real_images=real_images,
                fake_images=fake_images.detach(),
                gan_mode='discriminator',
                crop_specs=crop_specs,
                step_indices=step_indices,
                cond_images=cond_images)
            loss_d = gan_scale * loss_d
            if loss_scaler is None:
                loss_d.backward()
            else:
                loss_scaler.scale(loss_d).backward()
            with torch.no_grad():
                real_logits = self.discriminator(
                    real_images,
                    step_indices=step_indices,
                    crop_specs=crop_specs,
                    cond_images=cond_images)
                fake_logits = self.discriminator(
                    fake_images.detach(),
                    step_indices=step_indices,
                    crop_specs=crop_specs,
                    cond_images=cond_images)
                d_log_vars = disc.build_log_vars(
                    real_logits, fake_logits, loss_d)
            log_vars.update(d_log_vars)

            _set_requires_grad(self.discriminator, False)
            loss_g_gan = self.discriminator(
                fake_images=fake_images,
                gan_mode='generator',
                crop_specs=crop_specs,
                step_indices=step_indices,
                cond_images=cond_images)
            loss_generator = loss_diffusion + (w_gan * gan_scale) * loss_g_gan
            if loss_scaler is None:
                loss_generator.backward()
            else:
                loss_scaler.scale(loss_generator).backward()
            _set_requires_grad(self.discriminator, True)
            with torch.no_grad():
                fake_logits = self.discriminator(
                    fake_images,
                    step_indices=step_indices,
                    crop_specs=crop_specs,
                    cond_images=cond_images)
                g_log_vars = disc.build_generator_log_vars(
                    fake_logits, loss_g_gan)
            log_vars.update(g_log_vars)
            log_vars['loss'] = float((
                loss_diffusion.detach() + (w_gan * gan_scale) * loss_g_gan.detach()))
        elif isinstance(loss_diffusion, torch.Tensor) and loss_diffusion.requires_grad:
            if loss_scaler is None:
                loss_diffusion.backward()
            else:
                loss_scaler.scale(loss_diffusion).backward()

        return log_vars, bs


@MODELS.register_module()
class LatentDiffusionImageEditStep2AlphaDinoFeatureGAN(LatentDiffusionImageEditStep2DinoFeatureGAN):
    """Step-2 PIID + alpha-guided mask local crop DINO feature GAN.

    With ``condition_on_source=True``, D scores paired features
    ``cat([DINO(ref), DINO(target)])``:
    D(ref, x_edit)=1, D(ref, x_student)=0. Unpaired reals are disabled in that mode.
    """

    @staticmethod
    def _match_spatial(images: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] == ref.shape[-2:]:
            return images.to(device=ref.device, dtype=ref.dtype)
        out = F.interpolate(
            images.float(),
            size=ref.shape[-2:],
            mode='bilinear',
            align_corners=False)
        return out.to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _prepare_gan_real_images(
            real_images: torch.Tensor,
            crop_meta: Optional[Dict[str, torch.Tensor]],
            unpaired_images: Optional[torch.Tensor] = None,
            *,
            allow_unpaired: bool = True) -> torch.Tensor:
        """Case A (~local_enabled): dataset-sampled unpaired edit; Case B: paired edit.

        Unpaired reals are skipped when ``allow_unpaired=False`` (source-conditional GAN),
        because D(ref_i, edit_j) must not be labeled real.
        """
        if (not allow_unpaired) or crop_meta is None or 'local_enabled' not in crop_meta:
            return real_images
        local_enabled = crop_meta['local_enabled'].to(
            device=real_images.device, dtype=torch.bool)
        unpaired_mask = ~local_enabled
        if not bool(unpaired_mask.any()) or unpaired_images is None:
            return real_images
        real_for_gan = real_images.clone()
        unpaired = unpaired_images.to(
            device=real_images.device, dtype=real_images.dtype)
        if unpaired.shape[-2:] != real_images.shape[-2:]:
            unpaired = F.interpolate(
                unpaired.float(),
                size=real_images.shape[-2:],
                mode='bicubic',
                align_corners=False,
                antialias=True,
            ).clamp(0.0, 1.0).to(dtype=real_images.dtype)
        real_for_gan[unpaired_mask] = unpaired[unpaired_mask]
        return real_for_gan

    def train_minibatch(self, data, loss_scaler=None, running_status=None):
        bs, diffusion_args, diffusion_kwargs = self._prepare_train_minibatch_args(
            data, running_status)

        outputs = self.diffusion(
            *diffusion_args, return_loss=True, **diffusion_kwargs)
        if isinstance(outputs, tuple) and len(outputs) == 3:
            loss_diffusion, log_vars, extra = outputs
        else:
            loss_diffusion, log_vars = outputs
            extra = dict()

        step2_latent = extra.get('step2_latent')
        step2_alpha = extra.get('step2_alpha')
        real_images = data.get('edited_images')
        unpaired_images = data.get('unpaired_edited_images')
        source_images = data.get('source_images')
        w_gan = self.train_cfg.get('split_stage_gan_loss_weight', 1.0)
        gan_scale = split_stage_gan_loss_scale(running_status, self.train_cfg)
        log_vars['gan_loss_scale'] = gan_scale

        if (
                gan_scale > 0
                and self.discriminator is not None
                and step2_latent is not None
                and real_images is not None):
            fake_images = self._decode_rollout_latents_to_images(step2_latent)
            device = fake_images.device
            disc = _discriminator_module(self)
            step_indices = disc.make_step_indices(bs, device)
            _, _, height, width = fake_images.shape
            alpha_for_crop = None if step2_alpha is None else step2_alpha.detach()
            crop_specs = disc.sample_crop_specs(
                bs, device,
                alpha=alpha_for_crop,
                image_height=height,
                image_width=width)
            crop_meta = getattr(disc, '_last_crop_meta', None)
            condition_on_source = bool(getattr(disc, 'condition_on_source', False))
            gan_real_images = self._prepare_gan_real_images(
                real_images,
                crop_meta,
                unpaired_images=unpaired_images,
                allow_unpaired=not condition_on_source)
            cond_images = None
            if condition_on_source:
                if source_images is None:
                    raise ValueError(
                        'condition_on_source=True requires data["source_images"] '
                        'for D(x_src, x_target).')
                cond_images = self._match_spatial(source_images, fake_images)

            loss_d = self.discriminator(
                real_images=gan_real_images,
                fake_images=fake_images.detach(),
                gan_mode='discriminator',
                crop_specs=crop_specs,
                step_indices=step_indices,
                alpha=alpha_for_crop,
                cond_images=cond_images)
            loss_d = gan_scale * loss_d
            if loss_scaler is None:
                loss_d.backward()
            else:
                loss_scaler.scale(loss_d).backward()
            # Reuse D logits from the training forward (no second DINO pass before G).
            d_extra = getattr(disc, '_last_gan_extra', None) or {}
            if (not condition_on_source
                    and crop_meta is not None
                    and 'local_enabled' in crop_meta):
                d_extra['gan_unpaired_real_rate'] = float(
                    (~crop_meta['local_enabled']).float().mean())
            d_extra['dino_gan_condition_on_source'] = float(condition_on_source)
            real_logits = getattr(disc, '_last_logits_real', None)
            fake_logits_d = getattr(disc, '_last_logits_fake', None)
            if real_logits is not None and fake_logits_d is not None:
                d_log_vars = disc.build_log_vars(
                    real_logits, fake_logits_d, loss_d, extra=d_extra)
            else:
                d_log_vars = dict(d_extra)
                d_log_vars['loss_d'] = float(loss_d.detach())
            log_vars.update(d_log_vars)
            # Free D activations before the heavy PIID+GAN backward (FSDP unshard peak).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _set_requires_grad(self.discriminator, False)
            loss_g_gan = self.discriminator(
                fake_images=fake_images,
                gan_mode='generator',
                crop_specs=crop_specs,
                step_indices=step_indices,
                alpha=alpha_for_crop,
                cond_images=cond_images)
            loss_generator = loss_diffusion + (w_gan * gan_scale) * loss_g_gan
            if loss_scaler is None:
                loss_generator.backward()
            else:
                loss_scaler.scale(loss_generator).backward()
            _set_requires_grad(self.discriminator, True)
            g_extra = getattr(disc, '_last_gan_extra', None) or {}
            fake_logits_g = getattr(disc, '_last_logits_fake', None)
            if fake_logits_g is not None:
                g_log_vars = disc.build_generator_log_vars(
                    fake_logits_g, loss_g_gan, extra=g_extra)
            else:
                g_log_vars = dict(g_extra)
                g_log_vars['loss_g_gan'] = float(loss_g_gan.detach())
            log_vars.update(g_log_vars)
            log_vars['loss'] = float((
                loss_diffusion.detach() + (w_gan * gan_scale) * loss_g_gan.detach()))
        elif isinstance(loss_diffusion, torch.Tensor) and loss_diffusion.requires_grad:
            if loss_scaler is None:
                loss_diffusion.backward()
            else:
                loss_scaler.scale(loss_diffusion).backward()

        return log_vars, bs


@MODELS.register_module()
class LatentDiffusionImageEditSplitStageDinoFeatureGAN(LatentDiffusionImageEditSplitStageGAN):
    """Split-stage rollout PIID + TDM-style DINO feature GAN on step-2 endpoint."""

    train_minibatch = LatentDiffusionImageEditStep2DinoFeatureGAN.train_minibatch


@MODELS.register_module()
class LatentDiffusionImageEditSplitStageAlphaDinoFeatureGAN(LatentDiffusionImageEditSplitStageGAN):
    """Split-stage rollout PIID + source-conditional alpha-mask DINO GAN.

    The fake image is the decoded endpoint of the same 2-NFE rollout used for
    the step-2 PIID loss. The step-2 alpha map selects the local crop.
    """

    _match_spatial = staticmethod(
        LatentDiffusionImageEditStep2AlphaDinoFeatureGAN._match_spatial)
    _prepare_gan_real_images = staticmethod(
        LatentDiffusionImageEditStep2AlphaDinoFeatureGAN._prepare_gan_real_images)
    train_minibatch = LatentDiffusionImageEditStep2AlphaDinoFeatureGAN.train_minibatch
