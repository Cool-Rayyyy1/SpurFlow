# Copyright (c) 2026 EditFlow contributors

import torch

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

            loss_d = self.discriminator(
                real_images=real_images,
                fake_images=fake_images.detach(),
                gan_mode='discriminator',
                crop_specs=crop_specs,
                step_indices=step_indices)
            loss_d = gan_scale * loss_d
            if loss_scaler is None:
                loss_d.backward()
            else:
                loss_scaler.scale(loss_d).backward()
            with torch.no_grad():
                real_logits = self.discriminator(
                    real_images, step_indices=step_indices, crop_specs=crop_specs)
                fake_logits = self.discriminator(
                    fake_images.detach(), step_indices=step_indices, crop_specs=crop_specs)
                d_log_vars = disc.build_log_vars(
                    real_logits, fake_logits, loss_d)
            log_vars.update(d_log_vars)

            _set_requires_grad(self.discriminator, False)
            loss_g_gan = self.discriminator(
                fake_images=fake_images,
                gan_mode='generator',
                crop_specs=crop_specs,
                step_indices=step_indices)
            loss_generator = loss_diffusion + (w_gan * gan_scale) * loss_g_gan
            if loss_scaler is None:
                loss_generator.backward()
            else:
                loss_scaler.scale(loss_generator).backward()
            _set_requires_grad(self.discriminator, True)
            with torch.no_grad():
                fake_logits = self.discriminator(
                    fake_images, step_indices=step_indices, crop_specs=crop_specs)
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
class LatentDiffusionImageEditSplitStageDinoFeatureGAN(LatentDiffusionImageEditSplitStageGAN):
    """Split-stage rollout PIID + TDM-style DINO feature GAN on step-2 endpoint."""

    train_minibatch = LatentDiffusionImageEditStep2DinoFeatureGAN.train_minibatch
