# Copyright (c) 2026 EditFlow contributors
"""Step-2 DINO-GAN + final-image LPIPS for sharper edit regions."""

import torch
import torch.nn.functional as F

from mmgen.models.builder import MODELS

from .latent_diffusion_image_edit_gan import (
    LatentDiffusionImageEditStep2DinoFeatureGAN,
    _discriminator_module,
    _set_requires_grad,
)
from lakonlab.models.architecture.dinov3_discriminator import split_stage_gan_loss_scale
from .losses.perceptual_edit_loss import DEFAULT_LPIPS_WEIGHTS, LocalLPIPS


@MODELS.register_module()
class LatentDiffusionImageEditStep2DinoFeatureGANLpips(
        LatentDiffusionImageEditStep2DinoFeatureGAN):
    """Step-2 DINO-feature GAN (+ optional LPIPS) for local refinement.

    Discriminator targets (when ``condition_on_source=True``):

      D(x_src, x_gt_edit) -> 1
      D(x_src, x_student) -> 0

    Fake images come from the final 2-NFE decode (``step2_latent``). Edit
    direction stays on distill; GAN only refines sharpness/texture relative to
    the source.
    """

    def __init__(self, *args, lpips=None, **kwargs):
        super().__init__(*args, **kwargs)
        lpips = dict(lpips or {})
        self.lpips = LocalLPIPS(
            weights_path=lpips.get('weights_path', DEFAULT_LPIPS_WEIGHTS),
            vgg_weights_path=lpips.get(
                'vgg_weights_path',
                '/mnt/afs_zhangyunzhe/pretrained_models/lpips/vgg16-397923af.pth'),
            spatial=bool(lpips.get('spatial', False)),
        )
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

    def _resize_for_perceptual(self, images, size):
        if images.shape[-1] == size and images.shape[-2] == size:
            return images
        return F.interpolate(
            images, size=(size, size), mode='bilinear', align_corners=False)

    @staticmethod
    def _match_spatial(images: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] == ref.shape[-2:]:
            return images
        out = F.interpolate(
            images.float(),
            size=ref.shape[-2:],
            mode='bilinear',
            align_corners=False)
        return out.to(dtype=ref.dtype)

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
        # GAN/LPIPS real is always the paired GT edited target — never source alone.
        real_key = str(self.train_cfg.get('gan_real_key', 'edited_images'))
        if real_key == 'source_images':
            raise ValueError(
                'gan_real_key=source_images is not allowed; use source only as '
                'D conditioner: D(x_src, x_edit)=1, D(x_src, x_student)=0.')
        real_images = data.get(real_key)
        source_images = data.get('source_images')
        w_gan = float(self.train_cfg.get('split_stage_gan_loss_weight', 1.0))
        w_lpips = float(self.train_cfg.get('lpips_loss_weight', 0.0))
        perc_size = int(self.train_cfg.get('perceptual_image_size', 256))
        gan_scale = split_stage_gan_loss_scale(running_status, self.train_cfg)
        log_vars['gan_loss_scale'] = gan_scale
        log_vars['loss_diffusion'] = float(loss_diffusion.detach())

        fake_images = None
        if step2_latent is not None and real_images is not None and (
                (gan_scale > 0 and self.discriminator is not None) or w_lpips > 0):
            fake_images = self._decode_rollout_latents_to_images(step2_latent)

        loss_g_gan = None
        if (
                gan_scale > 0
                and self.discriminator is not None
                and fake_images is not None
                and real_images is not None):
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
                cond_images = self._match_spatial(source_images, fake_images)

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
            with torch.no_grad():
                fake_logits = self.discriminator(
                    fake_images,
                    step_indices=step_indices,
                    crop_specs=crop_specs,
                    cond_images=cond_images)
                g_log_vars = disc.build_generator_log_vars(
                    fake_logits, loss_g_gan)
            log_vars.update(g_log_vars)

        loss_lpips = None
        if w_lpips > 0 and fake_images is not None and real_images is not None:
            target = self._resize_for_perceptual(
                real_images.float().clamp(0.0, 1.0), perc_size)
            pred = self._resize_for_perceptual(fake_images, perc_size)
            loss_lpips = self.lpips(pred, target)
            log_vars['loss_lpips'] = float(loss_lpips.detach())

        loss_generator = loss_diffusion
        if loss_g_gan is not None:
            loss_generator = loss_generator + (w_gan * gan_scale) * loss_g_gan
        if loss_lpips is not None:
            loss_generator = loss_generator + w_lpips * loss_lpips

        if isinstance(loss_generator, torch.Tensor) and loss_generator.requires_grad:
            if loss_scaler is None:
                loss_generator.backward()
            else:
                loss_scaler.scale(loss_generator).backward()

        if loss_g_gan is not None:
            _set_requires_grad(self.discriminator, True)

        total = float(loss_diffusion.detach())
        if loss_g_gan is not None:
            total = total + float((w_gan * gan_scale) * loss_g_gan.detach())
        if loss_lpips is not None:
            total = total + float(w_lpips * loss_lpips.detach())
        log_vars['loss'] = total
        return log_vars, bs
