# Copyright (c) 2026 EditFlow contributors

"""Dual-LoRA teacher-x0 + step-1-only LPIPS/DINO.

Own wrapper for ``ArcFlowEditImitationSplitStageDualLoraTeacherX0``.
"""

import torch
import torch.nn.functional as F

from mmgen.models.builder import MODELS

from .latent_diffusion_image_edit import LatentDiffusionImageEdit
from .losses.perceptual_edit_loss import (
    DEFAULT_DINO_WEIGHTS,
    DEFAULT_LPIPS_WEIGHTS,
    FrozenDinoFeatureLoss,
    LocalLPIPS,
)


@MODELS.register_module()
class LatentDiffusionImageEditDualLoraTeacherX0Lpips(LatentDiffusionImageEdit):
    """Dual-LoRA teacher-x0 training with LPIPS/DINO on **step-1 only**.

    Expects diffusion ``ArcFlowEditImitationSplitStageDualLoraTeacherX0`` to
    return ``x0_hat`` from step-1:

        x0_hat = x_ref + pred_delta_step1

    Perceptual losses are applied after VAE decode vs ``edited_images``.
    Step-2 PIID-x0 is unchanged and does **not** receive LPIPS/DINO.
    """

    def __init__(
            self,
            *args,
            lpips=None,
            dino_loss=None,
            **kwargs):
        super().__init__(*args, **kwargs)
        lpips = dict(lpips or {})
        dino_loss = dict(dino_loss or {})
        self.lpips = LocalLPIPS(
            weights_path=lpips.get('weights_path', DEFAULT_LPIPS_WEIGHTS),
            vgg_weights_path=lpips.get(
                'vgg_weights_path',
                '/path/to/pretrained_models/lpips/vgg16-397923af.pth'),
            spatial=bool(lpips.get('spatial', False)),
        )
        self.dino_loss = FrozenDinoFeatureLoss(
            checkpoint_path=dino_loss.get('checkpoint_path', DEFAULT_DINO_WEIGHTS),
            input_size=int(dino_loss.get('input_size', 224)),
            feature_layers=tuple(dino_loss.get('feature_layers', (23,))),
            backbone_dtype=str(dino_loss.get('backbone_dtype', 'bf16')),
            loss_type=str(dino_loss.get('loss_type', 'cosine')),
        )
        self.lpips.eval()
        self.dino_loss.eval()

    def _decode_latents_to_images(self, latents):
        if self.vae is None:
            raise ValueError(
                'VAE is required to decode latents for dual-lora teacher-x0 '
                'LPIPS/DINO.')
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
        # Ask DualLoraTeacherX0 for step-1 x0_hat only.
        diffusion_kwargs['return_x0_hat'] = True
        return bs, diffusion_args, diffusion_kwargs

    def _resize_for_perceptual(self, images, size):
        if images.shape[-1] == size and images.shape[-2] == size:
            return images
        return F.interpolate(
            images, size=(size, size), mode='bilinear', align_corners=False)

    def _perceptual_pair_loss(self, pred_images, target_images):
        loss_lpips = self.lpips(pred_images, target_images)
        loss_dino = self.dino_loss(pred_images, target_images)
        return loss_lpips, loss_dino

    def _backward(self, loss, loss_scaler=None):
        if not (isinstance(loss, torch.Tensor) and loss.requires_grad):
            return
        if loss_scaler is None:
            loss.backward()
        else:
            loss_scaler.scale(loss).backward()

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

        log_vars['loss_diffusion'] = float(loss_diffusion.detach())
        loss = loss_diffusion
        loss_total_val = float(loss_diffusion.detach())

        # Step-1 residual x0 only (DualLoraTeacherX0 never puts step-2 here).
        x0_hat = extra.get('x0_hat')
        real_images = data.get('edited_images')
        w_lpips = float(self.train_cfg.get('lpips_loss_weight', 0.2))
        w_dino = float(self.train_cfg.get('dino_loss_weight', 0.1))
        perc_size = int(self.train_cfg.get('perceptual_image_size', 256))

        if x0_hat is not None and real_images is not None and (w_lpips > 0 or w_dino > 0):
            target = self._resize_for_perceptual(
                real_images.float().clamp(0.0, 1.0), perc_size)
            pred_images = self._decode_latents_to_images(x0_hat)
            pred_images = self._resize_for_perceptual(pred_images, perc_size)
            loss_lpips, loss_dino = self._perceptual_pair_loss(pred_images, target)
            log_vars['loss_lpips'] = float(loss_lpips.detach())
            log_vars['loss_dino'] = float(loss_dino.detach())
            loss_perc = w_lpips * loss_lpips + w_dino * loss_dino
            # Combined backward so step-1 PIID graph stays shared with perceptual.
            loss = loss + loss_perc
            loss_total_val = loss_total_val + float(loss_perc.detach())
            del pred_images, target, loss_lpips, loss_dino, loss_perc, x0_hat

        self._backward(loss, loss_scaler)
        log_vars['loss'] = loss_total_val
        return log_vars, bs
