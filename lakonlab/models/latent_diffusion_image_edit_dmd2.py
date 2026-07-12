# Copyright (c) 2026 EditFlow contributors
"""Top-level DMD2 trainer for pico image-editing distillation.

Faithful replication of the official DMD2 trainer (``DMD2/main/train_sd.py`` +
``sd_guidance.py``) on the EditFlow FLUX Kontext stack:

  * Two-timescale updates: fake score (+ cls head) every step, generator every
    ``dmd2_gen_update_ratio`` steps (``dfake_gen_update_ratio``).
  * One generator sample per step, shared between the generator turn (DM +
    gen cls loss) and the guidance turn (fake score + guidance cls loss),
    mirroring the official ``guidance_data_dict`` reuse.
  * DMD2-native GAN: the discriminator is the fake score network itself run in
    ``classify_mode`` plus a small trainable cls head (``FluxDMD2ClsHead``),
    classifying (optionally diffused) latents — NOT an external pixel-space
    discriminator. Losses are the official softplus objectives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mmgen.models.builder import MODELS, build_module

from .latent_diffusion_image_edit import LatentDiffusionImageEdit
from lakonlab.utils import tie_untrained_submodules


@MODELS.register_module()
class LatentDiffusionImageEditDMD2(LatentDiffusionImageEdit):
    """DMD2 alternating optimization for 2-NFE Kontext image editing.

    Optimizers (config keys):
      * ``diffusion`` — student generator (updated every ``dmd2_gen_update_ratio`` steps)
      * ``fake_score`` — trainable fake flow score (updated every step)
      * ``discriminator`` — DMD2 cls head on fake-score features (updated every step)

    The guidance side (fake score + cls head) corresponds to the official
    ``guidance_model``; both its optimizers step every iteration.
    """

    def __init__(self, *args, fake_score=None, discriminator=None, tie_fake_score=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_score = build_module(fake_score) if fake_score is not None else None
        self.discriminator = build_module(discriminator) if discriminator is not None else None
        self.tie_fake_score = tie_fake_score
        if (
                self.tie_fake_score
                and self.fake_score is not None
                and self.teacher is not None):
            # Share frozen Kontext backbone weights; keep fake_score LoRA trainable.
            tie_untrained_submodules(
                self.fake_score, self.teacher, tie_tgt_lora_base_layer=True)

    def _prepare_fake_score_kwargs(self, teacher_kwargs, bs, device):
        """Fake-score kwargs: same cond as teacher but single-batch (no CFG)."""
        fake_kwargs = {}
        for key, value in teacher_kwargs.items():
            if key == 'guidance_scale':
                continue
            if isinstance(value, torch.Tensor) and value.size(0) == 2 * bs:
                fake_kwargs[key] = value[bs:]
            else:
                fake_kwargs[key] = value
        # Distilled guidance embed for fake score training / scoring.
        fake_g = self.train_cfg.get('dmd2_fake_distilled_guidance_scale', 1.0)
        if fake_g is not None:
            fake_kwargs['guidance'] = torch.full(
                (bs,), float(fake_g), dtype=torch.float32, device=device)
        return fake_kwargs

    def _prepare_train_minibatch_args(self, data, running_status=None):
        bs, diffusion_args, diffusion_kwargs = super()._prepare_train_minibatch_args(
            data, running_status)
        if self.fake_score is None:
            raise ValueError('LatentDiffusionImageEditDMD2 requires `fake_score`.')

        device = diffusion_args[0].device
        teacher_kwargs = diffusion_kwargs.get('teacher_kwargs', dict())
        diffusion_kwargs['fake_score'] = self.fake_score
        diffusion_kwargs['fake_score_kwargs'] = self._prepare_fake_score_kwargs(
            teacher_kwargs, bs, device)
        return bs, diffusion_args, diffusion_kwargs

    def _should_update_generator(self, running_status):
        ratio = int(self.train_cfg.get('dmd2_gen_update_ratio', 5))
        ratio = max(ratio, 1)
        if running_status is None:
            return True
        return int(running_status.get('iteration', 0)) % ratio == 0

    @staticmethod
    def _freeze_trainable_params(module):
        """Temporarily freeze; returns the params to restore (official
        ``guidance_model.requires_grad_(False)`` during the generator turn)."""
        if module is None:
            return []
        frozen = [p for p in module.parameters() if p.requires_grad]
        for p in frozen:
            p.requires_grad_(False)
        return frozen

    @staticmethod
    def _restore_params(params):
        for p in params:
            p.requires_grad_(True)

    def _compute_cls_logits(self, latents, fake_score_kwargs):
        """Official ``compute_cls_logits``: (diffusion-GAN) noise the latents,
        run the fake score network in classify mode, apply the cls head."""
        kwargs = {k: v for k, v in fake_score_kwargs.items() if k != 'guidance_scale'}
        bs = latents.size(0)
        device = latents.device
        seq_len = latents.shape[2:].numel()
        ndim = latents.dim()
        latents = latents.float()

        if self.train_cfg.get('dmd2_diffusion_gan', True):
            # Official SDXL recipe: --diffusion_gan --diffusion_gan_max_timestep 1000
            max_p = float(self.train_cfg.get('dmd2_diffusion_gan_max_step_percent', 1.0))
            raw_t = torch.rand(bs, device=device, dtype=torch.float32) * max_p
            sigma = self.fake_score.timestep_sampler.warp_t(raw_t, seq_len=seq_len)
            sigma_r = sigma.reshape(bs, *((ndim - 1) * [1]))
            x_in = (1 - sigma_r) * latents + sigma_r * torch.randn_like(latents)
            t = sigma * self.fake_score.num_timesteps
        else:
            x_in = latents
            t = torch.zeros(bs, device=device, dtype=torch.float32)

        features = self.fake_score(
            return_denoising_output=True, x_t=x_in, t=t, classify_mode=True, **kwargs)
        return self.discriminator(features)

    def train_minibatch(self, data, loss_scaler=None, running_status=None):
        bs, diffusion_args, diffusion_kwargs = self._prepare_train_minibatch_args(
            data, running_status)
        update_gen = self._should_update_generator(running_status)
        # Official weights: gen_cls_loss_weight=5e-3, guidance_cls_loss_weight=1e-2.
        w_gen_cls = float(self.train_cfg.get('dmd2_gan_loss_weight', 0.0))
        w_guidance_cls = float(self.train_cfg.get('dmd2_guidance_cls_loss_weight', 1e-2))
        use_gan = self.discriminator is not None and w_gen_cls > 0
        fake_score_kwargs = diffusion_kwargs['fake_score_kwargs']
        log_vars = dict(dmd2_update_generator=float(update_gen))

        # ---- generator turn (rollout EVERY step; gradients only on update steps) ----
        gen_kwargs = dict(diffusion_kwargs)
        gen_kwargs['dmd2_turn'] = 'generator'
        gen_kwargs['compute_generator_gradient'] = update_gen
        loss_gen, gen_logs, gen_extra = self.diffusion(
            *diffusion_args, return_loss=True, **gen_kwargs)
        log_vars.update(gen_logs)
        x0_gen = gen_extra['x0_gen']
        x0_gen_detached = x0_gen.detach()

        if update_gen:
            if use_gan:
                # Freeze guidance side so gen cls grads reach the generator only
                # (official `guidance_model.requires_grad_(False)`).
                frozen = self._freeze_trainable_params(self.fake_score)
                frozen += self._freeze_trainable_params(self.discriminator)
                logits_fake = self._compute_cls_logits(x0_gen, fake_score_kwargs)
                self._restore_params(frozen)
                loss_gen_cls = F.softplus(-logits_fake).mean()
                loss_gen = loss_gen + w_gen_cls * loss_gen_cls
                log_vars['gen_cls_loss'] = float(loss_gen_cls.detach())

            if isinstance(loss_gen, torch.Tensor) and loss_gen.requires_grad:
                if loss_scaler is None:
                    loss_gen.backward()
                else:
                    loss_scaler.scale(loss_gen).backward()

        # ---- guidance turn (every step; reuses the SAME generator sample) ----
        guidance_kwargs = dict(diffusion_kwargs)
        guidance_kwargs['dmd2_turn'] = 'guidance'
        guidance_kwargs['x0_gen'] = x0_gen_detached
        loss_guidance, fake_logs, _ = self.diffusion(
            *diffusion_args, return_loss=True, **guidance_kwargs)
        log_vars.update(fake_logs)

        if use_gan:
            # Official compute_guidance_clean_cls_loss on latents:
            # real = real edited latents (x_0), fake = detached generator sample.
            real_latents = diffusion_args[0].detach()
            logits_real = self._compute_cls_logits(real_latents, fake_score_kwargs)
            logits_fake = self._compute_cls_logits(x0_gen_detached, fake_score_kwargs)
            loss_guidance_cls = (
                F.softplus(logits_fake).mean() + F.softplus(-logits_real).mean())
            loss_guidance = loss_guidance + w_guidance_cls * loss_guidance_cls
            log_vars['guidance_cls_loss'] = float(loss_guidance_cls.detach())
            log_vars['pred_realism_on_real'] = float(
                torch.sigmoid(logits_real).mean().detach())
            log_vars['pred_realism_on_fake'] = float(
                torch.sigmoid(logits_fake).mean().detach())

        if isinstance(loss_guidance, torch.Tensor) and loss_guidance.requires_grad:
            if loss_scaler is None:
                loss_guidance.backward()
            else:
                loss_scaler.scale(loss_guidance).backward()

        total = 0.0
        if isinstance(loss_gen, torch.Tensor):
            total += float(loss_gen.detach())
        if isinstance(loss_guidance, torch.Tensor):
            total += float(loss_guidance.detach())
        log_vars['loss'] = total

        return log_vars, bs
