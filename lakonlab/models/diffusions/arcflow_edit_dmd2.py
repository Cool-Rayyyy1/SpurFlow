# Copyright (c) 2026 EditFlow contributors
"""DMD2 distribution matching for data-dependent 2-NFE image editing.

Faithful adaptation of Tianwei Yin et al. DMD2 (NeurIPS 2024, official repo
``DMD2/main/sd_guidance.py`` + ``sd_unified_model.py``) to EditFlow / FLUX
Kontext:

  * Generator: ArcFlowEdit 2-NFE residual student.
  * Real score: frozen teacher GaussianFlow (Kontext).
  * Fake score: trainable flow model scoring generator samples.
  * Backward simulation (official semantics): pick a random step index on the
    student's own inference trajectory, bootstrap there *without* gradients,
    then run exactly ONE gradient step that predicts the clean latent.
  * Generator loss: distribution matching (DM) on that x0 prediction.
  * Fake-score loss: rectified-flow velocity MSE on the SAME (detached)
    generator sample, full timestep range (official ``compute_loss_fake``).

The DMD2-native GAN (cls head on fake-score features) lives in the top-level
model wrapper (``LatentDiffusionImageEditDMD2``).
"""

from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mmgen.models.builder import MODULES

from .arcflow_edit import ArcFlowEditImitationSplitStage
from lakonlab.utils import module_eval


@MODULES.register_module()
class ArcFlowEditDMD2Imitation(ArcFlowEditImitationSplitStage):
    """2-NFE edit generator with DMD2 distribution-matching training.

    ``forward_train`` handles two turns, controlled by ``dmd2_turn``:

    * ``'generator'``: backward-simulated one-step rollout to a clean latent.
      Always returns the generated sample in ``extra['x0_gen']`` (mirrors the
      official ``guidance_data_dict``); computes the DM loss only when
      ``compute_generator_gradient=True`` (i.e. on generator-update steps,
      matching ``dfake_gen_update_ratio``).
    * ``'guidance'``: trains the fake score on the sample produced by the
      generator turn (passed back in via ``x0_gen``), no second rollout.
    """

    def _score_kwargs_for_batch(self, score_kwargs, num_batches):
        """Slice CFG-doubled teacher kwargs down to a single batch if needed."""
        if not score_kwargs:
            return dict()
        out = {}
        for key, value in score_kwargs.items():
            if isinstance(value, torch.Tensor) and value.size(0) == 2 * num_batches:
                out[key] = value[num_batches:]
            else:
                out[key] = value
        return out

    def _predict_u(self, score_model, x_t, t, score_kwargs, guidance_scale):
        kwargs = dict(score_kwargs)
        # Flux distilled guidance embed (Kontext); keep if present.
        return score_model(
            x_t=x_t,
            t=t,
            return_u=True,
            guidance_scale=guidance_scale,
            **kwargs)

    def _x0_from_velocity(self, x_t, u, sigma):
        """Rectified-flow inversion: x_t = (1-sigma) x0 + sigma eps, u = eps - x0."""
        return x_t - sigma * u

    def _sample_dm_timesteps(self, num_batches, device, seq_len):
        """DM-loss timesteps, restricted to [min, max] step percent (official)."""
        max_step_percent = float(self.train_cfg.get('dmd2_max_step_percent', 0.98))
        min_step_percent = float(self.train_cfg.get('dmd2_min_step_percent', 0.02))
        raw_t = torch.rand(num_batches, device=device, dtype=torch.float32)
        raw_t = raw_t * (max_step_percent - min_step_percent) + min_step_percent
        sigma = self.timestep_sampler.warp_t(raw_t, seq_len=seq_len)
        t = sigma * self.num_timesteps
        return raw_t, sigma, t

    def _sample_fake_timesteps(self, num_batches, device, seq_len):
        """Fake-score training timesteps, FULL range (official compute_loss_fake)."""
        raw_t = torch.rand(num_batches, device=device, dtype=torch.float32)
        sigma = self.timestep_sampler.warp_t(raw_t, seq_len=seq_len)
        t = sigma * self.num_timesteps
        return raw_t, sigma, t

    @torch.no_grad()
    def _compute_dm_grad(self, x0_gen, teacher, fake_score, teacher_kwargs, fake_kwargs):
        """DMD2 distribution-matching gradient on generator samples (no grad).

        For FLUX Kontext we typically keep ``dmd2_real_guidance_scale=1.0`` and rely
        on the distilled ``guidance`` embed inside ``teacher_kwargs`` (same as PIID).
        If CFG scale > 1 is set, ``GaussianFlow.forward_u`` doubles ``x_t`` internally;
        ``teacher_kwargs`` must already contain neg+pos embeds (batch 2B).
        """
        num_batches = x0_gen.size(0)
        device = x0_gen.device
        seq_len = x0_gen.shape[2:].numel()
        ndim = x0_gen.dim()

        _, sigma_flat, t = self._sample_dm_timesteps(num_batches, device, seq_len)
        sigma = sigma_flat.reshape(num_batches, *((ndim - 1) * [1]))
        noise = torch.randn_like(x0_gen)
        x_t = (1 - sigma) * x0_gen + sigma * noise

        real_gs = float(self.train_cfg.get('dmd2_real_guidance_scale', 1.0))
        fake_gs = float(self.train_cfg.get('dmd2_fake_guidance_scale', 1.0))

        t_kw = teacher_kwargs if real_gs > 1.0 else self._score_kwargs_for_batch(
            teacher_kwargs, num_batches)
        u_real = self._predict_u(teacher, x_t, t, t_kw, real_gs)
        x0_real = self._x0_from_velocity(x_t, u_real, sigma)

        f_kw = self._score_kwargs_for_batch(fake_kwargs, num_batches)
        u_fake = self._predict_u(fake_score, x_t, t, f_kw, fake_gs)
        x0_fake = self._x0_from_velocity(x_t, u_fake, sigma)

        p_real = x0_gen - x0_real
        p_fake = x0_gen - x0_fake
        reduce_dims = tuple(range(1, p_real.dim()))
        grad = (p_real - p_fake) / torch.abs(p_real).mean(
            dim=reduce_dims, keepdim=True).clamp(min=1e-6)
        grad = torch.nan_to_num(grad)
        return grad, dict(
            dmtrain_grad_norm=float(torch.norm(grad).item()),
            dmtrain_sigma_mean=float(sigma_flat.mean().item()),
        )

    def _dm_loss(self, x0_gen, grad):
        weight = float(self.train_cfg.get('dmd2_dm_loss_weight', 1.0))
        target = (x0_gen - grad).detach()
        loss = 0.5 * F.mse_loss(x0_gen.float(), target.float())
        return weight * loss

    def _fake_score_loss(self, x0_gen, fake_score, fake_kwargs):
        """Train fake score with rectified-flow velocity matching on detached fakes."""
        x0 = x0_gen.detach()
        num_batches = x0.size(0)
        device = x0.device
        seq_len = x0.shape[2:].numel()
        ndim = x0.dim()

        _, sigma_flat, t = self._sample_fake_timesteps(num_batches, device, seq_len)
        sigma = sigma_flat.reshape(num_batches, *((ndim - 1) * [1]))
        noise = torch.randn_like(x0)
        x_t = (1 - sigma) * x0 + sigma * noise
        u_target = noise - x0

        f_kw = self._score_kwargs_for_batch(fake_kwargs, num_batches)
        fake_gs = float(self.train_cfg.get('dmd2_fake_train_guidance_scale', 1.0))
        u_pred = self._predict_u(fake_score, x_t, t, f_kw, fake_gs)
        loss = F.mse_loss(u_pred.float(), u_target.float())
        weight = float(self.train_cfg.get('dmd2_fake_loss_weight', 1.0))
        return weight * loss

    def _sample_dmd2_step_index(self, device):
        """Random inference-step index shared across ranks (official broadcast)."""
        nfe = int(self.train_cfg.get('nfe', 2))
        selected = torch.randint(low=0, high=nfe, size=(1,), device=device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(selected, src=0)
        return int(selected.item())

    def _backward_sim_rollout(
            self, x_0, path_epsilon, x_ref, kwargs, selected_step, enable_grad):
        """Official DMD2 backward simulation adapted to the ArcFlow 2-NFE sampler.

        1. Bootstrap (NO grad) to the state the student's own sampler visits at
           ``selected_step``. Because the ArcFlow sampler is a deterministic
           fixed-eps ODE, the bootstrap is the student's own step-1 transition
           (rather than the official predict-x0-then-renoise, which has no
           analogue on a deterministic path).
        2. Run exactly ONE gradient step from that state: predict the policy
           once and integrate straight to t=0, yielding the clean-latent
           prediction ``x0_gen``. Gradients never cross more than one student
           forward, matching the official trainer.

        With ``dmd2_backward_simulation=False``, the bootstrap state instead
        comes from noising the REAL edited latent on the fixed-eps path
        (official non-backsim "denoising on real data" mode).
        """
        device = path_epsilon.device
        num_batches = path_epsilon.size(0)
        seq_len = path_epsilon.shape[2:].numel()
        ndim = path_epsilon.dim()
        policy_eps = self.train_cfg.get('eps', 1e-4)
        base_segment_size, _ = self._rollout_segment_sizes()
        x_ref_scale_step2 = self.train_cfg.get('split_stage_step2_x_ref_scale', 1.0)

        raw_t = torch.ones(num_batches, dtype=torch.float32, device=device)
        sigma_t = self.timestep_sampler.warp_t(raw_t, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t = sigma_t.flatten() * self.num_timesteps
        x_t = path_epsilon
        x_ref_step = x_ref

        if selected_step > 0:
            raw_t_mid = raw_t - base_segment_size
            if self.train_cfg.get('dmd2_backward_simulation', True):
                with torch.no_grad():
                    denoising_output = self.pred(x_t, t, **kwargs)
                    policy = self.policy_class(
                        denoising_output, x_t, sigma_t, x_ref=x_ref,
                        path_epsilon=path_epsilon, eps=policy_eps)
                    x_t, sigma_t, t = self.momentum_integration(
                        sigma_t, x_t, sigma_t, raw_t_mid, policy,
                        eps=policy_eps, seq_len=seq_len)
                x_t = x_t.detach()
            else:
                sigma_t = self.timestep_sampler.warp_t(
                    raw_t_mid, seq_len=seq_len).reshape(
                        num_batches, *((ndim - 1) * [1]))
                x_t = (1 - sigma_t) * x_0 + sigma_t * path_epsilon
                t = sigma_t.flatten() * self.num_timesteps
            raw_t = raw_t_mid
            x_ref_step = x_ref * x_ref_scale_step2

        raw_t_end = torch.zeros(num_batches, dtype=torch.float32, device=device)
        grad_ctx = contextlib.nullcontext() if enable_grad else torch.no_grad()
        with grad_ctx:
            denoising_output = self.pred(x_t, t, **kwargs)
            policy = self.policy_class(
                denoising_output, x_t, sigma_t, x_ref=x_ref_step,
                path_epsilon=path_epsilon, eps=policy_eps)
            x0_gen, _, _ = self.momentum_integration(
                sigma_t, x_t, sigma_t, raw_t_end, policy,
                eps=policy_eps, seq_len=seq_len)
        return x0_gen

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=None,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        return super().forward_test(
            x_0=x_0, noise=noise, guidance_scale=guidance_scale,
            test_cfg_override=test_cfg_override, show_pbar=show_pbar, **kwargs)

    def forward_train(
            self,
            x_0,
            teacher=None,
            teacher_kwargs=dict(),
            fake_score=None,
            fake_score_kwargs=dict(),
            running_status=None,
            dmd2_turn='generator',
            compute_generator_gradient=True,
            x0_gen=None,
            **kwargs):
        x_ref = kwargs.pop('x_ref', None)
        if x_ref is None:
            raise ValueError(
                'ArcFlowEditDMD2Imitation requires `x_ref` (source/reference latents).')
        if fake_score is None:
            raise ValueError('ArcFlowEditDMD2Imitation requires `fake_score`.')

        log_vars = dict()
        extra = dict()

        if dmd2_turn == 'generator':
            if teacher is None:
                raise ValueError('ArcFlowEditDMD2Imitation requires `teacher` (real score).')

            path_epsilon = torch.randn_like(x_0)
            selected_step = self._sample_dmd2_step_index(x_0.device)
            log_vars['dmd2_selected_step'] = float(selected_step)

            x0_gen = self._backward_sim_rollout(
                x_0, path_epsilon, x_ref, kwargs, selected_step,
                enable_grad=compute_generator_gradient)
            extra['x0_gen'] = x0_gen

            if not compute_generator_gradient:
                # Off-cycle step: sample only feeds the guidance turn (official
                # `compute_generator_gradient=False` path).
                return None, log_vars, extra

            with module_eval(teacher), module_eval(fake_score):
                grad, dm_logs = self._compute_dm_grad(
                    x0_gen, teacher, fake_score, teacher_kwargs, fake_score_kwargs)
            log_vars.update(dm_logs)
            loss_dm = self._dm_loss(x0_gen, grad)
            log_vars['loss_dm'] = float(loss_dm.detach())
            loss = loss_dm

            # Optional weak regression to real edited latent (data-dependent anchor).
            reg_w = float(self.train_cfg.get('dmd2_reg_loss_weight', 0.0))
            if reg_w > 0:
                loss_reg = reg_w * F.mse_loss(x0_gen.float(), x_0.float())
                loss = loss + loss_reg
                log_vars['loss_reg'] = float(loss_reg.detach())

            log_vars['loss'] = float(loss.detach())
            return loss, log_vars, extra

        if dmd2_turn == 'guidance':
            if x0_gen is None:
                raise ValueError(
                    "dmd2_turn='guidance' requires `x0_gen` (the sample produced "
                    'by the generator turn); no second rollout is performed.')
            loss_fake = self._fake_score_loss(x0_gen, fake_score, fake_score_kwargs)
            log_vars['loss_fake_mean'] = float(loss_fake.detach())
            log_vars['loss'] = float(loss_fake.detach())
            return loss_fake, log_vars, extra

        raise ValueError(f'Unknown dmd2_turn={dmd2_turn!r}; expected generator|guidance.')
