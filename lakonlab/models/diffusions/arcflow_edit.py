# Copyright (c) 2026 EditFlow contributors

import torch

from copy import deepcopy
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES

from .arcflow import ArcFlowImitation, ArcFlowImitationBase
from .policies import ArcFlowEditPolicy, ArcFlowEditNewPolicy
from lakonlab.utils import module_eval


@MODULES.register_module()
class ArcFlowEditImitation(ArcFlowImitation):
    """Data-dependent ArcFlow distillation with edit residual parameterization.

    pred_delta = mixture(deltax) ~ x0_tgt - x_ref
    student_u = path_epsilon - x_ref - pred_delta
    teacher_u = path_epsilon - x0_tgt
    """

    def __init__(self, *args, policy_type='ArcFlowEdit', policy_kwargs=None, **kwargs):
        super().__init__(*args, policy_type=policy_type, policy_kwargs=policy_kwargs, **kwargs)

    def momentum_integration(
            self,
            sigma_t_src: torch.Tensor,
            x_t_start: torch.Tensor,
            sigma_t_start: torch.Tensor,
            raw_t_end: torch.Tensor,
            policy,
            eps=1e-4,
            seq_len=None):
        """Integrate edit student flow over [sigma_start, sigma_end].

        GM mixture part (base layer, no alpha):
            integrated_delta = integral of pred_delta from deltax/logweights/loggammas
        Edit offsets (alpha scales x_ref only, not the mixture):
            x_end = x_start - path_epsilon*dt + alpha*x_ref*dt + integrated_delta
        """
        num_batches = x_t_start.size(0)
        ndim = x_t_start.dim()

        sigma_t_end = self.timestep_sampler.warp_t(raw_t_end, seq_len=seq_len)
        sigma_t_end = sigma_t_end.reshape(num_batches, *((ndim - 1) * [1]))
        dt_step = (sigma_t_start - sigma_t_end).clamp(min=eps)

        if isinstance(policy, (ArcFlowEditPolicy, ArcFlowEditNewPolicy)):
            if policy.path_epsilon is None:
                raise ValueError(
                    'Edit policy requires `path_epsilon` for residual velocity integration.')
            x_t_end_sub, _, t_end = ArcFlowImitationBase.momentum_integration(
                self, sigma_t_src, x_t_start, sigma_t_start, raw_t_end,
                policy, eps=eps, seq_len=seq_len)
            integrated_delta = x_t_start - x_t_end_sub
            x_ref_term = policy.x_ref * dt_step
            if hasattr(policy, 'alpha'):
                x_ref_term = policy.alpha * x_ref_term
            x_t_end = (
                x_t_start
                - policy.path_epsilon * dt_step
                + x_ref_term
                + integrated_delta)
            return x_t_end, sigma_t_end, t_end

        return ArcFlowImitationBase.momentum_integration(
            self, sigma_t_src, x_t_start, sigma_t_start, raw_t_end,
            policy, eps=eps, seq_len=seq_len)

    def _direct_delta_loss(self, x_0, policy, t_src, sigma_t_src):
        """Anchor GM mixture pred_delta to the edit endpoint decomposition.

        Non-alpha: pred_delta ~ x0_tgt - x_ref  (endpoint: x_ref + pred_delta).
        Alpha: pred_delta ~ x0_tgt - alpha*x_ref  (endpoint: alpha*x_ref + pred_delta).
        Uses the same DiffusionMSELoss rescaling as PIID.
        """
        pred_delta = policy.compute_pred_delta(sigma_t_src, sigma_t_src)
        if hasattr(policy, 'alpha'):
            target_delta = x_0 - policy.alpha * policy.x_ref
        else:
            target_delta = x_0 - policy.x_ref
        return self.flow_loss(dict(
            u_t_pred=pred_delta,
            u_t=target_delta,
            timesteps=t_src,
        ))

    def _maybe_add_direct_delta_loss(
            self,
            loss,
            log_vars,
            x_0,
            policy,
            t_src,
            sigma_t_src):
        w = self.train_cfg.get('direct_delta_loss_weight', 0.0)
        if w <= 0:
            return loss, log_vars
        if hasattr(policy, 'alpha'):
            loss_direct = self._direct_delta_loss(x_0, policy, t_src, sigma_t_src)
            loss = loss + w * loss_direct
            log_vars['loss_direct_delta'] = float(loss_direct.detach())
        return loss, log_vars

    def forward_train(self, x_0, teacher=None, teacher_kwargs=dict(), running_status=None, **kwargs):
        x_ref = kwargs.pop('x_ref', None)
        if x_ref is None:
            raise ValueError('ArcFlowEditImitation requires `x_ref` (source/reference latents) in kwargs.')

        device = get_module_device(self)
        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()
        ndim = x_0.dim()
        assert ndim in [4, 5], f'Invalid x_0 shape: {x_0.shape}. Expected 4D or 5D tensor.'

        num_decay_iters = self.train_cfg.get('num_decay_iters', 0)
        if num_decay_iters > 0:
            teacher_ratio = 1 - min(running_status['iteration'], num_decay_iters) / num_decay_iters
            log_vars = dict(teacher_ratio=teacher_ratio)
        else:
            teacher_ratio = 0.0
            log_vars = dict()

        raw_t_src, sigma_t_src, t_src, segment_size = self.sample_t(
            num_batches, ndim, seq_len=seq_len, device=device)

        policy_kwargs = dict(eps=self.train_cfg.get('eps', 1e-4))
        if self.train_cfg.get('teacher_invert_epsilon', False):
            if teacher is None:
                raise ValueError(
                    'ArcFlowEditImitation with teacher_invert_epsilon requires `teacher`.')
            invert_steps = self.train_cfg.get('teacher_invert_steps', 10)
            with torch.no_grad(), module_eval(teacher):
                path_epsilon = self.teacher_invert_x0_to_noise(
                    teacher, x_0, teacher_kwargs,
                    num_steps=invert_steps, seq_len=seq_len)
            # Decouple the noise that builds the network input x_t from the
            # path_epsilon used in the residual velocity. With 'random', x_t is
            # mixed with fresh random noise (so the teacher-inverted endpoint does
            # not leak the target into the input), while the policy still uses the
            # teacher-inverted path_epsilon for student_u = path_epsilon - x_ref - delta.
            if self.train_cfg.get('xt_input_noise', 'epsilon') == 'random':
                input_noise = torch.randn_like(x_0)
            else:
                input_noise = path_epsilon
            x_t_src, _, _ = self.sample_forward_diffusion(x_0, t_src, input_noise)
            policy_kwargs['path_epsilon'] = path_epsilon
        else:
            path_epsilon = torch.randn_like(x_0)
            x_t_src, _, _ = self.sample_forward_diffusion(x_0, t_src, path_epsilon)
            policy_kwargs['path_epsilon'] = path_epsilon

        denoising_output = self.pred(x_t_src, t_src, **kwargs)
        policy = self.policy_class(
            denoising_output, x_t_src, sigma_t_src, x_ref=x_ref, **policy_kwargs)

        loss_diffusion, _, raw_t_dst = self.piid_segment_momentum(
            teacher, policy, x_t_src, raw_t_src, sigma_t_src, teacher_ratio, segment_size,
            teacher_kwargs)

        loss = loss_diffusion
        loss, log_vars = self._maybe_add_direct_delta_loss(
            loss, log_vars, x_0, policy, t_src, sigma_t_src)
        log_vars.update(self.flow_loss.log_vars)
        log_vars.update(loss_diffusion=float(loss_diffusion.detach()))

        return loss, log_vars

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=None,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        x_ref = kwargs.get('image_latents')
        if x_ref is None:
            raise ValueError('ArcFlowEditImitation inference requires `image_latents` (source latents).')

        import sys
        import mmcv

        x_t_src = torch.randn_like(x_0) if noise is None else noise
        path_epsilon = x_t_src.clone()
        num_batches = x_t_src.size(0)
        seq_len = x_t_src.shape[2:].numel()
        ori_dtype = x_t_src.dtype
        device = x_t_src.device
        x_t_src = x_t_src.float()
        path_epsilon = path_epsilon.float()
        ndim = x_t_src.dim()
        assert ndim in [4, 5], f'Invalid x_t_src shape: {x_t_src.shape}. Expected 4D or 5D tensor.'

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        eps = cfg.get('eps', 1e-4)
        nfe = cfg['nfe']
        timestep_ratio = max(cfg.get('timestep_ratio', 1.0), eps)
        base_segment_size = 1 / (nfe - 1 + timestep_ratio)
        # If True: final NFE predicts pred_delta and returns x_ref (+alpha) +
        # pred_delta as x0, skipping the last velocity integration step.
        step2_direct_x0 = bool(cfg.get('step2_direct_x0', False))

        raw_t_src = torch.ones((num_batches,), dtype=torch.float32, device=device)
        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps

        if show_pbar:
            pbar = mmcv.ProgressBar(nfe)

        for step_id in range(nfe):
            is_final_step = step_id == nfe - 1
            if is_final_step:
                segment_size = base_segment_size * timestep_ratio
            else:
                segment_size = base_segment_size

            raw_t_dst = raw_t_src - segment_size

            denoising_output = self.pred(x_t_src, t_src, **kwargs)
            policy = self.policy_class(
                denoising_output, x_t_src, sigma_t_src, x_ref=x_ref,
                path_epsilon=path_epsilon, eps=eps)
            if not is_final_step:
                temperature = cfg.get('temperature', 1.0)
                policy.temperature_(temperature)

            if is_final_step and step2_direct_x0:
                # Step-2 shortcut: x0 = alpha*x_ref + pred_delta (or x_ref + pred_delta).
                pred_delta = policy.compute_pred_delta(sigma_t_src, sigma_t_src)
                if hasattr(policy, 'alpha'):
                    x_t_src = policy.alpha * policy.x_ref + pred_delta
                else:
                    x_t_src = policy.x_ref + pred_delta
                if show_pbar:
                    pbar.update()
                break

            x_t_dst, sigma_t_dst, t_dst = self.momentum_integration(
                sigma_t_src, x_t_src, sigma_t_src, raw_t_dst,
                policy, eps=eps, seq_len=seq_len)

            x_t_src = x_t_dst
            raw_t_src = raw_t_dst
            sigma_t_src = sigma_t_dst
            t_src = t_dst

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t_src.to(ori_dtype)


@MODULES.register_module()
class ArcFlowEditImitationSplitStage(ArcFlowEditImitation):
    """Data-dependent fixed-eps edit distillation with a 2-step student rollout.

    Each minibatch samples ``path_epsilon`` and uses the real edited latent
    ``x0_tgt`` (data-dependent). The student performs a full ``nfe=2`` rollout
    from the sampled noise:

      * step-1 (t=1 -> first knot): PIID teacher-alignment loss only.
      * step-2 (student step-1 endpoint -> t=0): PIID teacher loss plus a
        direct flow-matching loss against ``path_epsilon - x0_tgt``. Step-2
        uses ``split_stage_step2_x_ref_scale * x_ref`` (default 1.0) inside
        the residual velocity / direct-flow target.

    Step-2 is chained from the differentiable student step-1 endpoint so
    step-2 gradients backpropagate into step-1.

    Optional ``split_stage_step2_warmup_iters``: linearly ramps the combined
    step-2 loss from 0 to full weight so step-1 PIID can stabilize first.
    """

    def _step2_warmup_scale(self, running_status):
        warmup_iters = self.train_cfg.get('split_stage_step2_warmup_iters', 0)
        if warmup_iters <= 0:
            return 1.0
        if running_status is None:
            return 1.0
        iteration = running_status.get('iteration', 0)
        return min(1.0, float(iteration) / float(warmup_iters))

    def _rollout_segment_sizes(self):
        eps = self.train_cfg.get('eps', 1e-4)
        nfe = self.train_cfg['nfe']
        if nfe != 2:
            raise ValueError(
                f'ArcFlowEditImitationSplitStage expects nfe=2, got nfe={nfe}.')
        timestep_ratio = max(self.train_cfg.get('timestep_ratio', 1.0), eps)
        one_minus_final_scale = 1 - timestep_ratio
        base_segment_size = 1 / (nfe - one_minus_final_scale)
        final_step_size = timestep_ratio * base_segment_size
        return base_segment_size, final_step_size

    def _direct_flow_loss(self, x_0, policy, t_src, sigma_t_src):
        """Analytic data flow-matching loss for the edit residual at current t.

        Non-alpha only: target_u = path_epsilon - x0_tgt,
        pred_u = path_epsilon - x_ref - pred_delta.
        For alpha policies use ``_direct_delta_loss`` instead.
        """
        target_u = policy.path_epsilon - x_0
        pred_u = policy.velocity(sigma_t_src, sigma_t_src)
        return self.flow_loss(dict(
            u_t_pred=pred_u,
            u_t=target_u,
            timesteps=t_src,
        ))

    def _split_stage_step2_x_ref_scale(self, cfg):
        return cfg.get(
            'split_stage_step2_x_ref_scale',
            self.train_cfg.get('split_stage_step2_x_ref_scale', 1.0))

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=None,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        """2-NFE inference aligned with split-stage training.

        Step-1 policy uses full ``x_ref``; step-2 (final segment) uses
        ``split_stage_step2_x_ref_scale * x_ref`` in the velocity / integration
        formula, matching ``forward_train``. Transformer ``image_latents`` stay
        at full reference strength on both steps.
        """
        x_ref = kwargs.get('image_latents')
        if x_ref is None:
            raise ValueError(
                'ArcFlowEditImitationSplitStage inference requires '
                '`image_latents` (source latents).')

        import sys
        import mmcv

        x_t_src = torch.randn_like(x_0) if noise is None else noise
        path_epsilon = x_t_src.clone()
        num_batches = x_t_src.size(0)
        seq_len = x_t_src.shape[2:].numel()
        ori_dtype = x_t_src.dtype
        device = x_t_src.device
        x_t_src = x_t_src.float()
        path_epsilon = path_epsilon.float()
        ndim = x_t_src.dim()
        assert ndim in [4, 5], f'Invalid x_t_src shape: {x_t_src.shape}. Expected 4D or 5D tensor.'

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        eps = cfg.get('eps', 1e-4)
        nfe = cfg['nfe']
        if nfe != 2:
            raise ValueError(
                f'ArcFlowEditImitationSplitStage expects nfe=2 at inference, got nfe={nfe}.')
        timestep_ratio = max(cfg.get('timestep_ratio', 1.0), eps)
        base_segment_size, _ = self._rollout_segment_sizes()
        x_ref_scale_step2 = self._split_stage_step2_x_ref_scale(cfg)

        raw_t_src = torch.ones((num_batches,), dtype=torch.float32, device=device)
        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps

        if show_pbar:
            pbar = mmcv.ProgressBar(nfe)

        for step_id in range(nfe):
            is_final_step = step_id == nfe - 1
            segment_size = base_segment_size if not is_final_step else (
                timestep_ratio * base_segment_size)

            raw_t_dst = raw_t_src - segment_size

            denoising_output = self.pred(x_t_src, t_src, **kwargs)
            x_ref_policy = x_ref if step_id == 0 else x_ref * x_ref_scale_step2
            policy = self.policy_class(
                denoising_output, x_t_src, sigma_t_src, x_ref=x_ref_policy,
                path_epsilon=path_epsilon, eps=eps)
            if not is_final_step:
                temperature = cfg.get('temperature', 1.0)
                policy.temperature_(temperature)

            x_t_dst, sigma_t_dst, t_dst = self.momentum_integration(
                sigma_t_src, x_t_src, sigma_t_src, raw_t_dst,
                policy, eps=eps, seq_len=seq_len)

            x_t_src = x_t_dst
            raw_t_src = raw_t_dst
            sigma_t_src = sigma_t_dst
            t_src = t_dst

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t_src.to(ori_dtype)

    def forward_train(self, x_0, teacher=None, teacher_kwargs=dict(), running_status=None, **kwargs):
        x_ref = kwargs.pop('x_ref', None)
        if x_ref is None:
            raise ValueError(
                'ArcFlowEditImitationSplitStage requires `x_ref` (source/reference latents) in kwargs.')

        device = get_module_device(self)
        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()
        ndim = x_0.dim()
        assert ndim in [4, 5], f'Invalid x_0 shape: {x_0.shape}. Expected 4D or 5D tensor.'

        num_decay_iters = self.train_cfg.get('num_decay_iters', 0)
        if num_decay_iters > 0:
            teacher_ratio = 1 - min(running_status['iteration'], num_decay_iters) / num_decay_iters
            log_vars = dict(teacher_ratio=teacher_ratio)
        else:
            teacher_ratio = 0.0
            log_vars = dict()

        base_segment_size, final_step_size = self._rollout_segment_sizes()
        policy_eps = self.train_cfg.get('eps', 1e-4)
        x_ref_scale_step2 = self.train_cfg.get('split_stage_step2_x_ref_scale', 0.5)
        w_diff = self.train_cfg.get('split_stage_diffusion_loss_weight', 0.5)
        w_teacher = self.train_cfg.get('split_stage_teacher_loss_weight', 0.5)
        step2_warmup_scale = self._step2_warmup_scale(running_status)
        log_vars['step2_warmup_scale'] = step2_warmup_scale

        path_epsilon = torch.randn_like(x_0)

        # ----- step-1: rollout from sampled noise at t=1 -----
        raw_t_step1 = torch.ones(num_batches, dtype=torch.float32, device=device)
        sigma_t_step1 = self.timestep_sampler.warp_t(raw_t_step1, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_step1 = sigma_t_step1.flatten() * self.num_timesteps
        x_t_step1 = path_epsilon

        denoising_output_step1 = self.pred(x_t_step1, t_step1, **kwargs)
        policy_step1 = self.policy_class(
            denoising_output_step1, x_t_step1, sigma_t_step1, x_ref=x_ref,
            path_epsilon=path_epsilon, eps=policy_eps)

        loss_step1, _, _ = self.piid_segment_momentum(
            teacher, policy_step1, x_t_step1, raw_t_step1, sigma_t_step1,
            teacher_ratio, base_segment_size, teacher_kwargs)

        loss = loss_step1
        loss_step2_teacher = None
        loss_step2_diff = None

        if step2_warmup_scale > 0:
            raw_t_step2 = raw_t_step1 - base_segment_size
            x_t_step2, sigma_t_step2, t_step2 = self.momentum_integration(
                sigma_t_step1, x_t_step1, sigma_t_step1, raw_t_step2,
                policy_step1, eps=policy_eps, seq_len=seq_len)

            # ----- step-2: start from student step-1 endpoint (grad flows back) -----
            x_ref_step2 = x_ref * x_ref_scale_step2
            denoising_output_step2 = self.pred(x_t_step2, t_step2, **kwargs)
            policy_step2 = self.policy_class(
                denoising_output_step2, x_t_step2, sigma_t_step2, x_ref=x_ref_step2,
                path_epsilon=path_epsilon, eps=policy_eps)

            loss_step2_teacher, _, _ = self.piid_segment_momentum(
                teacher, policy_step2, x_t_step2, raw_t_step2, sigma_t_step2,
                teacher_ratio, final_step_size, teacher_kwargs)
            loss_step2_diff = self._direct_flow_loss(
                x_0, policy_step2, t_step2, sigma_t_step2)

            loss = loss + step2_warmup_scale * (
                w_teacher * loss_step2_teacher + w_diff * loss_step2_diff)

        log_vars.update(self.flow_loss.log_vars)
        log_vars.update(
            loss=float(loss.detach()),
            loss_step1=float(loss_step1.detach()),
        )
        if loss_step2_teacher is not None:
            log_vars.update(
                loss_step2_teacher=float(loss_step2_teacher.detach()),
                loss_step2_direct=float(loss_step2_diff.detach()),
            )
        return loss, log_vars


@MODULES.register_module()
class ArcFlowEditImitationSplitStageGAN(ArcFlowEditImitationSplitStage):
    """Split-stage edit distillation with step-2 PIID + direct flow + GAN.

    Step-1 / step-2 PIID are active from the first iteration (no split-stage PIID
    warmup; pretrain ckpt already provides a good step-1). Step-2 also keeps the
    analytic direct flow MSE (``pred_delta ≈ x0_tgt - x_ref``) so GAN fine-tuning
    cannot freely collapse the edit residual. External GAN loss on the
    Kontext-decoded t=0 rollout endpoint ramps in after
    ``split_stage_gan_warmup_iters`` over ``split_stage_gan_ramp_iters``.
    """

    def _gan_loss_scale(self, running_status):
        from lakonlab.models.architecture.dinov3_discriminator import split_stage_gan_loss_scale
        return split_stage_gan_loss_scale(running_status, self.train_cfg)

    def forward_train(
            self,
            x_0,
            teacher=None,
            teacher_kwargs=dict(),
            running_status=None,
            return_step2_latent=False,
            **kwargs):
        x_ref = kwargs.pop('x_ref', None)
        if x_ref is None:
            raise ValueError(
                'ArcFlowEditImitationSplitStageGAN requires `x_ref` in kwargs.')

        device = get_module_device(self)
        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()
        ndim = x_0.dim()
        assert ndim in [4, 5], f'Invalid x_0 shape: {x_0.shape}. Expected 4D or 5D tensor.'

        num_decay_iters = self.train_cfg.get('num_decay_iters', 0)
        if num_decay_iters > 0:
            teacher_ratio = 1 - min(running_status['iteration'], num_decay_iters) / num_decay_iters
            log_vars = dict(teacher_ratio=teacher_ratio)
        else:
            teacher_ratio = 0.0
            log_vars = dict()

        base_segment_size, final_step_size = self._rollout_segment_sizes()
        policy_eps = self.train_cfg.get('eps', 1e-4)
        x_ref_scale_step2 = self.train_cfg.get('split_stage_step2_x_ref_scale', 1.0)
        w_diff = self.train_cfg.get('split_stage_diffusion_loss_weight', 0.5)
        w_teacher = self.train_cfg.get('split_stage_teacher_loss_weight', 0.5)
        gan_loss_scale = self._gan_loss_scale(running_status)
        log_vars['gan_loss_scale'] = gan_loss_scale

        path_epsilon = torch.randn_like(x_0)

        raw_t_step1 = torch.ones(num_batches, dtype=torch.float32, device=device)
        sigma_t_step1 = self.timestep_sampler.warp_t(raw_t_step1, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_step1 = sigma_t_step1.flatten() * self.num_timesteps
        x_t_step1 = path_epsilon

        denoising_output_step1 = self.pred(x_t_step1, t_step1, **kwargs)
        policy_step1 = self.policy_class(
            denoising_output_step1, x_t_step1, sigma_t_step1, x_ref=x_ref,
            path_epsilon=path_epsilon, eps=policy_eps)

        loss_step1, _, _ = self.piid_segment_momentum(
            teacher, policy_step1, x_t_step1, raw_t_step1, sigma_t_step1,
            teacher_ratio, base_segment_size, teacher_kwargs)

        loss = loss_step1

        raw_t_step2 = raw_t_step1 - base_segment_size
        x_t_step2, sigma_t_step2, t_step2 = self.momentum_integration(
            sigma_t_step1, x_t_step1, sigma_t_step1, raw_t_step2,
            policy_step1, eps=policy_eps, seq_len=seq_len)

        x_ref_step2 = x_ref * x_ref_scale_step2
        denoising_output_step2 = self.pred(x_t_step2, t_step2, **kwargs)
        policy_step2 = self.policy_class(
            denoising_output_step2, x_t_step2, sigma_t_step2, x_ref=x_ref_step2,
            path_epsilon=path_epsilon, eps=policy_eps)

        loss_step2_teacher, _, _ = self.piid_segment_momentum(
            teacher, policy_step2, x_t_step2, raw_t_step2, sigma_t_step2,
            teacher_ratio, final_step_size, teacher_kwargs)
        loss_step2_diff = self._direct_flow_loss(
            x_0, policy_step2, t_step2, sigma_t_step2)

        loss = loss + w_teacher * loss_step2_teacher + w_diff * loss_step2_diff

        step2_latent = None
        if gan_loss_scale > 0:
            raw_t_final = raw_t_step2 - final_step_size
            # Reuse PIID step-2 policy; integrate to t=0 for GAN decode (no second pred).
            # With gan_grad_step2_only, detach the integration-state input. Note this
            # only cuts the additive x_t path: policy_step2 itself was predicted from
            # the non-detached x_t_step2, so GAN grads still reach step-1 through the
            # network input (same chaining path as the step-2 PIID/direct losses).
            x_t_gan_in = (
                x_t_step2.detach()
                if self.train_cfg.get('gan_grad_step2_only', True)
                else x_t_step2)
            step2_latent, _, _ = self.momentum_integration(
                sigma_t_step2, x_t_gan_in, sigma_t_step2, raw_t_final,
                policy_step2, eps=policy_eps, seq_len=seq_len)

        log_vars.update(self.flow_loss.log_vars)
        log_vars.update(
            loss=float(loss.detach()),
            loss_step1=float(loss_step1.detach()),
            loss_step2_teacher=float(loss_step2_teacher.detach()),
            loss_step2_direct=float(loss_step2_diff.detach()),
        )

        if return_step2_latent:
            return loss, log_vars, dict(step2_latent=step2_latent)
        return loss, log_vars


@MODULES.register_module()
class ArcFlowEditImitationSplitStageDualLoraGAN(ArcFlowEditImitationSplitStageGAN):
    """Split-stage GAN with independent step-1 / step-2 LoRA adapters on the student.

    Step-1 ``pred`` uses the ``step1`` adapter; step-2 ``pred`` and the GAN rollout use
    ``step2``. Step-2 loss is PIID + direct flow MSE + GAN. GAN generator loss therefore
    backpropagates only into step-2 LoRA (plus shared trainable heads reached from the
    step-2 graph). Validation ``forward_test`` switches adapters per rollout step to
    match training.
    """

    LORA_STAGE_STEP1 = 'step1'
    LORA_STAGE_STEP2 = 'step2'

    @classmethod
    def _lora_stage_for_rollout_step(cls, step_index: int) -> str:
        return cls.LORA_STAGE_STEP1 if step_index == 0 else cls.LORA_STAGE_STEP2

    def pred(self, x_t=None, t=None, **kwargs):
        lora_stage = kwargs.pop('lora_stage', None)
        denoising = self.denoising
        if lora_stage is not None and getattr(denoising, 'dual_stage_lora', False):
            denoising.set_lora_stage(lora_stage)
        return super(ArcFlowEditImitationSplitStageGAN, self).pred(x_t, t, **kwargs)

    def forward_train(
            self,
            x_0,
            teacher=None,
            teacher_kwargs=dict(),
            running_status=None,
            return_step2_latent=False,
            **kwargs):
        x_ref = kwargs.pop('x_ref', None)
        if x_ref is None:
            raise ValueError(
                'ArcFlowEditImitationSplitStageDualLoraGAN requires `x_ref` in kwargs.')

        device = get_module_device(self)
        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()
        ndim = x_0.dim()
        assert ndim in [4, 5], f'Invalid x_0 shape: {x_0.shape}. Expected 4D or 5D tensor.'

        num_decay_iters = self.train_cfg.get('num_decay_iters', 0)
        if num_decay_iters > 0:
            teacher_ratio = 1 - min(running_status['iteration'], num_decay_iters) / num_decay_iters
            log_vars = dict(teacher_ratio=teacher_ratio)
        else:
            teacher_ratio = 0.0
            log_vars = dict()

        base_segment_size, final_step_size = self._rollout_segment_sizes()
        policy_eps = self.train_cfg.get('eps', 1e-4)
        x_ref_scale_step2 = self.train_cfg.get('split_stage_step2_x_ref_scale', 1.0)
        w_diff = self.train_cfg.get('split_stage_diffusion_loss_weight', 0.5)
        w_teacher = self.train_cfg.get('split_stage_teacher_loss_weight', 0.5)
        gan_loss_scale = self._gan_loss_scale(running_status)
        log_vars['gan_loss_scale'] = gan_loss_scale

        path_epsilon = torch.randn_like(x_0)

        raw_t_step1 = torch.ones(num_batches, dtype=torch.float32, device=device)
        sigma_t_step1 = self.timestep_sampler.warp_t(raw_t_step1, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_step1 = sigma_t_step1.flatten() * self.num_timesteps
        x_t_step1 = path_epsilon

        denoising_output_step1 = self.pred(
            x_t_step1, t_step1, lora_stage=self.LORA_STAGE_STEP1, **kwargs)
        policy_step1 = self.policy_class(
            denoising_output_step1, x_t_step1, sigma_t_step1, x_ref=x_ref,
            path_epsilon=path_epsilon, eps=policy_eps)

        loss_step1, _, _ = self.piid_segment_momentum(
            teacher, policy_step1, x_t_step1, raw_t_step1, sigma_t_step1,
            teacher_ratio, base_segment_size, teacher_kwargs)

        loss = loss_step1

        raw_t_step2 = raw_t_step1 - base_segment_size
        x_t_step2, sigma_t_step2, t_step2 = self.momentum_integration(
            sigma_t_step1, x_t_step1, sigma_t_step1, raw_t_step2,
            policy_step1, eps=policy_eps, seq_len=seq_len)

        x_ref_step2 = x_ref * x_ref_scale_step2
        denoising_output_step2 = self.pred(
            x_t_step2, t_step2, lora_stage=self.LORA_STAGE_STEP2, **kwargs)
        policy_step2 = self.policy_class(
            denoising_output_step2, x_t_step2, sigma_t_step2, x_ref=x_ref_step2,
            path_epsilon=path_epsilon, eps=policy_eps)

        loss_step2_teacher, _, _ = self.piid_segment_momentum(
            teacher, policy_step2, x_t_step2, raw_t_step2, sigma_t_step2,
            teacher_ratio, final_step_size, teacher_kwargs)
        loss_step2_diff = self._direct_flow_loss(
            x_0, policy_step2, t_step2, sigma_t_step2)

        loss = loss + w_teacher * loss_step2_teacher + w_diff * loss_step2_diff

        step2_latent = None
        if gan_loss_scale > 0:
            raw_t_final = raw_t_step2 - final_step_size
            x_t_gan_in = (
                x_t_step2.detach()
                if self.train_cfg.get('gan_grad_step2_only', True)
                else x_t_step2)
            step2_latent, _, _ = self.momentum_integration(
                sigma_t_step2, x_t_gan_in, sigma_t_step2, raw_t_final,
                policy_step2, eps=policy_eps, seq_len=seq_len)

        log_vars.update(self.flow_loss.log_vars)
        log_vars.update(
            loss=float(loss.detach()),
            loss_step1=float(loss_step1.detach()),
            loss_step2_teacher=float(loss_step2_teacher.detach()),
            loss_step2_direct=float(loss_step2_diff.detach()),
        )

        if return_step2_latent:
            return loss, log_vars, dict(step2_latent=step2_latent)
        return loss, log_vars

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=None,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        x_ref = kwargs.get('image_latents')
        if x_ref is None:
            raise ValueError(
                'ArcFlowEditImitationSplitStageDualLoraGAN inference requires '
                '`image_latents` (source latents).')

        import sys
        import mmcv

        x_t_src = torch.randn_like(x_0) if noise is None else noise
        path_epsilon = x_t_src.clone()
        num_batches = x_t_src.size(0)
        seq_len = x_t_src.shape[2:].numel()
        ori_dtype = x_t_src.dtype
        device = x_t_src.device
        x_t_src = x_t_src.float()
        path_epsilon = path_epsilon.float()
        ndim = x_t_src.dim()
        assert ndim in [4, 5], f'Invalid x_t_src shape: {x_t_src.shape}. Expected 4D or 5D tensor.'

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        eps = cfg.get('eps', 1e-4)
        nfe = cfg['nfe']
        if nfe != 2:
            raise ValueError(
                'ArcFlowEditImitationSplitStageDualLoraGAN expects nfe=2 at inference, '
                f'got nfe={nfe}.')
        base_segment_size, final_step_size = self._rollout_segment_sizes()
        x_ref_scale_step2 = self._split_stage_step2_x_ref_scale(cfg)

        raw_t_src = torch.ones((num_batches,), dtype=torch.float32, device=device)
        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps

        if show_pbar:
            pbar = mmcv.ProgressBar(nfe)

        for step_id in range(nfe):
            is_final_step = step_id == nfe - 1
            segment_size = base_segment_size if not is_final_step else final_step_size

            raw_t_dst = raw_t_src - segment_size

            denoising_output = self.pred(
                x_t_src, t_src,
                lora_stage=self._lora_stage_for_rollout_step(step_id),
                **kwargs)
            x_ref_policy = x_ref if step_id == 0 else x_ref * x_ref_scale_step2
            policy = self.policy_class(
                denoising_output, x_t_src, sigma_t_src, x_ref=x_ref_policy,
                path_epsilon=path_epsilon, eps=eps)
            if not is_final_step:
                temperature = cfg.get('temperature', 1.0)
                policy.temperature_(temperature)

            x_t_dst, sigma_t_dst, t_dst = self.momentum_integration(
                sigma_t_src, x_t_src, sigma_t_src, raw_t_dst,
                policy, eps=eps, seq_len=seq_len)

            x_t_src = x_t_dst
            raw_t_src = raw_t_dst
            sigma_t_src = sigma_t_dst
            t_src = t_dst

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t_src.to(ori_dtype)


@MODULES.register_module()
class ArcFlowEditImitationStep2GAN(ArcFlowEditImitation):
    """Standard PIID edit distillation with optional step-2 endpoint for DINOv3 GAN.

    Training loss follows ``ArcFlowEditImitation`` (random-segment PIID, not split-stage
    rollout). When ``return_step2_latent=True``, additionally runs the same 2-NFE rollout
    as ``forward_test`` (path_epsilon at t=1 -> t=0) for GAN. With ``gan_grad_step2_only``
    (default True), intermediate rollout states are detached so GAN gradients update the
    shared student weights only through the final NFE step.
    """

    def _rollout_nfe_latent(
            self,
            x_ref,
            path_epsilon,
            kwargs,
            grad_step2_only=None,
            return_step2_alpha=False):
        device = path_epsilon.device
        num_batches = path_epsilon.size(0)
        seq_len = path_epsilon.shape[2:].numel()
        ndim = path_epsilon.dim()

        if grad_step2_only is None:
            grad_step2_only = self.train_cfg.get('gan_grad_step2_only', True)

        cfg_eps = self.train_cfg.get('eps', 1e-4)
        nfe = self.train_cfg['nfe']
        timestep_ratio = max(self.train_cfg.get('timestep_ratio', 1.0), cfg_eps)
        base_segment_size = 1 / (nfe - 1 + timestep_ratio)

        x_t_src = path_epsilon.float()
        path_eps = path_epsilon.float()
        raw_t_src = torch.ones((num_batches,), dtype=torch.float32, device=device)
        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps
        step2_alpha = None

        for step_id in range(nfe):
            is_final_step = step_id == nfe - 1
            segment_size = base_segment_size * timestep_ratio if is_final_step else base_segment_size
            raw_t_dst = raw_t_src - segment_size

            denoising_output = self.pred(x_t_src, t_src, **kwargs)
            policy = self.policy_class(
                denoising_output, x_t_src, sigma_t_src, x_ref=x_ref,
                path_epsilon=path_eps, eps=cfg_eps)
            if return_step2_alpha and is_final_step and hasattr(policy, 'alpha'):
                step2_alpha = policy.alpha
            if not is_final_step:
                temperature = self.train_cfg.get('temperature', 1.0)
                policy.temperature_(temperature)

            x_t_dst, sigma_t_dst, t_dst = self.momentum_integration(
                sigma_t_src, x_t_src, sigma_t_src, raw_t_dst,
                policy, eps=cfg_eps, seq_len=seq_len)

            if grad_step2_only and not is_final_step:
                x_t_dst = x_t_dst.detach()

            x_t_src = x_t_dst
            raw_t_src = raw_t_dst
            sigma_t_src = sigma_t_dst
            t_src = t_dst

        if return_step2_alpha:
            return x_t_src, step2_alpha
        return x_t_src

    def forward_train(
            self,
            x_0,
            teacher=None,
            teacher_kwargs=dict(),
            running_status=None,
            return_step2_latent=False,
            **kwargs):
        x_ref = kwargs.pop('x_ref', None)
        if x_ref is None:
            raise ValueError(
                'ArcFlowEditImitationStep2GAN requires `x_ref` (source/reference latents) in kwargs.')

        device = get_module_device(self)
        num_batches = x_0.size(0)
        seq_len = x_0.shape[2:].numel()
        ndim = x_0.dim()
        assert ndim in [4, 5], f'Invalid x_0 shape: {x_0.shape}. Expected 4D or 5D tensor.'

        num_decay_iters = self.train_cfg.get('num_decay_iters', 0)
        if num_decay_iters > 0:
            teacher_ratio = 1 - min(running_status['iteration'], num_decay_iters) / num_decay_iters
            log_vars = dict(teacher_ratio=teacher_ratio)
        else:
            teacher_ratio = 0.0
            log_vars = dict()

        raw_t_src, sigma_t_src, t_src, segment_size = self.sample_t(
            num_batches, ndim, seq_len=seq_len, device=device)

        policy_kwargs = dict(eps=self.train_cfg.get('eps', 1e-4))
        path_epsilon = torch.randn_like(x_0)
        x_t_src, _, _ = self.sample_forward_diffusion(x_0, t_src, path_epsilon)
        policy_kwargs['path_epsilon'] = path_epsilon

        denoising_output = self.pred(x_t_src, t_src, **kwargs)
        policy = self.policy_class(
            denoising_output, x_t_src, sigma_t_src, x_ref=x_ref, **policy_kwargs)

        loss_diffusion, _, raw_t_dst = self.piid_segment_momentum(
            teacher, policy, x_t_src, raw_t_src, sigma_t_src, teacher_ratio, segment_size,
            teacher_kwargs)

        loss = loss_diffusion
        loss, log_vars = self._maybe_add_direct_delta_loss(
            loss, log_vars, x_0, policy, t_src, sigma_t_src)
        log_vars.update(self.flow_loss.log_vars)
        log_vars.update(loss_diffusion=float(loss_diffusion.detach()))

        if return_step2_latent:
            gan_scale = 0.0
            if running_status is not None:
                from lakonlab.models.architecture.dinov3_discriminator import split_stage_gan_loss_scale
                gan_scale = split_stage_gan_loss_scale(running_status, self.train_cfg)
            log_vars['gan_loss_scale'] = gan_scale
            step2_latent = None
            step2_alpha = None
            if gan_scale > 0:
                rollout = self._rollout_nfe_latent(
                    x_ref, path_epsilon, kwargs, return_step2_alpha=True)
                step2_latent, step2_alpha = rollout
            return loss, log_vars, dict(step2_latent=step2_latent, step2_alpha=step2_alpha)
        return loss, log_vars
