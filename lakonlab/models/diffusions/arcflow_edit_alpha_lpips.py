# Copyright (c) 2026 EditFlow contributors

import torch

from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES

from .arcflow_edit import ArcFlowEditImitationStep2GAN


@MODULES.register_module()
class ArcFlowEditAlphaLpipsImitation(ArcFlowEditImitationStep2GAN):
    """Alpha PIID + optional 2-NFE x0 rollout for LPIPS / DINO losses.

    ``forward_train`` keeps random-segment PIID. When ``return_step_x0s=True``,
    also runs a full 2-NFE rollout (same schedule as inference) and returns
    analytic endpoint estimates at each step:

        x0_hat = alpha * x_ref + pred_delta

    Intermediate rollout states are NOT detached so both step-1 and step-2
    perceptual losses can train the student.
    """

    def __init__(self, *args, policy_type='ArcFlowEditNewAlpha', policy_kwargs=None, **kwargs):
        super().__init__(*args, policy_type=policy_type, policy_kwargs=policy_kwargs, **kwargs)

    def _policy_x0(self, policy, sigma_t_src):
        pred_delta = policy.compute_pred_delta(sigma_t_src, sigma_t_src)
        if hasattr(policy, 'alpha'):
            return policy.alpha * policy.x_ref + pred_delta
        return policy.x_ref + pred_delta

    def _rollout_nfe_x0s(self, x_ref, path_epsilon, kwargs):
        """Full 2-NFE rollout; return list of x0_hat per step (len == nfe)."""
        device = path_epsilon.device
        num_batches = path_epsilon.size(0)
        seq_len = path_epsilon.shape[2:].numel()
        ndim = path_epsilon.dim()

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

        step_x0s = []
        for step_id in range(nfe):
            is_final_step = step_id == nfe - 1
            segment_size = (
                base_segment_size * timestep_ratio if is_final_step else base_segment_size)
            raw_t_dst = raw_t_src - segment_size

            denoising_output = self.pred(x_t_src, t_src, **kwargs)
            policy = self.policy_class(
                denoising_output, x_t_src, sigma_t_src, x_ref=x_ref,
                path_epsilon=path_eps, eps=cfg_eps)
            step_x0s.append(self._policy_x0(policy, sigma_t_src))

            if not is_final_step:
                temperature = self.train_cfg.get('temperature', 1.0)
                policy.temperature_(temperature)

            x_t_dst, sigma_t_dst, t_dst = self.momentum_integration(
                sigma_t_src, x_t_src, sigma_t_src, raw_t_dst,
                policy, eps=cfg_eps, seq_len=seq_len)

            x_t_src = x_t_dst
            raw_t_src = raw_t_dst
            sigma_t_src = sigma_t_dst
            t_src = t_dst

        return step_x0s

    def forward_train(
            self,
            x_0,
            teacher=None,
            teacher_kwargs=dict(),
            running_status=None,
            return_step2_latent=False,
            return_step_x0s=False,
            **kwargs):
        x_ref = kwargs.pop('x_ref', None)
        if x_ref is None:
            raise ValueError(
                'ArcFlowEditAlphaLpipsImitation requires `x_ref` '
                '(source/reference latents) in kwargs.')

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

        if return_step_x0s or return_step2_latent:
            step_x0s = self._rollout_nfe_x0s(x_ref, path_epsilon, kwargs)
            extra = dict(
                step_x0s=step_x0s,
                step2_latent=step_x0s[-1] if step_x0s else None,
            )
            return loss, log_vars, extra
        return loss, log_vars
