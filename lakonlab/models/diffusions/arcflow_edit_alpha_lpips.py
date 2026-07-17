# Copyright (c) 2026 EditFlow contributors

import torch

from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES

from .arcflow_edit import ArcFlowEditImitation


@MODULES.register_module()
class ArcFlowEditAlphaLpipsImitation(ArcFlowEditImitation):
    """Alpha PIID + analytic x0 from the same random-segment prediction.

    ``forward_train`` keeps random-segment PIID. When ``return_x0_hat=True``,
    also returns the analytic endpoint from that same student prediction
    (no extra 2-NFE rollout):

        x0_hat = alpha * x_ref + pred_delta
    """

    def __init__(self, *args, policy_type='ArcFlowEditNewAlpha', policy_kwargs=None, **kwargs):
        super().__init__(*args, policy_type=policy_type, policy_kwargs=policy_kwargs, **kwargs)

    def _policy_x0(self, policy, sigma_t_src):
        pred_delta = policy.compute_pred_delta(sigma_t_src, sigma_t_src)
        if hasattr(policy, 'alpha'):
            return policy.alpha * policy.x_ref + pred_delta
        return policy.x_ref + pred_delta

    def forward_train(
            self,
            x_0,
            teacher=None,
            teacher_kwargs=dict(),
            running_status=None,
            return_x0_hat=False,
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

        if return_x0_hat:
            # Same random-segment prediction as PIID; no extra rollout.
            x0_hat = self._policy_x0(policy, sigma_t_src)
            return loss, log_vars, dict(x0_hat=x0_hat)
        return loss, log_vars
