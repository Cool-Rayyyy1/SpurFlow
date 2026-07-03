# Copyright (c) 2026 EditFlow contributors

import torch

from copy import deepcopy
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES

from .arcflow_edit import ArcFlowEditImitation
from .policies import ArcFlowEditNewEpsPolicy
from lakonlab.utils import module_eval


@MODULES.register_module()
class ArcFlowEditTeacherInvEpsImitation(ArcFlowEditImitation):
    """Teacher-inverted edit distillation with a learned path-epsilon head.

    Train:
      x_t from random input noise (no target leakage in the network input)
      target_epsilon = teacher_invert(x0_tgt)
      pred_epsilon = epsilon head
      student_u = pred_epsilon - x_ref - pred_delta
      loss = PIID(student_u, teacher_u) + MSE(pred_epsilon, target_epsilon)

    Infer:
      x_t starts from sampled random noise
      pred_epsilon from epsilon head at t=1 (fixed for the trajectory)
      student_u = pred_epsilon - x_ref - pred_delta
    """

    def __init__(self, *args, policy_type='ArcFlowEditNewEps', policy_kwargs=None, **kwargs):
        super().__init__(*args, policy_type=policy_type, policy_kwargs=policy_kwargs, **kwargs)

    def _compute_target_epsilon(self, teacher, x_0, teacher_kwargs, seq_len):
        invert_steps = self.train_cfg.get('teacher_invert_steps', 10)
        with torch.no_grad(), module_eval(teacher):
            return self.teacher_invert_x0_to_noise(
                teacher, x_0, teacher_kwargs,
                num_steps=invert_steps, seq_len=seq_len)

    def _compute_epsilon_loss(self, pred_epsilon, target_epsilon):
        loss = torch.nn.functional.mse_loss(pred_epsilon, target_epsilon)
        scale = self.train_cfg.get('epsilon_loss_scale', 1.0)
        return loss * scale

    def forward_train(self, x_0, teacher=None, teacher_kwargs=dict(), running_status=None, **kwargs):
        x_ref = kwargs.pop('x_ref', None)
        if x_ref is None:
            raise ValueError(
                'ArcFlowEditTeacherInvEpsImitation requires `x_ref` (source/reference latents) in kwargs.')
        if teacher is None:
            raise ValueError(
                'ArcFlowEditTeacherInvEpsImitation requires `teacher` for inversion targets.')

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

        target_epsilon = self._compute_target_epsilon(teacher, x_0, teacher_kwargs, seq_len)
        input_noise = torch.randn_like(x_0)
        x_t_src, _, _ = self.sample_forward_diffusion(x_0, t_src, input_noise)

        denoising_output = self.pred(x_t_src, t_src, **kwargs)
        if 'epsilon' not in denoising_output:
            raise ValueError(
                'ArcFlowEditTeacherInvEpsImitation requires denoising output key `epsilon`. '
                'Use ArcFluxEditNewEpsTransformer2DModel for the student denoiser.')

        pred_epsilon = denoising_output['epsilon']
        epsilon_loss = self._compute_epsilon_loss(pred_epsilon, target_epsilon)

        policy = self.policy_class(
            denoising_output, x_t_src, sigma_t_src, x_ref=x_ref,
            path_epsilon=pred_epsilon,
            eps=self.train_cfg.get('eps', 1e-4))

        loss_diffusion, _, raw_t_dst = self.piid_segment_momentum(
            teacher, policy, x_t_src, raw_t_src, sigma_t_src, teacher_ratio, segment_size,
            teacher_kwargs)

        loss = loss_diffusion + epsilon_loss
        log_vars.update(self.flow_loss.log_vars)
        log_vars.update(
            loss_diffusion=float(loss_diffusion.detach()),
            loss_epsilon=float(epsilon_loss.detach()),
            loss=float(loss.detach()),
        )
        return loss, log_vars

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=None,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        x_ref = kwargs.get('image_latents')
        if x_ref is None:
            raise ValueError(
                'ArcFlowEditTeacherInvEpsImitation inference requires `image_latents` (source latents).')

        import sys
        import mmcv

        x_t_src = torch.randn_like(x_0) if noise is None else noise
        num_batches = x_t_src.size(0)
        seq_len = x_t_src.shape[2:].numel()
        ori_dtype = x_t_src.dtype
        device = x_t_src.device
        x_t_src = x_t_src.float()
        ndim = x_t_src.dim()
        assert ndim in [4, 5], f'Invalid x_t_src shape: {x_t_src.shape}. Expected 4D or 5D tensor.'

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        eps = cfg.get('eps', 1e-4)
        nfe = cfg['nfe']
        timestep_ratio = max(cfg.get('timestep_ratio', 1.0), eps)
        base_segment_size = 1 / (nfe - 1 + timestep_ratio)

        raw_t_src = torch.ones((num_batches,), dtype=torch.float32, device=device)
        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps

        init_output = self.pred(x_t_src, t_src, **kwargs)
        if 'epsilon' not in init_output:
            raise ValueError(
                'ArcFlowEditTeacherInvEpsImitation inference requires denoising output key `epsilon`.')
        path_epsilon = init_output['epsilon'].float()

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
