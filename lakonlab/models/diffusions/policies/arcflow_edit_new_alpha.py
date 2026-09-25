# Copyright (c) 2026 SpurFlow contributors

import torch

from typing import Dict

from .arcflow_edit_new import ArcFlowEditNewPolicy


class ArcFlowEditNewAlphaPolicy(ArcFlowEditNewPolicy):
    """Edit policy with continuous per-latent-position alpha gating x_ref.

    Each patch predicts four sigmoid channels, one for each position in its
    2x2 latent area. Alpha multiplies x_ref directly (no 1 + alpha).
    student_u = path_epsilon - alpha * x_ref - pred_delta
    """

    def __init__(
            self,
            denoising_output,
            x_t_src,
            sigma_t_src,
            x_ref,
            path_epsilon,
            checkpointing: bool = True,
            eps: float = 1e-4):
        if 'alpha' not in denoising_output:
            raise ValueError(
                'ArcFlowEditNewAlphaPolicy requires `alpha` in denoising_output.')
        self.alpha = denoising_output['alpha']
        super().__init__(
            denoising_output, x_t_src, sigma_t_src, x_ref,
            path_epsilon=path_epsilon, checkpointing=checkpointing, eps=eps)

    def train_velocity(self, sigma_t_src, sigma_t):
        pred_delta = self.compute_pred_delta(sigma_t_src, sigma_t)
        return self.path_epsilon - self.alpha * self.x_ref - pred_delta

    def loss_velocity(self, sigma_t_src, sigma_t):
        return self.train_velocity(sigma_t_src, sigma_t)

    def velocity(self, sigma_t_src, sigma_t):
        return self.train_velocity(sigma_t_src, sigma_t)

    def copy(self):
        new_policy = ArcFlowEditNewAlphaPolicy.__new__(ArcFlowEditNewAlphaPolicy)
        new_policy.x_t_src = self.x_t_src
        new_policy.x_ref = self.x_ref
        new_policy.path_epsilon = self.path_epsilon
        new_policy.alpha = self.alpha
        new_policy.ndim = self.ndim
        new_policy.checkpointing = self.checkpointing
        new_policy.eps = self.eps
        new_policy.sigma_t_src = self.sigma_t_src
        new_policy.denoising_output_x_0 = self.denoising_output_x_0.copy()
        return new_policy

    def detach_(self):
        super().detach_()
        if self.alpha is not None:
            self.alpha = self.alpha.detach()
        return self
