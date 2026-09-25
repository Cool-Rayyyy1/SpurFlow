# Copyright (c) 2026 SpurFlow contributors

import torch

from typing import Dict, Optional

from .base import BasePolicy


class ArcFlowEditPolicy(BasePolicy):
    """SpurFlow policy with residual parameterization.

    pred_delta = mixture(deltax; weights, gammas) ~ x0_tgt - x_ref
    student_u = path_epsilon - x_ref - pred_delta
    """

    def __init__(
            self,
            denoising_output: Dict[str, torch.Tensor],
            x_t_src: torch.Tensor,
            sigma_t_src: torch.Tensor,
            x_ref: torch.Tensor,
            path_epsilon: Optional[torch.Tensor] = None,
            checkpointing: bool = True,
            eps: float = 1e-4):
        self.x_t_src = x_t_src
        self.x_ref = x_ref
        self.path_epsilon = path_epsilon
        self.ndim = x_t_src.dim()
        self.checkpointing = checkpointing
        self.eps = eps

        self.sigma_t_src = sigma_t_src.reshape(
            *sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.denoising_output_x_0 = self._pack_denoising_output(denoising_output)

    @staticmethod
    def _pack_denoising_output(denoising_output):
        deltax = denoising_output['deltax']
        return dict(
            deltax=deltax,
            pred_delta=deltax,
            means_u=deltax,
            logweights=denoising_output['logweights'],
            loggammas=denoising_output.get('loggammas', None),
        )

    def compute_pred_delta(self, sigma_t_src, sigma_t):
        deltax = self.denoising_output_x_0['deltax']
        log_gammas = self.denoising_output_x_0['loggammas']
        logweights = self.denoising_output_x_0['logweights']
        weights = torch.softmax(logweights, dim=1)

        dt_past = sigma_t_src - sigma_t
        dt_past = dt_past.unsqueeze(1)
        decay_factor = torch.exp(log_gammas * dt_past)
        bs, k, c, h, w = decay_factor.shape
        decay_factor = torch.cat(
            [torch.ones((bs, 1, c, h, w), device=decay_factor.device), decay_factor],
            dim=1)
        mixture_terms = deltax * decay_factor
        return (weights * mixture_terms).sum(dim=1)

    def velocity(self, sigma_t_src, sigma_t):
        if self.path_epsilon is None:
            raise ValueError(
                'ArcFlowEditPolicy requires `path_epsilon` for residual velocity.')
        pred_delta = self.compute_pred_delta(sigma_t_src, sigma_t)
        return self.path_epsilon - self.x_ref - pred_delta

    def copy(self):
        new_policy = ArcFlowEditPolicy.__new__(ArcFlowEditPolicy)
        new_policy.x_t_src = self.x_t_src
        new_policy.x_ref = self.x_ref
        new_policy.path_epsilon = self.path_epsilon
        new_policy.ndim = self.ndim
        new_policy.checkpointing = self.checkpointing
        new_policy.eps = self.eps
        new_policy.sigma_t_src = self.sigma_t_src
        new_policy.denoising_output_x_0 = self.denoising_output_x_0.copy()
        return new_policy

    def detach_(self):
        self.denoising_output_x_0 = {
            k: v.detach() for k, v in self.denoising_output_x_0.items() if v is not None}
        return self

    def detach(self):
        new_policy = self.copy()
        return new_policy.detach_()

    def dropout_(self, p):
        if p <= 0 or p >= 1:
            return self
        logweights = self.denoising_output_x_0['logweights']
        dropout_mask = torch.rand(
            (*logweights.shape[:2], *((self.ndim - 1) * [1])), device=logweights.device) < p
        is_all_dropout = dropout_mask.all(dim=1, keepdim=True)
        dropout_mask &= ~is_all_dropout
        self.denoising_output_x_0['logweights'] = logweights.masked_fill(
            dropout_mask, float('-inf'))
        return self

    def dropout(self, p):
        new_policy = self.copy()
        return new_policy.dropout_(p)

    def temperature_(self, temp):
        temp = max(float(temp), self.eps)
        if temp != 1.0:
            self.denoising_output_x_0['logweights'] = (
                self.denoising_output_x_0['logweights'] / temp)
        return self

    def temperature(self, temp):
        new_policy = self.copy()
        return new_policy.temperature_(temp)

    def mode_(self):
        """Keep only the argmax mixture component (per spatial weight map).

        After this, softmax(weights) is one-hot on dim=K, so
        ``compute_pred_delta`` / ``velocity`` use a single δ_k* instead of the mean.
        """
        logweights = self.denoising_output_x_0['logweights']
        max_idx = logweights.argmax(dim=1, keepdim=True)
        mode_logweights = torch.full_like(logweights, float('-inf'))
        mode_logweights.scatter_(1, max_idx, 0.0)
        self.denoising_output_x_0['logweights'] = mode_logweights
        return self

    def mode(self):
        new_policy = self.copy()
        return new_policy.mode_()
