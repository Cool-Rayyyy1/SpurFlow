# Copyright (c) 2026 EditFlow contributors

import torch

from typing import Dict

from .arcflow_edit_new_alpha import ArcFlowEditNewAlphaPolicy


class ArcFlowEditNewAlphaHardPolicy(ArcFlowEditNewAlphaPolicy):
    """Alpha policy with per-sample hard (binary) gating of x_ref.

    The soft Softsign-01 alpha map is binarized per sample with an adaptive
    threshold (midpoint of the map's min/max):
      - alpha=1 (non-edit region): the x_ref token is copied directly;
        pred_delta is pushed to ~0 there by the teacher / direct-delta loss.
      - alpha=0 (edit region): x_ref is cut off; the GM delta branch
        generates the content.

        student_u = path_epsilon - alpha_hard * x_ref - pred_delta

    Near-uniform alpha maps (max - min < ``uniform_eps``) indicate a global
    edit (e.g. style transfer); the whole map gates to 0 (regenerate all).

    ``ste=True`` uses a straight-through estimator so gradients still reach
    the soft alpha head. With a frozen alpha head (the intended finetune
    setup) use ``ste=False`` so the binary mask is a pure detached routing
    signal.
    """

    def __init__(
            self,
            denoising_output: Dict[str, torch.Tensor],
            x_t_src: torch.Tensor,
            sigma_t_src: torch.Tensor,
            x_ref: torch.Tensor,
            path_epsilon: torch.Tensor,
            checkpointing: bool = True,
            eps: float = 1e-4,
            uniform_eps: float = 0.02,
            ste: bool = False):
        super().__init__(
            denoising_output, x_t_src, sigma_t_src, x_ref,
            path_epsilon=path_epsilon, checkpointing=checkpointing, eps=eps)
        self.uniform_eps = float(uniform_eps)
        self.ste = bool(ste)
        self.alpha_soft = self.alpha
        self.alpha = self._binarize(self.alpha_soft, self.uniform_eps, self.ste)

    @staticmethod
    def _binarize(alpha: torch.Tensor, uniform_eps: float, ste: bool) -> torch.Tensor:
        bs = alpha.size(0)
        expand = [1] * (alpha.dim() - 1)
        flat = alpha.detach().float().reshape(bs, -1)
        amin = flat.min(dim=1).values
        amax = flat.max(dim=1).values
        thr = ((amin + amax) * 0.5).reshape(bs, *expand)
        hard = (alpha.detach() > thr).to(alpha.dtype)
        is_uniform = ((amax - amin) < uniform_eps).reshape(bs, *expand)
        hard = torch.where(is_uniform, torch.zeros_like(hard), hard)
        if ste:
            hard = alpha + (hard - alpha).detach()
        return hard

    def copy(self):
        new_policy = ArcFlowEditNewAlphaHardPolicy.__new__(ArcFlowEditNewAlphaHardPolicy)
        new_policy.x_t_src = self.x_t_src
        new_policy.x_ref = self.x_ref
        new_policy.path_epsilon = self.path_epsilon
        new_policy.alpha = self.alpha
        new_policy.alpha_soft = self.alpha_soft
        new_policy.uniform_eps = self.uniform_eps
        new_policy.ste = self.ste
        new_policy.ndim = self.ndim
        new_policy.checkpointing = self.checkpointing
        new_policy.eps = self.eps
        new_policy.sigma_t_src = self.sigma_t_src
        new_policy.denoising_output_x_0 = self.denoising_output_x_0.copy()
        return new_policy

    def detach_(self):
        super().detach_()
        if self.alpha_soft is not None:
            self.alpha_soft = self.alpha_soft.detach()
        return self
