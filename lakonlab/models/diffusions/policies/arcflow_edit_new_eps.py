# Copyright (c) 2026 EditFlow contributors

import torch

from typing import Dict, Optional

from .arcflow_edit_new import ArcFlowEditNewPolicy


class ArcFlowEditNewEpsPolicy(ArcFlowEditNewPolicy):
    """Edit policy with learned path epsilon head.

    pred_epsilon = network epsilon head (teacher inversion target at train)
    pred_delta = mixture(deltax)
    student_u = pred_epsilon - x_ref - pred_delta
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
        if path_epsilon is None:
            if 'epsilon' not in denoising_output:
                raise ValueError(
                    'ArcFlowEditNewEpsPolicy requires `epsilon` in denoising_output '
                    'or an explicit `path_epsilon`.')
            path_epsilon = denoising_output['epsilon']
        super().__init__(
            denoising_output, x_t_src, sigma_t_src, x_ref,
            path_epsilon=path_epsilon, checkpointing=checkpointing, eps=eps)
