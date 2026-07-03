# Copyright (c) 2026 EditFlow contributors

from mmgen.models.builder import MODULES

from .arcflow_edit import ArcFlowEditImitation


@MODULES.register_module()
class ArcFlowEditAlphaImitation(ArcFlowEditImitation):
    """Fixed-eps edit distillation with per-patch alpha on the x_ref term."""

    def __init__(self, *args, policy_type='ArcFlowEditNewAlpha', policy_kwargs=None, **kwargs):
        super().__init__(*args, policy_type=policy_type, policy_kwargs=policy_kwargs, **kwargs)

    def policy_average_u_momentum(
            self,
            sigma_t_src,
            x_t_start,
            sigma_t_start,
            raw_t_start,
            raw_t_end,
            total_substeps,
            policy,
            seq_len=None,
            eps=1e-4):
        del x_t_start, raw_t_start, raw_t_end, total_substeps, seq_len, eps
        return policy.train_velocity(sigma_t_src, sigma_t_start)
