# Copyright (c) 2026 SpurFlow contributors

from mmgen.models.builder import MODULES

from .arcflow_edit import ArcFlowEditImitation


@MODULES.register_module()
class ArcFlowEditAlphaImitation(ArcFlowEditImitation):
    """Fixed-eps edit distillation with spatial alpha on the x_ref term.

    Training and inference reuse ArcFlowEditImitation's momentum integration
    and interval-average velocity logic. The only diffusion-level difference is
    that the policy scales x_ref by alpha.
    """

    def __init__(self, *args, policy_type='ArcFlowEditNewAlpha', policy_kwargs=None, **kwargs):
        super().__init__(*args, policy_type=policy_type, policy_kwargs=policy_kwargs, **kwargs)
