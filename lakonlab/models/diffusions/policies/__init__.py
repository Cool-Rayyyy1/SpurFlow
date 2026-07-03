from .arcflow import ArcFlowPolicy
from .arcflow_edit import ArcFlowEditPolicy
from .arcflow_edit_new import ArcFlowEditNewPolicy
from .arcflow_edit_new_eps import ArcFlowEditNewEpsPolicy
from .arcflow_edit_new_alpha import ArcFlowEditNewAlphaPolicy

POLICY_CLASSES = dict(
    ArcFlow=ArcFlowPolicy,
    ArcFlowEdit=ArcFlowEditPolicy,
    ArcFlowEditNew=ArcFlowEditNewPolicy,
    ArcFlowEditNewEps=ArcFlowEditNewEpsPolicy,
    ArcFlowEditNewAlpha=ArcFlowEditNewAlphaPolicy,
)
