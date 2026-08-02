"""Qwen-Image-Edit alpha transformer using shifted Softsign."""

import torch

from mmgen.models.builder import MODULES

from .arcqwen_edit_alpha import ArcQwenEditAlphaImageTransformer2DModel


@MODULES.register_module()
class ArcQwenEditAlphaSoftsign01ImageTransformer2DModel(
        ArcQwenEditAlphaImageTransformer2DModel):
    """Map alpha logits with Softsign-01 instead of sigmoid.

    ``alpha = 0.5 * (x / (1 + abs(x)) + 1)`` remains in ``(0, 1)`` and maps
    zero logits to 0.5.
    """

    def _activate_alpha(self, logits: torch.Tensor) -> torch.Tensor:
        return 0.5 * (logits / (1.0 + logits.abs()) + 1.0)
