# Copyright (c) 2026 EditFlow contributors
"""DMD2-native GAN classification head for FLUX transformer features.

Faithful analogue of the official DMD2 ``cls_pred_branch`` (sd_guidance.py):
the discriminator shares the *fake score* network backbone; only this small
prediction head is a separate trainable module. Features come from the fake
FLUX transformer run in ``classify_mode`` (final image-token representation
before ``proj_out``, the transformer analogue of the UNet bottleneck).

Official SDXL head: strided-conv stack pooling the 32x32 bottleneck to a
single logit. For token features we mean-pool over tokens and apply an MLP.
"""

import torch
import torch.nn as nn

from mmgen.models.builder import MODULES


@MODULES.register_module()
class FluxDMD2ClsHead(nn.Module):
    """Maps fake-score FLUX token features (B, L, D) to realism logits (B, 1)."""

    def __init__(self, feature_dim=3072, hidden_dim=1024):
        super().__init__()
        self.cls_pred_branch = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        # (B, L, D) -> (B, D): spatial pooling, matching the official conv
        # stack that pools the UNet bottleneck down to a single vector.
        pooled = features.mean(dim=1)
        pooled = pooled.to(self.cls_pred_branch[0].weight.dtype)
        return self.cls_pred_branch(pooled).float()
