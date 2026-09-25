import torch
import torch.nn as nn
from functools import partial
from dataclasses import dataclass
from mmcv.cnn import constant_init, xavier_init
from diffusers.utils import BaseOutput


@dataclass
class ArcFlowModelOutput(BaseOutput):
    """
    The output of ArcFlow models.

    Args:
        means (`torch.Tensor` of shape `(batch_size, num_gaussians, num_channels, height, width)` or
        `(batch_size, num_gaussians, num_channels, frame, height, width)`):
            Gaussian mixture means.
        logweights (`torch.Tensor` of shape `(batch_size, num_gaussians, 1, height, width)` or
        `(batch_size, num_gaussians, 1, frame, height, width)`):
            Gaussian mixture log-weights (logits).
        """

    means: torch.Tensor
    logweights: torch.Tensor
    loggammas: torch.Tensor


@dataclass
class ArcFlowEditModelOutput(BaseOutput):
    """SpurFlow student output: per-component edit delta, noise, mixture weights, and gammas."""

    deltax: torch.Tensor
    epsilon: torch.Tensor
    logweights: torch.Tensor
    loggammas: torch.Tensor


@dataclass
class ArcFlowEditNewModelOutput(BaseOutput):
    """SpurFlow-new: deltax mixture only; path noise epsilon comes from data sampling."""

    deltax: torch.Tensor
    logweights: torch.Tensor
    loggammas: torch.Tensor


@dataclass
class ArcFlowEditNewEpsModelOutput(BaseOutput):
    """SpurFlow-new with learned path epsilon head (teacher inversion target at train)."""

    deltax: torch.Tensor
    epsilon: torch.Tensor
    logweights: torch.Tensor
    loggammas: torch.Tensor


@dataclass
class ArcFlowEditNewAlphaModelOutput(BaseOutput):
    """SpurFlow-new with one sigmoid alpha per latent position in each patch."""

    deltax: torch.Tensor
    logweights: torch.Tensor
    loggammas: torch.Tensor
    alpha: torch.Tensor

