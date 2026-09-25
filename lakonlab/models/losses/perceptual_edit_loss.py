# Copyright (c) 2026 EditFlow contributors
"""LPIPS + frozen DINOv3 feature losses for edit-endpoint supervision."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv

from lakonlab.models.architecture.dinov3_discriminator import load_dinov3_vitl16_from_hf

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_LPIPS_WEIGHTS = (
    '/path/to/pretrained_models/lpips/vgg.pth')
DEFAULT_VGG16_WEIGHTS = (
    '/path/to/pretrained_models/lpips/vgg16-397923af.pth')
DEFAULT_DINO_WEIGHTS = (
    '/path/to/pretrained_models/'
    'dinov3-vitl16-pretrain-lvd1689m/model.safetensors')


class _Vgg16Slices(nn.Module):
    """VGG16 feature slices used by LPIPS (same cuts as mmgen / richzhang)."""

    def __init__(self, backbone_weights: Optional[Union[str, Path]] = None):
        super().__init__()
        if backbone_weights is not None and Path(backbone_weights).is_file():
            vgg = tv.vgg16(weights=None)
            state = torch.load(str(backbone_weights), map_location='cpu')
            # torchvision checkpoint is a flat state_dict
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            vgg.load_state_dict(state, strict=True)
        else:
            # Falls back to torchvision download into TORCH_HOME / hub cache.
            try:
                from torchvision.models import VGG16_Weights
                vgg = tv.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            except Exception:
                vgg = tv.vgg16(pretrained=True)
        feats = vgg.features
        self.slice1 = nn.Sequential(*[feats[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[feats[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[feats[i] for i in range(9, 16)])
        self.slice4 = nn.Sequential(*[feats[i] for i in range(16, 23)])
        self.slice5 = nn.Sequential(*[feats[i] for i in range(23, 30)])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        outs = namedtuple(
            'VggOutputs',
            ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        return outs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


class LocalLPIPS(nn.Module):
    """LPIPS (VGG) from local lin-layer + local VGG16 backbone checkpoints.

    - ``weights_path``: LPIPS linear layers (mmgen / richzhang ``v0.1/vgg.pth``)
    - ``vgg_weights_path``: torchvision ``vgg16-397923af.pth``
    """

    def __init__(
            self,
            weights_path: Union[str, Path] = DEFAULT_LPIPS_WEIGHTS,
            vgg_weights_path: Union[str, Path] = DEFAULT_VGG16_WEIGHTS,
            spatial: bool = False):
        super().__init__()
        from mmgen.models.architectures.lpips.networks_basic import (
            NetLinLayer,
            ScalingLayer,
            normalize_tensor,
            spatial_average,
            upsample,
        )

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f'LPIPS lin weights not found: {weights_path}. '
                'Download v0.1 vgg.pth into pretrained_models/lpips/.')

        self.spatial = bool(spatial)
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.L = len(self.channels)

        vgg_path = Path(vgg_weights_path) if vgg_weights_path else None
        if vgg_path is not None and not vgg_path.is_file():
            vgg_path = None
        self.net = _Vgg16Slices(backbone_weights=vgg_path)

        self.lin0 = NetLinLayer(self.channels[0], use_dropout=True)
        self.lin1 = NetLinLayer(self.channels[1], use_dropout=True)
        self.lin2 = NetLinLayer(self.channels[2], use_dropout=True)
        self.lin3 = NetLinLayer(self.channels[3], use_dropout=True)
        self.lin4 = NetLinLayer(self.channels[4], use_dropout=True)
        self.lins = nn.ModuleList(
            [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4])

        state = torch.load(str(weights_path), map_location='cpu')
        self.load_state_dict(state, strict=False)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        # Keep helpers as attrs for forward (avoid re-import each call).
        self._normalize_tensor = normalize_tensor
        self._spatial_average = spatial_average
        self._upsample = upsample

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Images in [0, 1]; returns scalar LPIPS."""
        pred = pred.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)
        pred_n = pred * 2.0 - 1.0
        target_n = target * 2.0 - 1.0
        with torch.amp.autocast('cuda', enabled=False):
            in0 = self.scaling_layer(target_n)
            in1 = self.scaling_layer(pred_n)
            outs0 = self.net(in0)
            outs1 = self.net(in1)
            res = []
            for kk in range(self.L):
                feats0 = self._normalize_tensor(outs0[kk])
                feats1 = self._normalize_tensor(outs1[kk])
                diffs = (feats0 - feats1) ** 2
                if self.spatial:
                    res.append(
                        self._upsample(
                            self.lins[kk].model(diffs), out_H=pred.shape[2]))
                else:
                    res.append(
                        self._spatial_average(
                            self.lins[kk].model(diffs), keepdim=True))
            return torch.stack(res, dim=0).sum(dim=0).mean()


class FrozenDinoFeatureLoss(nn.Module):
    """Cosine / MSE distance on frozen DINOv3 ViT-L/16 features."""

    def __init__(
            self,
            checkpoint_path: Union[str, Path] = DEFAULT_DINO_WEIGHTS,
            input_size: int = 224,
            feature_layers: Sequence[int] = (23,),
            backbone_dtype: str = 'bf16',
            loss_type: str = 'cosine'):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'DINOv3 weights not found: {checkpoint_path}')

        dtype_map = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
        self.backbone_dtype = dtype_map[str(backbone_dtype).lower()]
        self.input_size = int(input_size)
        self.feature_layers = tuple(int(v) for v in feature_layers)
        self.loss_type = str(loss_type).lower()

        self.backbone = load_dinov3_vitl16_from_hf(str(checkpoint_path))
        if self.backbone_dtype != torch.float32:
            self.backbone = self.backbone.to(dtype=self.backbone_dtype)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.register_buffer(
            'mean',
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'std',
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)

    def _extract(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float().clamp(0.0, 1.0)
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False)
        x = ((x - self.mean) / self.std).to(self.backbone_dtype)
        self.backbone.eval()
        with torch.autocast(device_type=x.device.type, enabled=False):
            feats = self.backbone.forward_intermediates(
                x,
                indices=list(self.feature_layers),
                norm=False,
                output_fmt='NCHW',
                intermediates_only=True,
            )
        if torch.is_tensor(feats):
            feats = (feats,)
        pooled = [f.float().mean(dim=(2, 3)) for f in feats]
        return torch.cat(pooled, dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = self._extract(pred)
        with torch.no_grad():
            tgt_f = self._extract(target)
        pred_f = F.normalize(pred_f, dim=1)
        tgt_f = F.normalize(tgt_f, dim=1)
        if self.loss_type == 'mse':
            return (pred_f - tgt_f).pow(2).mean()
        return (1.0 - (pred_f * tgt_f).sum(dim=1)).mean()
