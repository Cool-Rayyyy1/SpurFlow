# Copyright (c) 2026 EditFlow contributors
#
# TDM-style DINO feature discriminator for EditFlow step-2 GAN:
#   frozen DINOv3 intermediate features + trainable conv head(s)
#   shared global/local random crops for real & fake RGB images in [0, 1]
#   logistic (softplus) D/G losses

from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmgen.models.builder import MODULES
from torch.utils.checkpoint import checkpoint

from .dinov3_discriminator import load_dinov3_vitl16_from_hf


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sample_random_crop_spec(
        scale_range: Sequence[float],
        aspect_range: Sequence[float],
        device: torch.device) -> Tuple[float, float, float, float]:
    min_scale, max_scale = [float(v) for v in scale_range]
    min_aspect, max_aspect = [float(v) for v in aspect_range]
    for _ in range(10):
        area = torch.empty((), device=device).uniform_(min_scale, max_scale).item()
        aspect = math.exp(torch.empty((), device=device).uniform_(
            math.log(min_aspect), math.log(max_aspect)).item())
        crop_h = math.sqrt(area / aspect)
        crop_w = math.sqrt(area * aspect)
        if crop_h <= 1.0 and crop_w <= 1.0:
            top = torch.empty((), device=device).uniform_(0.0, 1.0 - crop_h).item()
            left = torch.empty((), device=device).uniform_(0.0, 1.0 - crop_w).item()
            return top, left, crop_h, crop_w
    side = min(1.0, math.sqrt(max_scale))
    return (1.0 - side) / 2.0, (1.0 - side) / 2.0, side, side


def sample_dino_crop_specs(
        batch_size: int,
        device: torch.device,
        *,
        num_global_crops: int = 1,
        num_local_crops: int = 1,
        global_crop_scale: Sequence[float] = (0.5, 1.0),
        local_crop_scale: Sequence[float] = (0.125, 0.5),
        crop_aspect_ratio: Sequence[float] = (0.75, 1.3333333333),
) -> List[torch.Tensor]:
    """Shared crop specs for real and fake. Index 0 is always the full image."""
    crop_specs = [
        torch.tensor(
            [[0.0, 0.0, 1.0, 1.0]] * int(batch_size),
            device=device,
            dtype=torch.float32,
        )
    ]
    for crop_idx in range(1, int(num_global_crops) + int(num_local_crops)):
        scale_range = global_crop_scale if crop_idx < int(num_global_crops) else local_crop_scale
        crop_specs.append(
            torch.tensor(
                [
                    sample_random_crop_spec(scale_range, crop_aspect_ratio, device)
                    for _ in range(int(batch_size))
                ],
                device=device,
                dtype=torch.float32,
            )
        )
    return crop_specs


def apply_dino_crop_specs(
        image_pixels: torch.Tensor,
        crop_specs: Sequence[torch.Tensor],
        *,
        num_global_crops: int = 1,
        global_input_size: int = 224,
        local_input_size: int = 224,
        clamp_pixels: bool = True,
) -> List[torch.Tensor]:
    batch_size, _, height, width = image_pixels.shape
    crops = []
    for crop_idx, specs in enumerate(crop_specs):
        output_size = global_input_size if crop_idx < int(num_global_crops) else local_input_size
        per_sample = []
        for batch_idx in range(batch_size):
            top, left, crop_h, crop_w = specs[batch_idx].tolist()
            y0 = min(max(int(round(top * height)), 0), height - 1)
            x0 = min(max(int(round(left * width)), 0), width - 1)
            y1 = min(max(int(round((top + crop_h) * height)), y0 + 1), height)
            x1 = min(max(int(round((left + crop_w) * width)), x0 + 1), width)
            per_sample.append(image_pixels[batch_idx:batch_idx + 1, :, y0:y1, x0:x1])

        resized = []
        for crop in per_sample:
            crop = F.interpolate(
                crop.float(),
                size=(output_size, output_size),
                mode='bicubic',
                align_corners=False,
                antialias=True,
            )
            if clamp_pixels:
                crop = crop.clamp(0.0, 1.0)
            resized.append(crop)
        crops.append(torch.cat(resized, dim=0))
    return crops


class DinoConvBlock(nn.Module):

    def __init__(
            self,
            channels: int,
            *,
            stride: int = 1,
            post_avgpool: bool = False,
            norm_groups: int = 32):
        super().__init__()
        groups = min(int(norm_groups), int(channels))
        while int(channels) % groups != 0:
            groups -= 1
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.act = nn.SiLU()
        self.post_avgpool = bool(post_avgpool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.conv(x)))
        if self.post_avgpool and min(x.shape[-2:]) > 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class DinoConvClsHead(nn.Module):

    def __init__(
            self,
            channels: int,
            *,
            num_blocks: int = 3,
            use_avgpool: bool = False,
            dense_output: bool = False,
            gradient_checkpointing: bool = True,
            norm_groups: int = 32):
        super().__init__()
        self.dense_output = bool(dense_output)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        stride = 1 if use_avgpool else 2
        self.blocks = nn.ModuleList([
            DinoConvBlock(
                channels,
                stride=stride,
                post_avgpool=use_avgpool,
                norm_groups=norm_groups,
            )
            for _ in range(int(num_blocks))
        ])

    def forward(self, crops: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        checkpoint_blocks = (
            self.gradient_checkpointing
            and torch.is_grad_enabled()
            and any(p.requires_grad for p in self.parameters()))
        for crop in crops:
            x = crop
            for block in self.blocks:
                if checkpoint_blocks:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            if not self.dense_output:
                x = x.mean(dim=1, keepdim=True)
            outputs.append(x)
        return outputs


class DinoDiscriminatorHead(nn.Module):

    def __init__(
            self,
            *,
            channels: int,
            num_discriminators: int,
            num_steps: int,
            num_blocks: int = 3,
            dense_output: bool = False,
            use_avgpool: bool = False,
            step_conditioning: bool = False,
            gradient_checkpointing: bool = True,
            norm_groups: int = 32):
        super().__init__()
        self.step_conditioning = bool(step_conditioning)
        if self.step_conditioning:
            self.step_embedding = nn.Embedding(int(num_steps), int(channels))
        self.heads = nn.ModuleList([
            DinoConvClsHead(
                channels=channels,
                num_blocks=num_blocks,
                use_avgpool=use_avgpool,
                dense_output=dense_output,
                gradient_checkpointing=gradient_checkpointing,
                norm_groups=norm_groups,
            )
            for _ in range(int(num_discriminators))
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(int(num_discriminators))
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(channels, 1) for _ in range(int(num_discriminators))
        ])

    def forward(
            self,
            intermediate_features: Sequence[Sequence[torch.Tensor]],
            step_indices: torch.Tensor) -> torch.Tensor:
        logits = []
        step_indices = step_indices.long()
        for head, norm, classifier, crops in zip(
                self.heads, self.norms, self.classifiers, intermediate_features):
            if self.step_conditioning:
                step_embedding = self.step_embedding(step_indices)[:, :, None, None]
                crops = [crop + step_embedding.to(crop.dtype) for crop in crops]
            features = torch.cat(head(crops), dim=0)
            logits.append(classifier(norm(features)).squeeze(dim=2))
        return torch.cat(logits, dim=1)


@MODULES.register_module()
class DinoFeatureDiscriminator(nn.Module):
    """Frozen DINOv3 ViT-L/16 features + TDM-style conv discriminator head."""

    def __init__(
            self,
            checkpoint_path: str,
            num_steps: int = 2,
            feature_layers: Sequence[int] = (23,),
            global_input_size: int = 224,
            local_input_size: int = 224,
            num_global_crops: int = 1,
            num_local_crops: int = 1,
            global_crop_scale: Sequence[float] = (0.5, 1.0),
            local_crop_scale: Sequence[float] = (0.125, 0.5),
            crop_aspect_ratio: Sequence[float] = (0.75, 1.3333333333),
            clamp_pixels: bool = True,
            step_conditioning: bool = False,
            head_num_blocks: int = 3,
            head_use_avgpool: bool = False,
            head_gradient_checkpointing: bool = True,
            head_norm_groups: int = 32,
            dense_output: bool = False,
            backbone_dtype: str = 'bf16',
            head_dtype: str = 'fp32',
            freeze_backbone: bool = True):
        super().__init__()
        self.num_steps = int(num_steps)
        self.feature_layers = tuple(int(v) for v in feature_layers)
        self.num_global_crops = int(num_global_crops)
        self.num_local_crops = int(num_local_crops)
        self.global_input_size = int(global_input_size)
        self.local_input_size = int(local_input_size)
        self.global_crop_scale = tuple(float(v) for v in global_crop_scale)
        self.local_crop_scale = tuple(float(v) for v in local_crop_scale)
        self.crop_aspect_ratio = tuple(float(v) for v in crop_aspect_ratio)
        self.clamp_pixels = bool(clamp_pixels)
        self.step_conditioning = bool(step_conditioning)

        dtype_map = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
        self.backbone_dtype = dtype_map[str(backbone_dtype).lower()]
        self.head_dtype = dtype_map[str(head_dtype).lower()]

        self.backbone = load_dinov3_vitl16_from_hf(checkpoint_path)
        if self.backbone_dtype != torch.float32:
            self.backbone = self.backbone.to(dtype=self.backbone_dtype)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        channels = int(self.backbone.num_features)
        self.head = DinoDiscriminatorHead(
            channels=channels,
            num_discriminators=len(self.feature_layers),
            num_steps=self.num_steps,
            num_blocks=int(head_num_blocks),
            dense_output=bool(dense_output),
            use_avgpool=bool(head_use_avgpool),
            step_conditioning=bool(step_conditioning),
            gradient_checkpointing=bool(head_gradient_checkpointing),
            norm_groups=int(head_norm_groups),
        ).to(dtype=self.head_dtype)

        self.register_buffer(
            'mean',
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'std',
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def make_step_indices(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Step-2 supervision for 2-NFE: final rollout index = num_steps - 1."""
        device = device or self.device
        return torch.full(
            (int(batch_size),), self.num_steps - 1, device=device, dtype=torch.long)

    def sample_crop_specs(self, batch_size: int, device: Optional[torch.device] = None) -> List[torch.Tensor]:
        return sample_dino_crop_specs(
            batch_size,
            device or self.device,
            num_global_crops=self.num_global_crops,
            num_local_crops=self.num_local_crops,
            global_crop_scale=self.global_crop_scale,
            local_crop_scale=self.local_crop_scale,
            crop_aspect_ratio=self.crop_aspect_ratio,
        )

    def apply_crop_specs(
            self,
            image_pixels: torch.Tensor,
            crop_specs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return apply_dino_crop_specs(
            image_pixels,
            crop_specs,
            num_global_crops=self.num_global_crops,
            global_input_size=self.global_input_size,
            local_input_size=self.local_input_size,
            clamp_pixels=self.clamp_pixels,
        )

    def _extract_features(
            self,
            image_pixels: torch.Tensor,
            crop_specs: Sequence[torch.Tensor],
            *,
            requires_input_grad: bool) -> List[List[torch.Tensor]]:
        if self.clamp_pixels:
            image_pixels = image_pixels.clamp(0.0, 1.0)
        crops = self.apply_crop_specs(image_pixels.to(self.device), crop_specs)
        features_by_layer: List[List[torch.Tensor]] = [[] for _ in self.feature_layers]
        self.backbone.eval()

        grad_context = contextlib.nullcontext() if requires_input_grad else torch.no_grad()
        with grad_context:
            for crop in crops:
                normalized = ((crop.float() - self.mean) / self.std).to(self.backbone_dtype)
                with torch.autocast(device_type=normalized.device.type, enabled=False):
                    layer_feats = self.backbone.forward_intermediates(
                        normalized,
                        indices=list(self.feature_layers),
                        norm=False,
                        output_fmt='NCHW',
                        intermediates_only=True,
                    )
                if torch.is_tensor(layer_feats):
                    layer_feats = (layer_feats,)
                for layer_idx, feature in enumerate(layer_feats):
                    features_by_layer[layer_idx].append(feature.to(self.head_dtype))
        return features_by_layer

    def logits_from_pixels(
            self,
            image_pixels: torch.Tensor,
            step_indices: torch.Tensor,
            *,
            crop_specs: Optional[Sequence[torch.Tensor]] = None,
            requires_input_grad: bool = False) -> torch.Tensor:
        if crop_specs is None:
            crop_specs = self.sample_crop_specs(image_pixels.shape[0], image_pixels.device)
        features = self._extract_features(
            image_pixels, crop_specs, requires_input_grad=requires_input_grad)
        return self.head(features, step_indices.to(self.device))

    @staticmethod
    def _discriminator_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
        return F.softplus(logits_fake.float()).mean() + F.softplus(-logits_real.float()).mean()

    @staticmethod
    def _generator_loss(logits_fake: torch.Tensor) -> torch.Tensor:
        return F.softplus(-logits_fake.float()).mean()

    @staticmethod
    def build_log_vars(real_logits: torch.Tensor, fake_logits: torch.Tensor, loss_d: torch.Tensor) -> Dict[str, float]:
        return dict(
            loss_d=float(loss_d.detach()),
            d_real=float(real_logits.detach().mean()),
            d_fake=float(fake_logits.detach().mean()),
            dino_gan_realism_real=float(torch.sigmoid(real_logits.float()).detach().mean()),
            dino_gan_realism_fake=float(torch.sigmoid(fake_logits.float()).detach().mean()),
            dino_gan_discriminator_accuracy=float(
                (real_logits.float() > fake_logits.float()).float().mean()),
        )

    @staticmethod
    def build_generator_log_vars(fake_logits: torch.Tensor, loss_g: torch.Tensor) -> Dict[str, float]:
        return dict(
            loss_g_gan=float(loss_g.detach()),
            g_fake_logits=float(fake_logits.detach().mean()),
            dino_gan_generator_realism=float(torch.sigmoid(fake_logits.float()).detach().mean()),
        )

    def forward(
            self,
            images: Optional[torch.Tensor] = None,
            real_images: Optional[torch.Tensor] = None,
            fake_images: Optional[torch.Tensor] = None,
            gan_mode: Optional[str] = None,
            step_indices: Optional[torch.Tensor] = None,
            crop_specs: Optional[Sequence[torch.Tensor]] = None):
        if gan_mode == 'discriminator':
            assert real_images is not None and fake_images is not None
            if step_indices is None:
                step_indices = self.make_step_indices(fake_images.shape[0], fake_images.device)
            if crop_specs is None:
                crop_specs = self.sample_crop_specs(fake_images.shape[0], fake_images.device)
            logits_real = self.logits_from_pixels(
                real_images.detach(), step_indices, crop_specs=crop_specs, requires_input_grad=False)
            logits_fake = self.logits_from_pixels(
                fake_images.detach(), step_indices, crop_specs=crop_specs, requires_input_grad=False)
            return self._discriminator_loss(logits_real, logits_fake)

        if gan_mode == 'generator':
            assert fake_images is not None
            if step_indices is None:
                step_indices = self.make_step_indices(fake_images.shape[0], fake_images.device)
            if crop_specs is None:
                crop_specs = self.sample_crop_specs(fake_images.shape[0], fake_images.device)
            logits_fake = self.logits_from_pixels(
                fake_images, step_indices, crop_specs=crop_specs, requires_input_grad=True)
            return self._generator_loss(logits_fake)

        if images is None:
            raise ValueError('DinoFeatureDiscriminator requires `images` when gan_mode is None.')
        if step_indices is None:
            step_indices = self.make_step_indices(images.shape[0], images.device)
        return self.logits_from_pixels(
            images, step_indices, crop_specs=crop_specs, requires_input_grad=False)
