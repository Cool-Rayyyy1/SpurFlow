# Copyright (c) 2026 SpurFlow contributors

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.transforms import Resize

from safetensors.torch import load_file
from mmgen.models.builder import MODULES


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_REGISTER_TOKENS = 4


def split_stage_gan_loss_scale(running_status, train_cfg):
    """Return GAN loss multiplier in [0, 1] after warmup + linear ramp."""
    warmup_iters = train_cfg.get('split_stage_gan_warmup_iters', 500)
    ramp_iters = train_cfg.get('split_stage_gan_ramp_iters', 500)
    if running_status is None:
        return 0.0
    iteration = running_status.get('iteration', 0)
    if iteration < warmup_iters:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return min(1.0, float(iteration - warmup_iters) / float(ramp_iters))


def _get_hf_tensor(state_dict, key):
    return state_dict[key] if key in state_dict else None


def load_dinov3_vitl16_from_hf(checkpoint_path):
    """Load a local DINOv3 ViT-L/16 HF safetensors checkpoint into timm."""
    state_dict_hf = load_file(checkpoint_path)
    model = timm.create_model('vit_large_patch16_dinov3', pretrained=False, num_classes=0)
    mapped = {
        'cls_token': state_dict_hf['embeddings.cls_token'],
        'reg_token': state_dict_hf['embeddings.register_tokens'],
        'patch_embed.proj.weight': state_dict_hf['embeddings.patch_embeddings.weight'],
        'patch_embed.proj.bias': state_dict_hf['embeddings.patch_embeddings.bias'],
        'norm.weight': state_dict_hf['norm.weight'],
        'norm.bias': state_dict_hf['norm.bias'],
    }
    for layer_id in range(24):
        hf_prefix = f'layer.{layer_id}'
        timm_prefix = f'blocks.{layer_id}'
        mapped[f'{timm_prefix}.norm1.weight'] = state_dict_hf[f'{hf_prefix}.norm1.weight']
        mapped[f'{timm_prefix}.norm1.bias'] = state_dict_hf[f'{hf_prefix}.norm1.bias']
        mapped[f'{timm_prefix}.norm2.weight'] = state_dict_hf[f'{hf_prefix}.norm2.weight']
        mapped[f'{timm_prefix}.norm2.bias'] = state_dict_hf[f'{hf_prefix}.norm2.bias']
        q_weight = state_dict_hf[f'{hf_prefix}.attention.q_proj.weight']
        k_weight = state_dict_hf[f'{hf_prefix}.attention.k_proj.weight']
        v_weight = state_dict_hf[f'{hf_prefix}.attention.v_proj.weight']
        mapped[f'{timm_prefix}.attn.qkv.weight'] = torch.cat([q_weight, k_weight, v_weight], dim=0)
        q_bias = _get_hf_tensor(state_dict_hf, f'{hf_prefix}.attention.q_proj.bias')
        if q_bias is not None:
            k_bias = _get_hf_tensor(state_dict_hf, f'{hf_prefix}.attention.k_proj.bias')
            v_bias = _get_hf_tensor(state_dict_hf, f'{hf_prefix}.attention.v_proj.bias')
            if k_bias is None:
                k_bias = torch.zeros_like(q_bias)
            if v_bias is None:
                v_bias = torch.zeros_like(q_bias)
            mapped[f'{timm_prefix}.attn.qkv.bias'] = torch.cat([q_bias, k_bias, v_bias], dim=0)
        mapped[f'{timm_prefix}.attn.proj.weight'] = state_dict_hf[f'{hf_prefix}.attention.o_proj.weight']
        o_bias = _get_hf_tensor(state_dict_hf, f'{hf_prefix}.attention.o_proj.bias')
        if o_bias is not None:
            mapped[f'{timm_prefix}.attn.proj.bias'] = o_bias
        mapped[f'{timm_prefix}.mlp.fc1.weight'] = state_dict_hf[f'{hf_prefix}.mlp.up_proj.weight']
        mapped[f'{timm_prefix}.mlp.fc1.bias'] = state_dict_hf[f'{hf_prefix}.mlp.up_proj.bias']
        mapped[f'{timm_prefix}.mlp.fc2.weight'] = state_dict_hf[f'{hf_prefix}.mlp.down_proj.weight']
        mapped[f'{timm_prefix}.mlp.fc2.bias'] = state_dict_hf[f'{hf_prefix}.mlp.down_proj.bias']
        mapped[f'{timm_prefix}.gamma_1'] = state_dict_hf[f'{hf_prefix}.layer_scale1.lambda1']
        mapped[f'{timm_prefix}.gamma_2'] = state_dict_hf[f'{hf_prefix}.layer_scale2.lambda1']
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        raise RuntimeError(f'Failed to map DINOv3 weights, missing keys: {missing[:8]}')
    if unexpected:
        unexpected_bias = [k for k in unexpected if k.endswith('attn.qkv.bias')]
        if len(unexpected_bias) != len(unexpected):
            raise RuntimeError(f'Unexpected DINOv3 keys after mapping: {unexpected[:8]}')
    return model


class _ScoreHead(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, features):
        return self.net(features)


@MODULES.register_module()
class DINOv3PatchDiscriminator(nn.Module):
    """Frozen DINOv3 backbone + trainable global/patch score heads.

    D(x) = global_weight * s_global + patch_weight * mean_i(s_patch_i)
    """

    def __init__(
            self,
            checkpoint_path,
            input_size=224,
            global_weight=0.25,
            patch_weight=1.0,
            freeze_backbone=True):
        super().__init__()
        self.input_size = input_size
        self.global_weight = global_weight
        self.patch_weight = patch_weight
        self.backbone = load_dinov3_vitl16_from_hf(checkpoint_path)
        self.resize = Resize((input_size, input_size), antialias=True)
        feature_dim = self.backbone.num_features
        self.global_head = _ScoreHead(feature_dim)
        self.patch_head = _ScoreHead(feature_dim)
        self.register_buffer(
            'mean',
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'std',
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def preprocess(self, images):
        """DINOv3-style resize to a square, then ImageNet normalization."""
        images = self.resize(images.float())
        return (images - self.mean) / self.std

    @staticmethod
    def _hinge_discriminator_loss(real_logits, fake_logits):
        loss_real = F.relu(1.0 - real_logits).mean()
        loss_fake = F.relu(1.0 + fake_logits).mean()
        return loss_real + loss_fake

    @staticmethod
    def _hinge_generator_loss(fake_logits):
        return (-fake_logits).mean()

    def score_images(self, images, return_parts=False):
        images = self.preprocess(images.float())
        tokens = self.backbone.forward_features(images)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1 + NUM_REGISTER_TOKENS:]
        s_global = self.global_head(cls_token)
        s_patch = self.patch_head(patch_tokens).mean(dim=1)
        logits = self.global_weight * s_global + self.patch_weight * s_patch
        if return_parts:
            return logits, s_global, s_patch
        return logits

    def forward(
            self,
            images=None,
            real_images=None,
            fake_images=None,
            gan_mode=None,
            return_parts=False):
        if gan_mode == 'discriminator':
            real_logits = self.score_images(real_images)
            fake_logits = self.score_images(fake_images)
            return self._hinge_discriminator_loss(real_logits, fake_logits)
        if gan_mode == 'generator':
            fake_logits = self.score_images(fake_images)
            return self._hinge_generator_loss(fake_logits)
        if images is None:
            raise ValueError('DINOv3PatchDiscriminator requires `images` when gan_mode is None.')
        return self.score_images(images, return_parts=return_parts)

    @staticmethod
    def build_log_vars(real_logits, fake_logits, loss_d):
        return dict(
            loss_d=float(loss_d.detach()),
            d_real=float(real_logits.detach().mean()),
            d_fake=float(fake_logits.detach().mean()),
        )

    @staticmethod
    def build_generator_log_vars(fake_logits, loss_g):
        return dict(
            loss_g_gan=float(loss_g.detach()),
            g_fake_logits=float(fake_logits.detach().mean()),
        )

    def discriminator_loss(self, real_images, fake_images):
        real_logits = self.score_images(real_images)
        fake_logits = self.score_images(fake_images)
        loss = self._hinge_discriminator_loss(real_logits, fake_logits)
        log_vars = self.build_log_vars(real_logits, fake_logits, loss)
        return loss, log_vars

    def generator_loss(self, fake_images):
        fake_logits = self.score_images(fake_images)
        loss = self._hinge_generator_loss(fake_logits)
        log_vars = self.build_generator_log_vars(fake_logits, loss)
        return loss, log_vars
