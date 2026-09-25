# Copyright (c) 2026 SpurFlow contributors

import numpy as np
import torch
import torch.nn as nn

from typing import Any, Dict, Optional
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from accelerate import init_empty_weights
from peft import LoraConfig
from mmcv.cnn import constant_init
from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger

from lakonlab.runner.checkpoint import _load_checkpoint, load_full_state_dict
from ..utils import flex_freeze
from .arc_output import ArcFlowEditNewAlphaModelOutput
from .arcflux_edit_new import _ArcFluxEditNewTransformer2DModel


class _ArcFluxEditNewAlphaTransformer2DModel(_ArcFluxEditNewTransformer2DModel):
    """Per-subpixel sigmoid alpha head.

    Each DiT token predicts ``patch_size ** 2`` logits. Sigmoid maps them to
    ``(0, 1)``, and each channel controls its own latent position in the patch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj_out_alpha = nn.Linear(
            self.inner_dim, self.logweights_channels)

    def init_weights(self):
        super().init_weights()
        constant_init(self.proj_out_alpha.to_empty(device='cpu'), val=0)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.Tensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            controlnet_blocks_repeat: bool = False):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop('scale', 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            assert joint_attention_kwargs is None or joint_attention_kwargs.get('scale', None) is None

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
        image_rotary_emb = tuple([x.to(hidden_states.dtype) for x in image_rotary_emb])

        if joint_attention_kwargs is not None and 'ip_adapter_image_embeds' in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop('ip_adapter_image_embeds')
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({'ip_hidden_states': ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = self.norm_out(hidden_states, temb)

        bs, seq_len, _ = hidden_states.size()
        out_deltax = self.proj_out_deltax(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.out_channels)
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.logweights_channels).log_softmax(dim=-2)
        out_log_gammas = self.proj_out_loggamma(hidden_states).reshape(
            bs, seq_len, self.num_gammas, self.logweights_channels)
        out_alpha = torch.sigmoid(
            self.proj_out_alpha(hidden_states).reshape(
                bs, seq_len, 1, self.logweights_channels))

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        return ArcFlowEditNewAlphaModelOutput(
            deltax=out_deltax,
            logweights=out_logweights,
            loggammas=out_log_gammas,
            alpha=out_alpha)


@MODULES.register_module()
class ArcFluxEditNewAlphaTransformer2DModel(_ArcFluxEditNewAlphaTransformer2DModel):

    def __init__(
            self,
            *args,
            patch_size=2,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            pretrained_adapter=None,
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            inherit_proj_out_deltax=True,
            deltax_init='zero',
            **kwargs):
        self.inherit_proj_out_deltax = inherit_proj_out_deltax
        self.deltax_init = deltax_init
        with init_empty_weights():
            super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        assert self.patch_size * self.patch_size == self.logweights_channels

        self.init_weights(pretrained, pretrained_adapter)

        if autocast_dtype is not None:
            assert torch_dtype == 'float32'
        self.autocast_dtype = autocast_dtype

        self.use_lora = use_lora
        self.lora_target_modules = lora_target_modules
        self.lora_rank = lora_rank
        if self.use_lora:
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights='gaussian',
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
            )
            self.add_adapter(transformer_lora_config)

        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

        self.freeze = freeze
        if self.freeze:
            flex_freeze(
                self,
                exclude_keys=freeze_exclude,
                exclude_fp32=freeze_exclude_fp32,
                exclude_autocast_dtype=freeze_exclude_autocast_dtype)

        if checkpointing:
            self.enable_gradient_checkpointing()

    def init_weights(self, pretrained=None, pretrained_adapter=None):
        _ArcFluxEditNewAlphaTransformer2DModel.init_weights(self)
        if pretrained is not None:
            logger = get_root_logger()
            checkpoint = _load_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            if self.inherit_proj_out_deltax:
                if 'proj_out.weight' in state_dict and state_dict['proj_out.weight'].size(0) == self.out_channels:
                    state_dict['proj_out_deltax.weight'] = state_dict['proj_out.weight'][None].expand(
                        self.num_gaussians, -1, -1).reshape(self.num_gaussians * self.out_channels, -1)
                    del state_dict['proj_out.weight']
                if 'proj_out.bias' in state_dict and state_dict['proj_out.bias'].size(0) == self.out_channels:
                    state_dict['proj_out_deltax.bias'] = state_dict['proj_out.bias'][None].expand(
                        self.num_gaussians, -1).reshape(self.num_gaussians * self.out_channels)
                    p2 = self.patch_size * self.patch_size
                    rand_noise = torch.randn(
                        (self.num_gaussians * self.out_channels // p2),
                        dtype=state_dict['proj_out_deltax.bias'].dtype,
                        device=state_dict['proj_out_deltax.bias'].device) * 0.05
                    state_dict['proj_out_deltax.bias'] += rand_noise[:, None].expand(-1, p2).flatten()
                    del state_dict['proj_out.bias']
            else:
                state_dict.pop('proj_out.weight', None)
                state_dict.pop('proj_out.bias', None)
            state_dict.pop('proj_out_alpha.weight', None)
            state_dict.pop('proj_out_alpha.bias', None)
            if pretrained_adapter is not None:
                adapter_state_dict = _load_checkpoint(
                    pretrained_adapter, map_location='cpu', logger=logger)
                lora_state_dict = dict()
                for k, v in adapter_state_dict.items():
                    if 'lora' in k:
                        lora_state_dict[k] = v
                    else:
                        state_dict[k] = v
                load_full_state_dict(self, state_dict, logger=logger, assign=True)
                if len(lora_state_dict) > 0:
                    self.load_lora_adapter(lora_state_dict, prefix=None)
                    self.fuse_lora()
                    self.unload_lora()
            else:
                load_full_state_dict(self, state_dict, logger=logger, assign=True)

    @staticmethod
    def _prepare_latent_image_ids(height, width, device, dtype, modality=0):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 0] = modality
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels)

        return latent_image_ids.to(device=device, dtype=dtype)

    def patchify(self, latents):
        if self.patch_size > 1:
            bs, c, h, w = latents.size()
            latents = latents.reshape(
                bs, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size
            ).permute(
                0, 1, 3, 5, 2, 4
            ).reshape(
                bs, c * self.patch_size * self.patch_size, h // self.patch_size, w // self.patch_size)
        return latents

    def unpatchify(self, mp):
        if self.patch_size > 1:
            bs, k, c, h, w = mp['deltax'].size()
            mp['deltax'] = mp['deltax'].reshape(
                bs, k, c // (self.patch_size * self.patch_size), self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, k, c // (self.patch_size * self.patch_size), h * self.patch_size, w * self.patch_size)
            if 'epsilon' in mp:
                eps = mp['epsilon']
                _, c_eps, h_eps, w_eps = eps.size()
                mp['epsilon'] = eps.reshape(
                    bs, c_eps // (self.patch_size * self.patch_size),
                    self.patch_size, self.patch_size, h_eps, w_eps
                ).permute(
                    0, 1, 4, 2, 5, 3
                ).reshape(
                    bs, c_eps // (self.patch_size * self.patch_size),
                    h_eps * self.patch_size, w_eps * self.patch_size)
            alpha = mp['alpha']
            mp['alpha'] = alpha.reshape(
                bs, 1, 1, self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, 1, h * self.patch_size, w * self.patch_size)
            mp['logweights'] = mp['logweights'].reshape(
                bs, k, 1, self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, k, 1, h * self.patch_size, w * self.patch_size)
            mp['loggammas'] = mp['loggammas'].reshape(
                bs, k - 1, 1, self.patch_size, self.patch_size, h, w
            ).permute(
                0, 1, 2, 5, 3, 6, 4
            ).reshape(
                bs, k - 1, 1, h * self.patch_size, w * self.patch_size)
        return mp

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            mask: Optional[torch.Tensor] = None,
            masked_image_latents: Optional[torch.Tensor] = None,
            image_latents: Optional[torch.Tensor] = None,
            **kwargs):
        hidden_states = self.patchify(hidden_states)
        bs, c, h, w = hidden_states.size()
        dtype = self.x_embedder.weight.dtype
        device = hidden_states.device
        hidden_states = hidden_states.reshape(bs, c, h * w).permute(0, 2, 1)
        target_seq_len = hidden_states.size(1)
        img_ids = self._prepare_latent_image_ids(
            h, w, device, dtype, modality=0)
        txt_ids = img_ids.new_zeros((encoder_hidden_states.shape[-2], 3))

        if mask is not None and masked_image_latents is not None:
            hidden_states = torch.cat(
                (hidden_states.to(dtype=dtype),
                 masked_image_latents.to(dtype=dtype),
                 mask.to(dtype=dtype)), dim=-1)

        if image_latents is not None:
            image_latents = self.patchify(image_latents)
            _, c_ref, h_ref, w_ref = image_latents.size()
            if h_ref != h or w_ref != w:
                raise ValueError(
                    f'Reference latents spatial size {(h_ref, w_ref)} must match target {(h, w)}.')
            ref_hidden = image_latents.reshape(bs, c_ref, h_ref * w_ref).permute(0, 2, 1)
            ref_ids = self._prepare_latent_image_ids(
                h_ref, w_ref, device, dtype, modality=1)
            hidden_states = torch.cat(
                [hidden_states.to(dtype=dtype), ref_hidden.to(dtype=dtype)], dim=1)
            img_ids = torch.cat([img_ids, ref_ids], dim=0)

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            output = super().forward(
                hidden_states=hidden_states.to(dtype),
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                pooled_projections=pooled_projections.to(dtype),
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                **kwargs)

        alpha = output.alpha[:, :target_seq_len].permute(
            0, 2, 3, 1).reshape(
                bs, 1, self.logweights_channels, h, w)

        output_dict = dict(
            deltax=output.deltax[:, :target_seq_len].permute(0, 2, 3, 1).reshape(
                bs, self.num_gaussians, self.out_channels, h, w),
            logweights=output.logweights[:, :target_seq_len].permute(0, 2, 3, 1).reshape(
                bs, self.num_gaussians, self.logweights_channels, h, w),
            loggammas=output.loggammas[:, :target_seq_len].permute(0, 2, 3, 1).reshape(
                bs, self.num_gammas, self.logweights_channels, h, w),
            alpha=alpha,
        )
        return self.unpatchify(output_dict)
