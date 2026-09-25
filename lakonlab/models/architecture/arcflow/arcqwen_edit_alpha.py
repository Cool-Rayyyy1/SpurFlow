# Copyright (c) 2026 SpurFlow contributors

from math import prod

import torch
import torch.nn as nn

from typing import Any, Dict, List, Optional, Tuple
from accelerate import init_empty_weights
from diffusers.models.transformers.transformer_qwenimage import (
    compute_text_seq_len_from_mask)
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from peft import LoraConfig
from mmcv.cnn import constant_init
from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger

from lakonlab.runner.checkpoint import _load_checkpoint, load_full_state_dict
from lakonlab.utils import materialize_meta_states
from ..utils import flex_freeze
from .arc_output import ArcFlowEditNewAlphaModelOutput
from .arcqwen_edit import _ArcQwenEditImageTransformer2DModel


class _ArcQwenEditAlphaImageTransformer2DModel(_ArcQwenEditImageTransformer2DModel):
    """Qwen edit student with per-subpixel sigmoid alpha head.

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

    def _activate_alpha(self, logits: torch.Tensor) -> torch.Tensor:
        """Map alpha logits to (0, 1); subclasses may change the mapping."""
        return torch.sigmoid(logits)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_shapes: Optional[List[List[Tuple[int, int, int]]]] = None,
            txt_seq_lens: Optional[List[int]] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop('scale', 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            assert attention_kwargs is None or attention_kwargs.get('scale', None) is None

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]])
                 for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask)

        temb = self.time_text_embed(timestep, hidden_states)
        image_rotary_emb = self.pos_embed(
            img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        block_attention_kwargs = (
            attention_kwargs.copy() if attention_kwargs is not None else {})
        if encoder_hidden_states_mask is not None:
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones(
                (batch_size, image_seq_len),
                dtype=torch.bool,
                device=hidden_states.device)
            joint_attention_mask = torch.cat(
                [encoder_hidden_states_mask, image_mask], dim=1)
            joint_attention_mask = joint_attention_mask[:, None, None, :]
            block_attention_kwargs['attention_mask'] = joint_attention_mask

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None,
                    temb,
                    image_rotary_emb,
                    block_attention_kwargs,
                    modulate_index,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                    modulate_index=modulate_index,
                )

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        hidden_states = self.norm_out(hidden_states, temb)
        bs, seq_len, _ = hidden_states.size()
        out_deltax = self.proj_out_deltax(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.out_channels)
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.logweights_channels).log_softmax(dim=-2)
        out_log_gammas = self.proj_out_loggamma(hidden_states).reshape(
            bs, seq_len, self.num_gammas, self.logweights_channels)
        out_alpha = self._activate_alpha(
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
class ArcQwenEditAlphaImageTransformer2DModel(_ArcQwenEditAlphaImageTransformer2DModel):

    def __init__(
            self,
            *args,
            patch_size=2,
            freeze=False,
            freeze_exclude=(),
            pretrained=None,
            pretrained_adapter=None,
            inherit_proj_out_deltax=False,
            deltax_init='kaiming',
            torch_dtype='float32',
            autocast_dtype=None,
            freeze_exclude_fp32=True,
            freeze_exclude_autocast_dtype='float32',
            checkpointing=True,
            use_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            **kwargs):
        self.inherit_proj_out_deltax = inherit_proj_out_deltax
        self.deltax_init = deltax_init
        with init_empty_weights():
            super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        assert self.patch_size * self.patch_size == self.logweights_channels

        self.init_weights(pretrained, pretrained_adapter)
        materialize_meta_states(self, device='cpu')

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
        _ArcQwenEditAlphaImageTransformer2DModel.init_weights(self)
        if pretrained is None:
            return

        logger = get_root_logger()
        checkpoint = _load_checkpoint(pretrained, map_location='cpu', logger=logger)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

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
        # Keep zero-logit alpha init (sigmoid -> 0.5).
        state_dict.pop('proj_out_alpha.weight', None)
        state_dict.pop('proj_out_alpha.bias', None)

        if pretrained_adapter is not None:
            adapter_state_dict = _load_checkpoint(
                pretrained_adapter, map_location='cpu', logger=logger)
            lora_state_dict = dict()
            for key, value in adapter_state_dict.items():
                if 'lora' in key:
                    lora_state_dict[key] = value
                else:
                    state_dict[key] = value
            load_full_state_dict(self, state_dict, logger=logger, assign=True)
            if len(lora_state_dict) > 0:
                self.load_lora_adapter(lora_state_dict, prefix=None)
                self.fuse_lora()
                self.unload_lora()
        else:
            load_full_state_dict(self, state_dict, logger=logger, assign=True)

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

    def _prepare_edit_sequence(
            self,
            hidden_states: torch.Tensor,
            image_latents: Optional[torch.Tensor] = None):
        hidden_states = self.patchify(hidden_states)
        bs, c, h, w = hidden_states.size()
        target_seq_len = h * w
        hidden_states = hidden_states.reshape(bs, c, target_seq_len).permute(0, 2, 1)
        img_shapes = [[(1, h, w)]]

        if image_latents is not None:
            ref = self.patchify(image_latents)
            _, c_ref, h_ref, w_ref = ref.size()
            if h_ref != h or w_ref != w:
                raise ValueError(
                    f'Reference latents spatial size {(h_ref, w_ref)} must match target {(h, w)}.')
            ref_hidden = ref.reshape(bs, c_ref, h_ref * w_ref).permute(0, 2, 1)
            hidden_states = torch.cat([hidden_states, ref_hidden], dim=1)
            img_shapes = [[(1, h, w), (1, h_ref, w_ref)]]

        return hidden_states, img_shapes, target_seq_len, h, w

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            image_latents: Optional[torch.Tensor] = None,
            **kwargs):
        hidden_states, img_shapes, target_seq_len, h, w = self._prepare_edit_sequence(
            hidden_states, image_latents=image_latents)
        bs = hidden_states.size(0)

        if self.autocast_dtype is not None:
            dtype = getattr(torch, self.autocast_dtype)
        else:
            dtype = self.img_in.weight.dtype

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=dtype if self.autocast_dtype is not None else None):
            output = super().forward(
                hidden_states=hidden_states.to(dtype),
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                **kwargs)

        alpha = output.alpha[:, :target_seq_len].permute(
            0, 2, 3, 1).reshape(
                bs, 1, self.logweights_channels, h, w)
        output_dict = {
            'deltax': output.deltax[:, :target_seq_len].permute(0, 2, 3, 1).reshape(
                bs, self.num_gaussians, self.out_channels, h, w),
            'logweights': output.logweights[:, :target_seq_len].permute(0, 2, 3, 1).reshape(
                bs, self.num_gaussians, self.logweights_channels, h, w),
            'loggammas': output.loggammas[:, :target_seq_len].permute(0, 2, 3, 1).reshape(
                bs, self.num_gammas, self.logweights_channels, h, w),
            'alpha': alpha,
        }
        return self.unpatchify(output_dict)
