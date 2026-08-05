# Copyright (c) 2026 EditFlow contributors
"""FLUX.2 Klein Softsign-01 alpha student DiT.

Alpha design (matches Qwen softsign / train_*_alpha_softsign_*):
    alpha = 0.5 * (raw / (1 + abs(raw)) + 1)   # Softsign-01 in (0, 1), 0 -> 0.5
    student_u = path_epsilon - alpha * x_ref - pred_delta

Base Klein: DiT still gets ``guidance=None`` (no guidance embeds); classic CFG
is applied outside via doubled cond/uncond batches (guidance_scale=4.0).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from mmcv.cnn import constant_init
from mmgen.models.builder import MODULES

from lakonlab.utils import materialize_meta_states

from ..diffusers.flux2 import (
    Flux2Transformer2DModel,
    _prepare_image_ids,
    _prepare_latent_ids,
    _prepare_text_ids,
)
from ..utils import flex_freeze
from .arc_output import ArcFlowEditNewAlphaModelOutput


@MODULES.register_module()
class ArcFlux2EditAlphaSoftsign01Transformer2DModel(Flux2Transformer2DModel):
    """Softsign-01 alpha student for FLUX.2 Klein."""

    def __init__(
            self,
            *args,
            num_gaussians: int = 16,
            logweights_channels: int = 4,
            inherit_proj_out_deltax: bool = False,
            deltax_init: str = 'kaiming',
            freeze: bool = False,
            freeze_exclude=(),
            freeze_exclude_fp32: bool = True,
            freeze_exclude_autocast_dtype: str = 'float32',
            **kwargs):
        self.num_gaussians = int(num_gaussians)
        self.logweights_channels = int(logweights_channels)
        self.num_gammas = self.num_gaussians - 1
        self.inherit_proj_out_deltax = bool(inherit_proj_out_deltax)
        self.deltax_init = deltax_init
        kwargs.pop('num_gaussians', None)
        kwargs.pop('logweights_channels', None)
        kwargs.pop('num_gammas', None)
        kwargs.pop('inherit_proj_out_deltax', None)
        kwargs.pop('deltax_init', None)
        kwargs['guidance_embeds'] = False

        # Build backbone + LoRA first; freeze only after ArcFlow heads exist.
        super().__init__(
            *args,
            freeze=False,
            freeze_exclude=[],
            **kwargs)

        assert self.patch_size * self.patch_size == self.logweights_channels, (
            f'patch_size**2 ({self.patch_size ** 2}) must equal '
            f'logweights_channels ({self.logweights_channels}).')

        inner = self.inner_dim
        out_channels = self.out_channels  # 128 after outer patchify
        self.proj_out_deltax = nn.Linear(inner, self.num_gaussians * out_channels)
        self.proj_out_logweights = nn.Linear(
            inner, self.num_gaussians * self.logweights_channels)
        self.proj_out_loggamma = nn.Linear(
            inner, self.num_gammas * self.logweights_channels)
        self.proj_out_alpha = nn.Linear(inner, self.logweights_channels)

        # EMA build wraps the whole module in init_empty_weights(); new heads are
        # meta and cannot .to() — materialize on CPU then init, matching ArcQwen.
        for head in (
                self.proj_out_deltax, self.proj_out_logweights,
                self.proj_out_loggamma, self.proj_out_alpha):
            materialize_meta_states(head, device='cpu')
        self._init_arcflow_heads()

        dtype = device = None
        for p in self.parameters():
            if not p.is_meta:
                dtype = p.dtype
                device = p.device
                break
        if dtype is not None:
            self.proj_out_deltax.to(device=device, dtype=dtype)
            self.proj_out_logweights.to(device=device, dtype=dtype)
            self.proj_out_loggamma.to(device=device, dtype=dtype)
            self.proj_out_alpha.to(device=device, dtype=dtype)

        self.freeze = freeze
        if self.freeze:
            flex_freeze(
                self,
                exclude_keys=list(freeze_exclude),
                exclude_fp32=freeze_exclude_fp32,
                exclude_autocast_dtype=freeze_exclude_autocast_dtype)

    def _init_arcflow_heads(self):
        if self.deltax_init == 'kaiming':
            nn.init.kaiming_uniform_(self.proj_out_deltax.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(self.inner_dim)
            nn.init.uniform_(self.proj_out_deltax.bias, -bound, bound)
        elif self.deltax_init == 'zero':
            constant_init(self.proj_out_deltax, val=0)
        else:
            raise ValueError(
                f'Invalid deltax_init={self.deltax_init!r}. Supported: "zero", "kaiming".')
        constant_init(self.proj_out_logweights, val=0)
        constant_init(self.proj_out_loggamma, val=0)
        target_gammas = torch.logspace(
            math.log10(0.2), math.log10(4.0), self.num_gammas, base=10)
        target_log_gammas = torch.log(target_gammas)
        if self.logweights_channels > 1:
            target_log_gammas = target_log_gammas.unsqueeze(1).repeat(
                1, self.logweights_channels).flatten()
        with torch.no_grad():
            self.proj_out_loggamma.bias.copy_(target_log_gammas.to(
                dtype=self.proj_out_loggamma.bias.dtype,
                device=self.proj_out_loggamma.bias.device))
        # Softsign-01(0) = 0.5
        constant_init(self.proj_out_alpha, val=0)

    def _activate_alpha(self, logits: torch.Tensor) -> torch.Tensor:
        """Softsign-01: maps R -> (0, 1) with zero-logit -> 0.5."""
        return 0.5 * (logits / (1.0 + logits.abs()) + 1.0)

    def unpatchify(self, mp):
        if self.patch_size > 1:
            bs, k, c, h, w = mp['deltax'].size()
            mp['deltax'] = mp['deltax'].reshape(
                bs, k, c // (self.patch_size * self.patch_size),
                self.patch_size, self.patch_size, h, w
            ).permute(0, 1, 2, 5, 3, 6, 4).reshape(
                bs, k, c // (self.patch_size * self.patch_size),
                h * self.patch_size, w * self.patch_size)
            alpha = mp['alpha']
            mp['alpha'] = alpha.reshape(
                bs, 1, 1, self.patch_size, self.patch_size, h, w
            ).permute(0, 1, 2, 5, 3, 6, 4).reshape(
                bs, 1, h * self.patch_size, w * self.patch_size)
            mp['logweights'] = mp['logweights'].reshape(
                bs, k, 1, self.patch_size, self.patch_size, h, w
            ).permute(0, 1, 2, 5, 3, 6, 4).reshape(
                bs, k, 1, h * self.patch_size, w * self.patch_size)
            mp['loggammas'] = mp['loggammas'].reshape(
                bs, k - 1, 1, self.patch_size, self.patch_size, h, w
            ).permute(0, 1, 2, 5, 3, 6, 4).reshape(
                bs, k - 1, 1, h * self.patch_size, w * self.patch_size)
        return mp

    def _model_dtype(self):
        """Backbone compute dtype (bf16); heads may stay fp32 via flex_freeze."""
        return self.x_embedder.weight.dtype

    def _forward_features(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            img_ids: torch.Tensor,
            txt_ids: torch.Tensor,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None):
        """Flux2 backbone through ``norm_out`` (skip ``proj_out``)."""
        num_txt_tokens = encoder_hidden_states.shape[1]
        dtype = self._model_dtype()
        hidden_states = hidden_states.to(dtype=dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=dtype)

        # Timesteps sinusoidal emb is often float32; cast to weight dtype before Linear.
        timestep = timestep.to(dtype=dtype) * 1000
        temb = self.time_guidance_embed(timestep, None)
        if temb.dtype != dtype:
            temb = temb.to(dtype=dtype)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        return self.norm_out(hidden_states, temb)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            image_latents: Optional[torch.Tensor] = None,
            txt_ids: Optional[torch.Tensor] = None,
            guidance: Optional[torch.Tensor] = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs):
        kwargs.pop('pooled_projections', None)
        kwargs.pop('guidance_scale', None)
        del guidance  # no guidance embeds; CFG is outside via doubled batch

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop('scale', 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        hidden_states = self.patchify(hidden_states)
        bs, c, h, w = hidden_states.size()
        device = hidden_states.device
        # Latents from VAE path are float32; backbone weights are bf16.
        dtype = self._model_dtype()
        tokens, _, _ = self._pack_tokens(hidden_states)
        target_seq_len = tokens.size(1)
        img_ids = _prepare_latent_ids(hidden_states).to(device=device)

        if image_latents is not None:
            image_latents = self.patchify(image_latents)
            _, c_ref, h_ref, w_ref = image_latents.size()
            if h_ref != h or w_ref != w:
                raise ValueError(
                    f'Reference latents spatial size {(h_ref, w_ref)} '
                    f'must match target {(h, w)}.')
            ref_tokens, _, _ = self._pack_tokens(image_latents)
            ref_ids = _prepare_image_ids(image_latents).to(device=device)
            tokens = torch.cat(
                [tokens.to(dtype=dtype), ref_tokens.to(dtype=dtype)], dim=1)
            img_ids = torch.cat([img_ids, ref_ids], dim=1)

        if txt_ids is None:
            txt_ids = _prepare_text_ids(encoder_hidden_states).to(device=device)
        else:
            txt_ids = txt_ids.to(device=device)

        features = self._forward_features(
            hidden_states=tokens.to(dtype=dtype),
            encoder_hidden_states=encoder_hidden_states.to(dtype=dtype),
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        features = features[:, :target_seq_len]
        # ArcFlow heads are freeze-excluded fp32; cast before head Linear.
        features = features.float()
        seq_len = features.size(1)

        out_deltax = self.proj_out_deltax(features).reshape(
            bs, seq_len, self.num_gaussians, self.out_channels)
        out_logweights = self.proj_out_logweights(features).reshape(
            bs, seq_len, self.num_gaussians, self.logweights_channels
        ).log_softmax(dim=-2)
        out_log_gammas = self.proj_out_loggamma(features).reshape(
            bs, seq_len, self.num_gammas, self.logweights_channels)
        out_alpha = self._activate_alpha(
            self.proj_out_alpha(features).reshape(
                bs, seq_len, 1, self.logweights_channels))

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        alpha = out_alpha.permute(0, 2, 3, 1).reshape(
            bs, 1, self.logweights_channels, h, w)
        output_dict = dict(
            deltax=out_deltax.permute(0, 2, 3, 1).reshape(
                bs, self.num_gaussians, self.out_channels, h, w),
            logweights=out_logweights.permute(0, 2, 3, 1).reshape(
                bs, self.num_gaussians, self.logweights_channels, h, w),
            loggammas=out_log_gammas.permute(0, 2, 3, 1).reshape(
                bs, self.num_gammas, self.logweights_channels, h, w),
            alpha=alpha,
        )
        return self.unpatchify(output_dict)
