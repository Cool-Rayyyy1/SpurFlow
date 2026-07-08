# Copyright (c) 2026 EditFlow contributors

import math

import numpy as np
import torch
import torch.nn as nn

from typing import Any, Dict, Optional, Tuple
from accelerate import init_empty_weights
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from peft import LoraConfig
from mmcv.cnn import constant_init
from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger

from lakonlab.runner.checkpoint import _load_checkpoint, load_full_state_dict
from ..utils import flex_freeze
from .arc_output import ArcFlowEditNewModelOutput, ArcFlowEditNewEpsModelOutput
from .arcflux import _ArcFluxTransformer2DModel


class EditOutputHeadBundle(nn.Module):
    """Per-stage output heads used by dual-stage LoRA students."""

    def __init__(
            self,
            inner_dim,
            num_gaussians,
            num_gammas,
            out_channels,
            logweights_channels,
            ada_norm_cls):
        super().__init__()
        self.norm_out = ada_norm_cls(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_deltax = nn.Linear(
            inner_dim, num_gaussians * out_channels)
        self.proj_out_logweights = nn.Linear(
            inner_dim, num_gaussians * logweights_channels)
        self.proj_out_loggamma = nn.Linear(
            inner_dim, num_gammas * logweights_channels)


class _ArcFluxEditNewTransformer2DModel(_ArcFluxTransformer2DModel):

    def __init__(self, *args, **kwargs):
        super(FluxTransformer2DModel, self).__init__()

        num_gaussians = kwargs.get('num_gaussians', 16)
        logweights_channels = kwargs.get('logweights_channels', 1)
        in_channels = kwargs.get('in_channels', 64)
        out_channels = kwargs.get('out_channels', None)
        num_layers = kwargs.get('num_layers', 19)
        num_single_layers = kwargs.get('num_single_layers', 38)
        attention_head_dim = kwargs.get('attention_head_dim', 128)
        num_attention_heads = kwargs.get('num_attention_heads', 24)
        joint_attention_dim = kwargs.get('joint_attention_dim', 4096)
        pooled_projection_dim = kwargs.get('pooled_projection_dim', 768)
        guidance_embeds = kwargs.get('guidance_embeds', False)
        axes_dims_rope = kwargs.get('axes_dims_rope', (16, 56, 56))

        self.num_gaussians = num_gaussians
        self.num_gammas = num_gaussians - 1
        self.logweights_channels = logweights_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        from diffusers.models.transformers.transformer_flux import (
            FluxPosEmbed, FluxTransformerBlock, FluxSingleTransformerBlock)
        from diffusers.models.embeddings import (
            CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings)
        from diffusers.models.normalization import AdaLayerNormContinuous

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds
            else CombinedTimestepTextProjEmbeddings)
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim)

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(num_layers)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(num_single_layers)
        ])

        self.dual_stage_heads = kwargs.pop('dual_stage_heads', False)
        if self.dual_stage_heads:
            self.head_stages = nn.ModuleDict({
                self.LORA_STAGE_STEP1: self._make_head_bundle(AdaLayerNormContinuous),
                self.LORA_STAGE_STEP2: self._make_head_bundle(AdaLayerNormContinuous),
            })
            self._active_head_stage = self.LORA_STAGE_STEP1
        else:
            self._install_shared_heads(AdaLayerNormContinuous)

        self.predict_path_epsilon = kwargs.pop('predict_path_epsilon', False)
        if self.predict_path_epsilon:
            self.proj_out_epsilon = nn.Linear(self.inner_dim, self.out_channels)

        self.gradient_checkpointing = False

    LORA_STAGE_STEP1 = 'step1'
    LORA_STAGE_STEP2 = 'step2'

    def _make_head_bundle(self, ada_norm_cls):
        return EditOutputHeadBundle(
            self.inner_dim,
            self.num_gaussians,
            self.num_gammas,
            self.out_channels,
            self.logweights_channels,
            ada_norm_cls,
        )

    def _install_shared_heads(self, ada_norm_cls):
        self.norm_out = ada_norm_cls(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_deltax = nn.Linear(self.inner_dim, self.num_gaussians * self.out_channels)
        self.proj_out_logweights = nn.Linear(
            self.inner_dim, self.num_gaussians * self.logweights_channels)
        self.proj_out_loggamma = nn.Linear(
            self.inner_dim, self.num_gammas * self.logweights_channels)

    def _active_heads(self):
        if getattr(self, 'dual_stage_heads', False):
            return self.head_stages[self._active_head_stage]
        return self

    def _init_proj_out_deltax_layer(self, layer):
        deltax_init = getattr(self, 'deltax_init', 'zero')
        layer = layer.to_empty(device='cpu')
        if deltax_init == 'kaiming':
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(self.inner_dim)
            nn.init.uniform_(layer.bias, -bound, bound)
        elif deltax_init == 'zero':
            constant_init(layer, val=0)
            rand_noise = torch.randn(
                (self.num_gaussians * self.out_channels // self.logweights_channels)) * 0.1
            layer.bias.data.copy_(
                rand_noise[:, None].expand(-1, self.logweights_channels).flatten())
        else:
            raise ValueError(
                f'Invalid deltax_init={deltax_init!r}. Supported: "zero", "kaiming".')

    def _init_head_bundle(self, bundle):
        bundle.norm_out.to_empty(device='cpu')
        self._init_proj_out_deltax_layer(bundle.proj_out_deltax)
        constant_init(bundle.proj_out_logweights.to_empty(device='cpu'), val=0)
        constant_init(bundle.proj_out_loggamma.to_empty(device='cpu'), val=0)
        min_gamma = 0.2
        max_gamma = 4.0
        target_gammas = torch.logspace(
            math.log10(min_gamma), math.log10(max_gamma), self.num_gammas, base=10)
        target_log_gammas = torch.log(target_gammas)
        if self.logweights_channels > 1:
            target_log_gammas = target_log_gammas.unsqueeze(1).repeat(
                1, self.logweights_channels).flatten()
        bundle.proj_out_loggamma.bias.data.copy_(target_log_gammas)

    def _copy_head_bundle(self, src_stage: str, dst_stage: str) -> None:
        src = self.head_stages[src_stage]
        dst = self.head_stages[dst_stage]
        dst.load_state_dict(src.state_dict(), assign=True)

    def _init_proj_out_deltax(self):
        if getattr(self, 'dual_stage_heads', False):
            self._init_head_bundle(self.head_stages[self.LORA_STAGE_STEP1])
            self._copy_head_bundle(self.LORA_STAGE_STEP1, self.LORA_STAGE_STEP2)
            return
        self._init_proj_out_deltax_layer(self.proj_out_deltax)

    def init_weights(self):
        if getattr(self, 'dual_stage_heads', False):
            self._init_head_bundle(self.head_stages[self.LORA_STAGE_STEP1])
            self._init_head_bundle(self.head_stages[self.LORA_STAGE_STEP2])
            self._copy_head_bundle(self.LORA_STAGE_STEP1, self.LORA_STAGE_STEP2)
        else:
            self._init_proj_out_deltax()
            constant_init(self.proj_out_logweights.to_empty(device='cpu'), val=0)
            constant_init(self.proj_out_loggamma.to_empty(device='cpu'), val=0)
            min_gamma = 0.2
            max_gamma = 4.0
            target_gammas = torch.logspace(
                math.log10(min_gamma), math.log10(max_gamma), self.num_gammas, base=10)
            target_log_gammas = torch.log(target_gammas)
            if self.logweights_channels > 1:
                target_log_gammas = target_log_gammas.unsqueeze(1).repeat(
                    1, self.logweights_channels).flatten()
            self.proj_out_loggamma.bias.data.copy_(target_log_gammas)
        if getattr(self, 'predict_path_epsilon', False):
            constant_init(self.proj_out_epsilon.to_empty(device='cpu'), val=0)

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

        embed_dtype = self.x_embedder.weight.dtype
        hidden_states = self.x_embedder(hidden_states.to(embed_dtype))

        timestep = timestep.to(embed_dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(embed_dtype) * 1000
        if pooled_projections is not None:
            pooled_projections = pooled_projections.to(embed_dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(embed_dtype)

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
        image_rotary_emb = tuple([x.to(embed_dtype) for x in image_rotary_emb])

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

        hidden_states = self._active_heads().norm_out(hidden_states, temb)

        bs, seq_len, _ = hidden_states.size()
        heads = self._active_heads()
        out_deltax = heads.proj_out_deltax(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.out_channels)
        out_logweights = heads.proj_out_logweights(hidden_states).reshape(
            bs, seq_len, self.num_gaussians, self.logweights_channels).log_softmax(dim=-2)
        out_log_gammas = heads.proj_out_loggamma(hidden_states).reshape(
            bs, seq_len, self.num_gammas, self.logweights_channels)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        output_kwargs = dict(
            deltax=out_deltax,
            logweights=out_logweights,
            loggammas=out_log_gammas)
        if getattr(self, 'predict_path_epsilon', False):
            out_epsilon = self.proj_out_epsilon(hidden_states).reshape(
                bs, seq_len, self.out_channels)
            output_kwargs['epsilon'] = out_epsilon
            return ArcFlowEditNewEpsModelOutput(**output_kwargs)
        return ArcFlowEditNewModelOutput(**output_kwargs)


@MODULES.register_module()
class ArcFluxEditNewTransformer2DModel(_ArcFluxEditNewTransformer2DModel):

    LORA_STAGE_STEP1 = 'step1'
    LORA_STAGE_STEP2 = 'step2'

    def _set_lora_stage_active_only(self, stage: str) -> None:
        """Switch active dual-stage adapter without toggling ``requires_grad``.

        PEFT ``set_adapter`` flips grad flags on LoRA modules. Under FSDP those
        parameters are not leaf tensors, so runtime adapter switching must only
        update ``_active_adapter`` on each tuner layer.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        for module in self.modules():
            if isinstance(module, BaseTunerLayer):
                module._active_adapter = [stage]

    def _enable_all_dual_lora_grads(self) -> None:
        """Keep both step adapters trainable before FSDP wraps the model."""
        from peft.tuners.tuners_utils import BaseTunerLayer

        stages = {self.LORA_STAGE_STEP1, self.LORA_STAGE_STEP2}
        for module in self.modules():
            if not isinstance(module, BaseTunerLayer):
                continue
            for adapter_name in stages:
                if adapter_name in module.lora_A:
                    module.lora_A[adapter_name].requires_grad_(True)
                if adapter_name in module.lora_B:
                    module.lora_B[adapter_name].requires_grad_(True)

    def set_lora_stage(self, stage: str) -> None:
        """Activate one dual-stage LoRA adapter and matching output heads."""
        if not getattr(self, 'dual_stage_lora', False):
            return
        if stage not in (self.LORA_STAGE_STEP1, self.LORA_STAGE_STEP2):
            raise ValueError(
                f'Invalid lora stage {stage!r}; expected '
                f'{self.LORA_STAGE_STEP1!r} or {self.LORA_STAGE_STEP2!r}.')
        self._set_lora_stage_active_only(stage)
        if getattr(self, 'dual_stage_heads', False):
            self._active_head_stage = stage

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
            dual_stage_lora=False,
            lora_target_modules=None,
            lora_rank=16,
            lora_dropout=0.0,
            inherit_proj_out_deltax=True,
            deltax_init='zero',
            predict_path_epsilon=False,
            **kwargs):
        self.inherit_proj_out_deltax = inherit_proj_out_deltax
        self.deltax_init = deltax_init
        self.predict_path_epsilon = predict_path_epsilon
        self.dual_stage_lora = bool(dual_stage_lora)
        with init_empty_weights():
            super().__init__(
                *args,
                predict_path_epsilon=predict_path_epsilon,
                dual_stage_heads=self.dual_stage_lora,
                **kwargs)
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
            if self.dual_stage_lora:
                for adapter_name in (self.LORA_STAGE_STEP1, self.LORA_STAGE_STEP2):
                    self.add_adapter(transformer_lora_config, adapter_name=adapter_name)
                self._enable_all_dual_lora_grads()
                self._set_lora_stage_active_only(self.LORA_STAGE_STEP1)
            else:
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

    def _remap_state_dict_for_dual_heads(self, state_dict):
        if not getattr(self, 'dual_stage_heads', False):
            return state_dict
        head_names = (
            'norm_out',
            'proj_out_deltax',
            'proj_out_logweights',
            'proj_out_loggamma',
        )
        remapped = dict(state_dict)
        for head_name in head_names:
            head_keys = [
                key for key in state_dict
                if key == head_name or key.startswith(f'{head_name}.')
            ]
            if not head_keys:
                continue
            for stage in (self.LORA_STAGE_STEP1, self.LORA_STAGE_STEP2):
                for key in head_keys:
                    suffix = key[len(head_name):]
                    remapped[f'head_stages.{stage}.{head_name}{suffix}'] = state_dict[key]
            for key in head_keys:
                remapped.pop(key, None)
        return remapped

    def init_weights(self, pretrained=None, pretrained_adapter=None):
        super().init_weights()
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
            state_dict = self._remap_state_dict_for_dual_heads(state_dict)
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

        output['deltax'] = output['deltax'][:, :target_seq_len].permute(0, 2, 3, 1).reshape(
            bs, self.num_gaussians, self.out_channels, h, w)
        output['logweights'] = output['logweights'][:, :target_seq_len].permute(0, 2, 3, 1).reshape(
            bs, self.num_gaussians, self.logweights_channels, h, w)
        output['loggammas'] = output['loggammas'][:, :target_seq_len].permute(0, 2, 3, 1).reshape(
            bs, self.num_gammas, self.logweights_channels, h, w)
        if 'epsilon' in output:
            output['epsilon'] = output['epsilon'][:, :target_seq_len].permute(0, 2, 1).reshape(
                bs, self.out_channels, h, w)
        return self.unpatchify(output)


@MODULES.register_module()
class ArcFluxEditNewEpsTransformer2DModel(ArcFluxEditNewTransformer2DModel):
    """ArcFluxEditNew with a 4th head that predicts teacher-inverted path epsilon."""

    def __init__(self, *args, **kwargs):
        kwargs['predict_path_epsilon'] = True
        super().__init__(*args, **kwargs)
