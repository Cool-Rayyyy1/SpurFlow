# Copyright (c) 2026 SpurFlow contributors
"""Editable wrapper around diffusers ``Flux2Transformer2DModel`` (teacher).

FLUX.2-klein-base-9B: ``guidance_embeds=false`` so DiT always gets
``guidance=None``; classic CFG is done by the caller (doubled cond/uncond batch).
Packing matches ``Flux2KleinPipeline``: 4D (T,H,W,L) ids, target∥ref concat on seq.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from peft import LoraConfig
from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger

from lakonlab.runner.checkpoint import load_checkpoint, _load_checkpoint
from lakonlab.utils import materialize_meta_states
from ..utils import flex_freeze


def _resolve_lora_linear_targets(
        model: nn.Module,
        target_modules: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Map LoRA name patterns to concrete ``nn.Linear`` module names.

    Flux2 double-stream attention uses ``to_out = ModuleList(Linear, Dropout)`` while
    single-stream uses a bare ``to_out`` Linear. PEFT cannot wrap ModuleList, so a
    bare ``\"to_out\"`` pattern blows up — resolve to Linear paths only.
    """
    if not target_modules:
        return target_modules
    resolved = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name == t or name.endswith('.' + t) for t in target_modules):
            resolved.append(name)
    if not resolved:
        raise ValueError(
            f'No nn.Linear modules matched lora_target_modules={list(target_modules)}. '
            'Check Flux2 module names (use to_out.0 for double-stream, to_out for single-stream).')
    return resolved

try:
    from diffusers.models import Flux2Transformer2DModel as _Flux2Transformer2DModel
except ImportError:  # pragma: no cover
    _Flux2Transformer2DModel = None


def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, 4) with T=0, L=0 (Flux2KleinPipeline)."""
    batch_size, _, height, width = latents.shape
    t = torch.arange(1, device=latents.device)
    h = torch.arange(height, device=latents.device)
    w = torch.arange(width, device=latents.device)
    l = torch.arange(1, device=latents.device)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1).to(
        dtype=torch.float32)


def _prepare_image_ids(
        image_latents: torch.Tensor,
        scale: int = 10) -> torch.Tensor:
    """Packed ref latents (B, C, H, W) -> (B, H*W, 4) with T=scale."""
    batch_size, _, height, width = image_latents.shape
    t = torch.tensor([scale], device=image_latents.device)
    h = torch.arange(height, device=image_latents.device)
    w = torch.arange(width, device=image_latents.device)
    l = torch.arange(1, device=image_latents.device)
    ids = torch.cartesian_prod(t, h, w, l)
    return ids.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=torch.float32)


def _prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    """(B, L, D) -> (B, L, 4) text position ids."""
    batch_size, seq_len, _ = prompt_embeds.shape
    t = torch.arange(1, device=prompt_embeds.device)
    h = torch.arange(1, device=prompt_embeds.device)
    w = torch.arange(1, device=prompt_embeds.device)
    l = torch.arange(seq_len, device=prompt_embeds.device)
    ids = torch.cartesian_prod(t, h, w, l)
    return ids.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=torch.float32)


if _Flux2Transformer2DModel is None:

    @MODULES.register_module()
    class Flux2Transformer2DModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Flux2Transformer2DModel requires diffusers>=0.37 with FLUX.2 support.')

else:

    @MODULES.register_module()
    class Flux2Transformer2DModel(_Flux2Transformer2DModel):
        """Teacher / tied-backbone wrapper for FLUX.2 Klein DiT."""

        def __init__(
                self,
                *args,
                patch_size=2,
                freeze=False,
                freeze_exclude=[],
                pretrained=None,
                pretrained_lora=None,
                pretrained_lora_scale=1.0,
                torch_dtype='float32',
                freeze_exclude_fp32=True,
                freeze_exclude_autocast_dtype='float32',
                checkpointing=True,
                use_lora=False,
                lora_target_modules=None,
                lora_rank=16,
                lora_dropout=0.0,
                **kwargs):
            # Official Klein config uses transformer patch_size=1; SpurFlow keeps
            # an outer patch_size=2 for 32->128 channel packing before the DiT.
            kwargs.setdefault('patch_size', 1)
            kwargs.setdefault('guidance_embeds', False)
            with init_empty_weights():
                super().__init__(*args, **kwargs)
            self.patch_size = patch_size

            self.init_weights(pretrained, pretrained_lora, pretrained_lora_scale)
            materialize_meta_states(self, device='cpu')

            self.use_lora = use_lora
            self.lora_rank = lora_rank
            if self.use_lora:
                resolved_targets = _resolve_lora_linear_targets(self, lora_target_modules)
                self.lora_target_modules = resolved_targets
                transformer_lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank,
                    init_lora_weights='gaussian',
                    target_modules=resolved_targets,
                    lora_dropout=lora_dropout,
                )
                self.add_adapter(transformer_lora_config)
            else:
                self.lora_target_modules = lora_target_modules

            if torch_dtype is not None:
                self.to(getattr(torch, torch_dtype))

            self.freeze = freeze
            if self.freeze:
                flex_freeze(
                    self,
                    exclude_keys=freeze_exclude,
                    exclude_fp32=freeze_exclude_fp32,
                    exclude_autocast_dtype=freeze_exclude_autocast_dtype)

            if checkpointing and hasattr(self, 'enable_gradient_checkpointing'):
                self.enable_gradient_checkpointing()

        def init_weights(self, pretrained=None, pretrained_lora=None, pretrained_lora_scale=1.0):
            if pretrained is not None:
                logger = get_root_logger()
                load_checkpoint(
                    self, pretrained, map_location='cpu', strict=False, logger=logger, assign=True)
                if pretrained_lora is not None:
                    if not isinstance(pretrained_lora, (list, tuple)):
                        pretrained_lora = [pretrained_lora]
                    if not isinstance(pretrained_lora_scale, (list, tuple)):
                        pretrained_lora_scale = [pretrained_lora_scale]
                    for lora_path, lora_scale in zip(pretrained_lora, pretrained_lora_scale):
                        lora_state_dict = _load_checkpoint(
                            lora_path, map_location='cpu', logger=logger)
                        self.load_lora_adapter(lora_state_dict)
                        if hasattr(self, 'fuse_lora'):
                            self.fuse_lora(lora_scale=lora_scale)
                            self.unload_lora()

        def patchify(self, latents):
            if self.patch_size > 1:
                bs, c, h, w = latents.size()
                latents = latents.reshape(
                    bs, c, h // self.patch_size, self.patch_size,
                    w // self.patch_size, self.patch_size
                ).permute(0, 1, 3, 5, 2, 4).reshape(
                    bs, c * self.patch_size * self.patch_size,
                    h // self.patch_size, w // self.patch_size)
            return latents

        def unpatchify(self, latents):
            if self.patch_size > 1:
                bs, c, h, w = latents.size()
                latents = latents.reshape(
                    bs, c // (self.patch_size * self.patch_size),
                    self.patch_size, self.patch_size, h, w
                ).permute(0, 1, 4, 2, 5, 3).reshape(
                    bs, c // (self.patch_size * self.patch_size),
                    h * self.patch_size, w * self.patch_size)
            return latents

        def _pack_tokens(self, latents):
            bs, c, h, w = latents.shape
            return latents.reshape(bs, c, h * w).permute(0, 2, 1), h, w

        def forward(
                self,
                hidden_states: torch.Tensor,
                timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor = None,
                image_latents: Optional[torch.Tensor] = None,
                txt_ids: Optional[torch.Tensor] = None,
                guidance: Optional[torch.Tensor] = None,
                joint_attention_kwargs=None,
                **kwargs):
            # Base Klein: no guidance embeds; CFG is applied by doubling batch outside.
            kwargs.pop('pooled_projections', None)
            kwargs.pop('guidance_scale', None)
            del guidance

            hidden_states = self.patchify(hidden_states)
            bs, c, h, w = hidden_states.size()
            device = hidden_states.device
            # Match backbone weight dtype (bf16); latents arrive as float32.
            dtype = self.x_embedder.weight.dtype
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

            # Official Flux2KleinPipeline: scheduler_t in [0, 1000] is passed as
            # t/1000, then Flux2Transformer2DModel.forward does timestep * 1000.
            # SpurFlow uses num_timesteps=1, so t is already sigma in [0, 1]
            # (same as Kontext). Pass t as-is; do NOT divide by 1000 again.
            output = super().forward(
                hidden_states=tokens.to(dtype=dtype),
                encoder_hidden_states=encoder_hidden_states.to(dtype=dtype),
                timestep=timestep.to(dtype=dtype),
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
                **kwargs)[0]

            # Flux2 keeps ref tokens in the image stream; keep target only.
            output = output[:, :target_seq_len].permute(0, 2, 1).reshape(
                bs, self.out_channels, h, w)
            return self.unpatchify(output)
