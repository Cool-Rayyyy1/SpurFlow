# Copyright (c) 2026 SpurFlow contributors
"""FLUX.2 Klein **base** 9B edit distillation wrapper.

Teacher checkpoint: ``FLUX.2-klein-base-9B`` (undistilled).
Student goal: 2-NFE Softsign alpha that beats official distilled klein-9B.

Guidance:
  - Teacher: classic CFG scale=4.0 (Flux2KleinPipeline does cond then uncond
    DiT forwards when ``guidance_scale > 1`` and ``not is_distilled``).
    ``guidance_embeds=false`` — scale is NOT single-pass embedded like Flux2-dev.
  - Student: cond-only (guidance_scale=1); no uncond forward.

Image prep:
  - Dataset ``resize_mode='flux2'`` matches Flux2KleinPipeline
    (area-cap ~1MP + multiple-of-16; NOT Kontext buckets).
"""

import torch

from mmgen.models.builder import MODELS

from .latent_diffusion_image_edit import LatentDiffusionImageEdit


@MODELS.register_module()
class LatentDiffusionFlux2KleinImageEdit(LatentDiffusionImageEdit):
    """SpurFlow wrapper for FLUX.2-klein-base-9B teacher + cond-only student."""

    def _encode_negative_prompt_embeds(self, data, prompt_embed_kwargs, bs, negative_prompt):
        if 'negative_prompt_embed_kwargs' in data:
            return data['negative_prompt_embed_kwargs']
        assert self.text_encoder is not None, \
            'Text encoder must be provided for encoding the negative prompt.'
        if 'negative_prompt_kwargs' in data:
            return self.text_encoder(**{
                k: v for k, v in data['negative_prompt_kwargs'].items()})
        return self.text_encoder(prompt=[negative_prompt] * bs)

    @staticmethod
    def _cat_cfg_prompt_embeds(neg_kwargs, pos_kwargs):
        """Concat neg/pos Flux2 embeds along batch (fixed-length Qwen3 pads)."""
        out = {
            'encoder_hidden_states': torch.cat(
                [neg_kwargs['encoder_hidden_states'],
                 pos_kwargs['encoder_hidden_states']], dim=0),
        }
        if 'txt_ids' in neg_kwargs and 'txt_ids' in pos_kwargs:
            out['txt_ids'] = torch.cat(
                [neg_kwargs['txt_ids'], pos_kwargs['txt_ids']], dim=0)
        return out

    def _prepare_train_minibatch_diffusion_args(self, data):
        # Student path: always cond-only (no CFG doubling).
        diffusion_args, diffusion_kwargs, prompt_embed_kwargs, bs, device = (
            super()._prepare_train_minibatch_diffusion_args(data))

        # Never inject Kontext-style guidance embeds / pooled projections.
        diffusion_kwargs.pop('guidance', None)
        diffusion_kwargs.pop('pooled_projections', None)
        # Guard against a mis-set student_guidance_scale>1 leaking doubled batch.
        if self.train_cfg.get('student_guidance_scale', None) not in (None, 0, 0.0, 1, 1.0):
            raise ValueError(
                'FLUX.2 Klein Softsign student is cond-only. '
                'Unset student_guidance_scale or set it to 1.0 '
                f'(got {self.train_cfg.get("student_guidance_scale")}).')
        return diffusion_args, diffusion_kwargs, prompt_embed_kwargs, bs, device

    def _prepare_train_minibatch_teacher_args(self, data, prompt_embed_kwargs, bs, device):
        # Teacher: official klein-base CFG=4.0 → doubled cond/uncond embeds.
        teacher_guidance_scale = self.train_cfg.get('teacher_guidance_scale', 4.0)
        teacher_use_guidance = (
            teacher_guidance_scale is not None
            and teacher_guidance_scale != 0.0
            and teacher_guidance_scale != 1.0)

        if teacher_use_guidance:
            negative_prompt = self.train_cfg.get('teacher_negative_prompt', '')
            negative_prompt_embed_kwargs = self._encode_negative_prompt_embeds(
                data, prompt_embed_kwargs, bs, negative_prompt)
            teacher_kwargs = self._cat_cfg_prompt_embeds(
                negative_prompt_embed_kwargs, prompt_embed_kwargs)
            teacher_kwargs.update(
                guidance_scale=teacher_guidance_scale,
                guidance_norm_rescale=self.train_cfg.get(
                    'teacher_guidance_norm_rescale', False),
                guidance_norm_rescale_patch_size=self.train_cfg.get(
                    'teacher_guidance_norm_rescale_patch_size', 2))
        else:
            teacher_kwargs = prompt_embed_kwargs.copy()

        teacher_kwargs.pop('guidance', None)
        teacher_kwargs.pop('pooled_projections', None)

        if 'source_images' in data:
            assert self.vae is not None, 'VAE must be provided for encoding source images.'
            source_latents = self._encode_images(data['source_images'])
            image_latents = self.patchify(source_latents)
            if teacher_use_guidance:
                image_latents = torch.cat([image_latents, image_latents], dim=0)
            teacher_kwargs['image_latents'] = image_latents

        return teacher_kwargs

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        # Student infer: cond-only.
        cfg_override = dict(test_cfg_override)
        cfg_override.setdefault('guidance_scale', 1.0)
        cfg_override.pop('distilled_guidance_scale', None)
        return super().val_step(data, test_cfg_override=cfg_override, **kwargs)
