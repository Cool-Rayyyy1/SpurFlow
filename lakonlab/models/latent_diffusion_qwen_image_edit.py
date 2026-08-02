# Copyright (c) 2026 EditFlow contributors

import torch

from mmgen.models.builder import MODELS

from .latent_diffusion_image_edit import LatentDiffusionImageEdit


@MODELS.register_module()
class LatentDiffusionQwenImageEdit(LatentDiffusionImageEdit):
    """Qwen-Image-Edit distillation with VL prompt encoding and packed source latents."""

    def _encode_images(self, images):
        if hasattr(self.vae, 'dtype'):
            vae_dtype = self.vae.dtype
        else:
            vae_dtype = next(self.vae.parameters()).dtype
        with torch.no_grad():
            # Qwen VAE expects [-1, 1] inputs, same as diffusers VaeImageProcessor.
            return self.vae.encode((images * 2 - 1).to(vae_dtype)).float()

    def _encode_negative_prompt_embeds(self, data, prompt_embed_kwargs, bs, negative_prompt):
        """Encode Qwen VL negative prompt (needs source image like the positive)."""
        if 'negative_prompt_embed_kwargs' in data:
            return data['negative_prompt_embed_kwargs']
        assert self.text_encoder is not None, \
            'Text encoder must be provided for encoding the negative prompt.'
        if 'negative_prompt_kwargs' in data:
            negative_prompt_kwargs = {k: v for k, v in data['negative_prompt_kwargs'].items()}
        else:
            negative_prompt_kwargs = dict(prompt=[negative_prompt] * bs)
        if 'condition_source_images' in data:
            negative_prompt_kwargs['condition_source_images'] = data['condition_source_images']
        elif 'source_images' in data:
            negative_prompt_kwargs['source_images'] = data['source_images']
        return self.text_encoder(**negative_prompt_kwargs)

    def _prepare_train_minibatch_diffusion_args(self, data):
        if 'prompt_embed_kwargs' in data:
            prompt_embed_kwargs = data['prompt_embed_kwargs']
        elif 'prompt_kwargs' in data:
            assert self.text_encoder is not None, 'Text encoder must be provided for encoding text to embeddings.'
            prompt_kwargs = {k: v for k, v in data['prompt_kwargs'].items()}
            if 'condition_source_images' in data:
                prompt_kwargs['condition_source_images'] = data['condition_source_images']
            elif 'source_images' in data:
                prompt_kwargs['source_images'] = data['source_images']
            prompt_embed_kwargs = self.text_encoder(**prompt_kwargs)
        else:
            raise ValueError('Either `prompt_embed_kwargs` or `prompt_kwargs` should be provided in the input data.')

        if self.train_cfg.get('use_edited_x0', False) and 'edited_images' in data:
            assert self.vae is not None, 'VAE must be provided for encoding edited images to latents.'
            latents = self._encode_images(data['edited_images'])
        elif 'latents' in data:
            latents = data['latents']
        elif 'edited_images' in data:
            assert self.vae is not None, 'VAE must be provided for encoding edited images to latents.'
            latents = self._encode_images(data['edited_images'])
        elif 'images' in data:
            assert self.vae is not None, 'VAE must be provided for encoding images to latents.'
            latents = self._encode_images(data['images'])
        else:
            raise ValueError('Either `latents`, `edited_images`, or `images` should be provided in the input data.')

        v = next(iter(prompt_embed_kwargs.values()))
        bs = v.size(0)
        device = v.device

        diffusion_args = (self.patchify(latents), )

        # Optional student true-CFG: double prompt embeds / image_latents like the
        # teacher, but keep a single-batch x_ref for residual velocity.
        student_guidance_scale = self.train_cfg.get('student_guidance_scale', None)
        student_use_guidance = (
            student_guidance_scale is not None
            and student_guidance_scale != 0.0
            and student_guidance_scale != 1.0)
        if student_use_guidance:
            negative_prompt = self.train_cfg.get(
                'student_negative_prompt',
                self.train_cfg.get('teacher_negative_prompt', ' '))
            negative_prompt_embed_kwargs = self._encode_negative_prompt_embeds(
                data, prompt_embed_kwargs, bs, negative_prompt)
            diffusion_kwargs = self._cat_padded_prompt_embeds(
                negative_prompt_embed_kwargs, prompt_embed_kwargs)
        else:
            diffusion_kwargs = prompt_embed_kwargs.copy()

        if 'source_images' in data:
            assert self.vae is not None, 'VAE must be provided for encoding source images.'
            source_latents = self._encode_images(data['source_images'])
            source_latents = self.patchify(source_latents)
            image_latents = source_latents
            if student_use_guidance:
                image_latents = torch.cat([image_latents, image_latents], dim=0)
            diffusion_kwargs['image_latents'] = image_latents
            if self.train_cfg.get('use_uedit', False) or self.train_cfg.get('use_uedit_new', False):
                diffusion_kwargs['x_ref'] = source_latents

        return diffusion_args, diffusion_kwargs, prompt_embed_kwargs, bs, device

    @staticmethod
    def _cat_padded_prompt_embeds(neg_kwargs, pos_kwargs):
        """Concat negative/positive Qwen VL prompt embeds along batch dim.

        Unlike fixed-length T5 embeds, Qwen VL embeds have variable text length,
        so both sides are right-padded (with mask=0) to a common length first.
        """
        import torch.nn.functional as F

        max_len = max(
            neg_kwargs['encoder_hidden_states'].size(1),
            pos_kwargs['encoder_hidden_states'].size(1))

        def _pad(embed_kwargs):
            embeds = embed_kwargs['encoder_hidden_states']
            mask = embed_kwargs.get('encoder_hidden_states_mask')
            # diffusers drops the mask when every token is valid; synthesize one
            # so CFG padding can still right-pad shorter negative prompts.
            if mask is None:
                mask = torch.ones(
                    embeds.size(0), embeds.size(1),
                    dtype=torch.long, device=embeds.device)
            pad_len = max_len - embeds.size(1)
            if pad_len > 0:
                embeds = F.pad(embeds, (0, 0, 0, pad_len))
                mask = F.pad(mask, (0, pad_len))
            return embeds, mask

        neg_embeds, neg_mask = _pad(neg_kwargs)
        pos_embeds, pos_mask = _pad(pos_kwargs)
        return dict(
            encoder_hidden_states=torch.cat([neg_embeds, pos_embeds], dim=0),
            encoder_hidden_states_mask=torch.cat([neg_mask, pos_mask], dim=0))

    def _prepare_train_minibatch_teacher_args(self, data, prompt_embed_kwargs, bs, device):
        # Qwen-Image-Edit is not guidance-distilled: the teacher needs true CFG
        # (official pipeline default true_cfg_scale=4.0 with negative prompt ' ')
        # to produce an edit-following velocity field.
        teacher_guidance_scale = self.train_cfg.get('teacher_guidance_scale', None)
        teacher_use_guidance = (teacher_guidance_scale is not None
                                and teacher_guidance_scale != 0.0 and teacher_guidance_scale != 1.0)

        if teacher_use_guidance:
            negative_prompt = self.train_cfg.get('teacher_negative_prompt', ' ')
            negative_prompt_embed_kwargs = self._encode_negative_prompt_embeds(
                data, prompt_embed_kwargs, bs, negative_prompt)
            teacher_kwargs = self._cat_padded_prompt_embeds(
                negative_prompt_embed_kwargs, prompt_embed_kwargs)
            # Match QwenImageEditPlusPipeline: after CFG extrapolation, rescale the
            # combined velocity back to the conditional prediction's per-token norm
            # (token = 2x2 packed latent patch, the transformer's packing unit).
            teacher_kwargs.update(
                guidance_scale=teacher_guidance_scale,
                guidance_norm_rescale=self.train_cfg.get('teacher_guidance_norm_rescale', True),
                guidance_norm_rescale_patch_size=2)
        else:
            teacher_kwargs = prompt_embed_kwargs.copy()

        if 'source_images' in data:
            assert self.vae is not None, 'VAE must be provided for encoding source images.'
            source_latents = self._encode_images(data['source_images'])
            image_latents = self.patchify(source_latents)
            if teacher_use_guidance:
                image_latents = torch.cat([image_latents, image_latents], dim=0)
            teacher_kwargs['image_latents'] = image_latents

        return teacher_kwargs

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        if 'prompt_embed_kwargs' in data:
            prompt_embed_kwargs = data['prompt_embed_kwargs']
        elif 'prompt_kwargs' in data:
            assert self.text_encoder is not None, 'Text encoder must be provided for encoding text to embeddings.'
            prompt_kwargs = {k: v for k, v in data['prompt_kwargs'].items()}
            if 'condition_source_images' in data:
                prompt_kwargs['condition_source_images'] = data['condition_source_images']
            elif 'source_images' in data:
                prompt_kwargs['source_images'] = data['source_images']
            prompt_embed_kwargs = self.text_encoder(**prompt_kwargs)
        else:
            raise ValueError('Either `prompt_embed_kwargs` or `prompt_kwargs` should be provided in the input data.')

        v = next(iter(prompt_embed_kwargs.values()))
        bs = v.size(0)
        device = v.device

        from copy import deepcopy
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        guidance_scale = cfg.get('guidance_scale', 1.0)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        with torch.no_grad():
            use_guidance = guidance_scale != 0.0 and guidance_scale != 1.0
            if use_guidance:
                negative_prompt = cfg.get(
                    'negative_prompt',
                    self.train_cfg.get(
                        'student_negative_prompt',
                        self.train_cfg.get('teacher_negative_prompt', ' ')))
                negative_prompt_embed_kwargs = self._encode_negative_prompt_embeds(
                    data, prompt_embed_kwargs, bs, negative_prompt)
                embed_kwargs = self._cat_padded_prompt_embeds(
                    negative_prompt_embed_kwargs, prompt_embed_kwargs)
            else:
                embed_kwargs = prompt_embed_kwargs.copy()

            if 'noise' in data:
                noise = data['noise']
            else:
                latent_size = cfg['latent_size']
                noise = torch.randn((bs, *latent_size), device=device)
            noise = self.patchify(noise.to(device))

            if 'source_images' in data:
                assert self.vae is not None, 'VAE must be provided for encoding source images.'
                source_latents = self._encode_images(data['source_images'].to(device))
                image_latents = self.patchify(source_latents)
                if use_guidance:
                    image_latents = torch.cat([image_latents, image_latents], dim=0)
                embed_kwargs['image_latents'] = image_latents

            latents_out = diffusion(
                noise=noise,
                guidance_scale=guidance_scale,
                test_cfg_override=test_cfg_override,
                **embed_kwargs)
            latents_out = self.unpatchify(latents_out)

            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            latents_out = latents_out.to(vae_dtype)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            out_images = (self.vae.decode(latents_out).float() / 2 + 0.5).clamp(min=0, max=1)

            return dict(num_samples=bs, pred_imgs=out_images)
