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
        diffusion_kwargs = prompt_embed_kwargs.copy()

        if 'source_images' in data:
            assert self.vae is not None, 'VAE must be provided for encoding source images.'
            source_latents = self._encode_images(data['source_images'])
            source_latents = self.patchify(source_latents)
            diffusion_kwargs['image_latents'] = source_latents
            if self.train_cfg.get('use_uedit', False) or self.train_cfg.get('use_uedit_new', False):
                diffusion_kwargs['x_ref'] = source_latents

        return diffusion_args, diffusion_kwargs, prompt_embed_kwargs, bs, device

    def _prepare_train_minibatch_teacher_args(self, data, prompt_embed_kwargs, bs, device):
        teacher_kwargs = super()._prepare_train_minibatch_teacher_args(
            data, prompt_embed_kwargs, bs, device)

        if 'source_images' in data:
            assert self.vae is not None, 'VAE must be provided for encoding source images.'
            source_latents = self._encode_images(data['source_images'])
            image_latents = self.patchify(source_latents)
            first_val = next(iter(teacher_kwargs.values()))
            if isinstance(first_val, torch.Tensor) and first_val.size(0) == 2 * bs:
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
                if 'negative_prompt_embed_kwargs' in data:
                    negative_prompt_embed_kwargs = data['negative_prompt_embed_kwargs']
                elif 'negative_prompt_kwargs' in data:
                    negative_prompt_kwargs = {k: v for k, v in data['negative_prompt_kwargs'].items()}
                    if 'condition_source_images' in data:
                        negative_prompt_kwargs['condition_source_images'] = data['condition_source_images']
                    elif 'source_images' in data:
                        negative_prompt_kwargs['source_images'] = data['source_images']
                    negative_prompt_embed_kwargs = self.text_encoder(**negative_prompt_kwargs)
                else:
                    raise ValueError(
                        'Either `negative_prompt_embed_kwargs` or `negative_prompt_kwargs` should be provided in the '
                        'input data for classifier-free guidance.')
                embed_kwargs = {
                    k: torch.cat([negative_prompt_embed_kwargs[k], v], dim=0)
                    for k, v in prompt_embed_kwargs.items()}
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

            out_images = (self.vae.decode(latents_out).float() / 2 + 0.5).clamp(min=0, max=1)

            return dict(num_samples=bs, pred_imgs=out_images)
