# Kontext FSDP training (lower per-GPU VRAM than DDP; needs torchrun multi-GPU).
# Typical: 8 GPUs hybrid shard for edit / split-stage GAN.

model = dict(
    diffusion=dict(
        denoising=dict(
            freeze_exclude_autocast_dtype='bfloat16')),

    text_encoder=dict(
        type='PretrainedFluxTextEncoder',
        from_pretrained='/path/to/checkpoints/FLUX.1-Kontext-dev',
    )
)
train_cfg = dict(
    grad_accum_batch_size=1,
    diffusion_grad_clip=50.0,
    diffusion_grad_clip_begin_iter=100,
)
optimizer = {
    'diffusion': dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0,
        paramwise_cfg=dict(
            custom_keys={
                'proj_out_loggamma': dict(lr_mult=0.1),
            }),
    ),
}
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001)
runner = dict(
    type='DynamicIterBasedRunnerMod',
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16=True,
    ckpt_fp16_ema=True,
    ckpt_bf16_optim=True,
    gc_interval=20)
dist_params = dict(backend='nccl')
log_level = 'INFO'
module_wrapper = 'fsdp'
fsdp_kwargs = dict(
    wrap_frozen_modules=True,
    ignore_frozen_parameters=False,
    fsdp_modules=[
        'diffusers.models.transformers.transformer_flux.FluxTransformerBlock',
        'diffusers.models.transformers.transformer_flux.FluxSingleTransformerBlock',
    ],
    exclude_keys=['vae'],
    tie_key_mappings=['teacher->diffusion', 'teacher->diffusion_ema'],
)
cudnn_benchmark = True
mp_start_method = 'fork'
