# ~80GB+ VRAM target for FLUX.2-klein-base-9B Softsign alpha distillation (no GAN).
# Base uses classic CFG (cond+uncond); peak VRAM is higher than distilled klein-9B.

model = dict(
    diffusion=dict(
        denoising=dict(
            freeze_exclude_autocast_dtype='bfloat16')),

    text_encoder=dict(
        type='PretrainedFlux2KleinTextEncoder',
        from_pretrained='/mnt/afs_zhangyunzhe/pretrained_models/FLUX.2-klein-base-9B',
        max_sequence_length=512,
        text_encoder_out_layers=(9, 18, 27),
    )
)
train_cfg = dict(
    diffusion_grad_clip=50.0,
    diffusion_grad_clip_begin_iter=100,
)
optimizer = {
    'diffusion': dict(
        type='AdamW8bit', lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0,
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
    gc_interval=20)
dist_params = dict(backend='nccl')
log_level = 'INFO'
module_wrapper = 'ddp'
cudnn_benchmark = True
mp_start_method = 'fork'
