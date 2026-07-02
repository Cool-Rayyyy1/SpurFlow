_base_ = ['./_fsdp_train.py', './_data_trainval_data.py']

# `train_flux_edit_fixedeps_data_split_stage_gan.sh`
# Resume fixed-eps pretrain, then split-stage rollout with step-2 DINOv3 GAN.
# Step-1 / step-2 PIID are full weight from iter 0. GAN is on from iter 0 by default
# (override split_stage_gan_warmup_iters / split_stage_gan_ramp_iters to delay or ramp).
# Fake: Kontext unpatchify + VAE decode, then DINOv3 Resize. Real: edited_images + Resize.
name = 'gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_gan'
kontext_model = '/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev'
kontext_transformer = f'{kontext_model}/transformer/diffusion_pytorch_model.safetensors.index.json'
dinov3_model = '/mnt/afs_zhangyunzhe/pretrained_models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors'

model = dict(
    type='LatentDiffusionImageEditSplitStageGAN',
    vae=dict(
        type='PretrainedVAE',
        from_pretrained=kontext_model,
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='ArcFlowEditImitationSplitStageGAN',
        policy_type='ArcFlowEdit',
        denoising=dict(
            type='ArcFluxEditNewTransformer2DModel',
            patch_size=2,
            freeze=True,
            freeze_exclude=[
                'proj_out_deltax',
                'proj_out_logweights',
                'proj_out_loggamma',
                'norm_out',
                'lora'],
            inherit_proj_out_deltax=False,
            deltax_init='kaiming',
            pretrained=kontext_transformer,
            num_gaussians=16,
            logweights_channels=4,
            in_channels=64,
            num_layers=19,
            num_single_layers=38,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            guidance_embeds=True,
            torch_dtype='bfloat16',
            checkpointing=False,
            use_lora=True,
            lora_target_modules=[
                'proj_mlp',
                'proj_out',
                'ff.net.0.proj',
                'ff.net.2',
                'ff_context.net.0.proj',
                'ff_context.net.2',
                'timestep_embedder.linear_1',
                'timestep_embedder.linear_2'],
            lora_dropout=0.05,
            lora_rank=256),
        flow_loss=dict(
            type='DiffusionMSELoss',
            data_info=dict(pred='u_t_pred', target='u_t'),
            rescale_mode='constant',
            rescale_cfg=dict(scale=30.0)),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=3.2,
            logit_normal_enable=False),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    teacher=dict(
        type='GaussianFlow',
        denoising=dict(
            type='FluxTransformer2DModel',
            patch_size=2,
            freeze=True,
            pretrained=kontext_transformer,
            in_channels=64,
            num_layers=19,
            num_single_layers=38,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            guidance_embeds=True,
            torch_dtype='bfloat16'),
        num_timesteps=1,
        denoising_mean_mode='U'),
    tie_teacher=True,
    discriminator=dict(
        type='DINOv3PatchDiscriminator',
        checkpoint_path=dinov3_model,
        input_size=224,
        global_weight=0.25,
        patch_weight=1.0,
        freeze_backbone=True),
)

save_interval = 500
must_save_interval = 1000
eval_interval = 500
work_dir = f'work_dirs/{name}'
# yapf: disable
train_cfg = dict(
    use_edited_x0=True,
    use_uedit=True,
    fixed_path_epsilon=True,
    split_stage_teacher_loss_weight=0.5,
    split_stage_step2_x_ref_scale=1.0,
    split_stage_gan_warmup_iters=0,
    split_stage_gan_ramp_iters=0,
    split_stage_gan_loss_weight=1.0,
    num_decay_iters=2000,
    window_substeps=3,
    gm_dropout=0.1,
    num_intermediate_states=4,
    distilled_guidance_scale=3.5,
    teacher_distilled_guidance_scale=3.5,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
)
test_cfg = dict(
    distilled_guidance_scale=3.5,
    fixed_path_epsilon=True,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
    latent_size=(16, 128, 128),
)
# yapf: enable

optimizer = {
    'diffusion': dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0,
        paramwise_cfg=dict(
            custom_keys={
                'proj_out_loggamma': dict(lr_mult=0.1),
            }),
    ),
    'discriminator': dict(
        type='AdamW', lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0,
    ),
}

sample_eval = dict(
    type='EditFlowSampleImagesHook',
    enabled=True,
    data='val',
    interval=save_interval,
    must_save_interval=must_save_interval,
    output_dir='samples',
    max_samples=8,
    priority='LOW',
)

data = dict(
    workers_per_gpu=1,
    train=dict(resize_mode='kontext'),
    val=dict(resize_mode='kontext'),
    train_dataloader=dict(samples_per_gpu=1),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    persistent_workers=False,
    prefetch_factor=2,
)
checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir='checkpoints/')

total_iters = 25000
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHookMod',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=100,
        momentum_policy='karras',
        momentum_cfg=dict(gamma=7.0),
        priority='VERY_HIGH'),
]

load_from = None
resume_from = None
workflow = [('train', save_interval)]
