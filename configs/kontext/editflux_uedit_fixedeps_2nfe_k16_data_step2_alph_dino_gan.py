_base_ = ['./_fsdp_train.py', './_data_trainval_data.py']

# `train_flux_edit_fixedeps_data_step2_alph_dino_gan.sh`
# Fixed-eps alpha PIID + step-2 TDM-style DINO feature GAN with mask-guided local crop.
# Fake: 2-NFE rollout -> step2 alpha + endpoint latent -> VAE decode.
# Crops (shared real/fake): full global, random local, alpha-mask local (low alpha = edit).
# GAN grads flow through both NFE steps (gan_grad_step2_only=False by default).
name = 'gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_step2_alph_dino_gan'
kontext_model = '/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev'
kontext_transformer = f'{kontext_model}/transformer/diffusion_pytorch_model.safetensors.index.json'
dinov3_model = '/mnt/afs_zhangyunzhe/pretrained_models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors'

model = dict(
    type='LatentDiffusionImageEditStep2AlphaDinoFeatureGAN',
    vae=dict(
        type='PretrainedVAE',
        from_pretrained=kontext_model,
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='ArcFlowEditImitationStep2GAN',
        policy_type='ArcFlowEditNewAlpha',
        denoising=dict(
            type='ArcFluxEditNewAlphaTransformer2DModel',
            patch_size=2,
            freeze=True,
            freeze_exclude=[
                'proj_out_deltax',
                'proj_out_logweights',
                'proj_out_loggamma',
                'proj_out_alpha',
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
            checkpointing=True,
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
        type='DinoAlphaMaskFeatureDiscriminator',
        checkpoint_path=dinov3_model,
        num_steps=2,
        feature_layers=(23,),
        global_input_size=224,
        local_input_size=224,
        num_global_crops=1,
        num_local_crops=1,
        global_crop_scale=(0.5, 1.0),
        local_crop_scale=(0.125, 0.5),
        crop_aspect_ratio=(0.75, 1.3333333333),
        clamp_pixels=True,
        step_conditioning=False,
        head_num_blocks=3,
        head_use_avgpool=False,
        head_gradient_checkpointing=True,
        head_norm_groups=32,
        dense_output=False,
        backbone_dtype='bf16',
        head_dtype='fp32',
        freeze_backbone=True,
        use_mask_local_crop=True,
        p_disable_local=0.0,
        edit_is_low_alpha=True,
        alpha_smooth_sigma=2.0,
        mass_threshold_percentile=30.0,
        mass_coverage_min=0.85,
        mass_coverage_max=0.90,
        union_area_max_ratio=0.55,
        bbox_expand_factor=1.4,
        min_crop_area_ratio=0.05,
        max_crop_area_ratio=0.55,
        min_edit_mass_ratio=0.002,
        min_component_pixels=16,
        gan_global_weight=1.0,
        gan_random_local_weight=1.0,
        gan_mask_local_weight=1.0),
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
    split_stage_gan_warmup_iters=0,
    split_stage_gan_ramp_iters=0,
    split_stage_gan_loss_weight=0.05,
    # Anchor pred_delta ~ x0_tgt - alpha*x_ref (same role as split-stage direct flow).
    direct_delta_loss_weight=0.5,
    gan_grad_step2_only=True,
    num_decay_iters=0,
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
        type='AdamW', lr=5e-5, betas=(0.0, 0.95), weight_decay=0.01,
    ),
}

fsdp_kwargs = dict(
    wrap_frozen_modules=True,
    ignore_frozen_parameters=False,
    fsdp_modules=[
        'diffusers.models.transformers.transformer_flux.FluxTransformerBlock',
        'diffusers.models.transformers.transformer_flux.FluxSingleTransformerBlock',
    ],
    exclude_keys=['vae', 'discriminator'],
    tie_key_mappings=['teacher->diffusion', 'teacher->diffusion_ema'],
)

sample_eval = dict(
    type='EditFlowSampleImagesHook',
    enabled=True,
    # Fixed ImgEdit-Bench subset: 9 categories x 5 examples (seeded).
    dataset=dict(
        type='ImgEditBenchSample',
        annotations_path=(
            '/mnt/afs_zhangyunzhe/EditFlow/evaluation/imgedit_bench/'
            'annotations/basic_edit.json'),
        bench_root='/mnt/afs_zhangyunzhe/dataset/imgedit/benchmark/Benchmark',
        categories=[
            'action', 'add', 'adjust', 'background', 'compose',
            'extract', 'remove', 'replace', 'style'],
        samples_per_category=5,
        seed=42,
        resize_mode='kontext',
    ),
    interval=save_interval,
    must_save_interval=must_save_interval,
    output_dir='samples',
    max_samples=None,
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
