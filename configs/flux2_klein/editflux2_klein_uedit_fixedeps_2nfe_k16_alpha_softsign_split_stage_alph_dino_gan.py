_base_ = ['./_fsdp_train.py', './_data_trainval_data.py']

# `train_flux2_klein_edit_fixedeps_alpha_softsign_alph_dino_gan.sh`
# FLUX.2-klein-base Softsign-01 alpha + split-stage PIID + source-cond DINO GAN.
#
# Softsign-01 (same as klein softsign baseline / Qwen softsign):
#   alpha = 0.5 * (raw / (1 + abs(raw)) + 1)
#   student_u = path_epsilon - alpha * x_ref - pred_delta
#
# GAN recipe matches Qwen alph_dino_gan:
#   DinoAlphaMaskFeatureDiscriminator, condition_on_source,
#   cat([DINO(ref), DINO(target)]), alpha-mask local crop from step-2 alpha,
#   gan_weight=0.05, gan_grad_step2_only=False (grads through both NFE steps).
# Distillation is split-stage so the 2-NFE unroll is shared with GAN.
#
# Guidance (unchanged from klein softsign):
#   Teacher (klein-base): classic CFG=4.0 (cond + uncond)
#   Student: CFG=1 / cond-only

name = (
    'gmklein_base_uedit_fixedeps_alpha_softsign01_k16_2nfe_'
    'pico400k_split_stage_alph_dino_gan')
klein_model = '/mnt/afs_gaochengmin/checkpoints/FLUX.2-klein-base-9B'
klein_transformer = f'{klein_model}/transformer/diffusion_pytorch_model.safetensors.index.json'
dinov3_model = (
    '/mnt/afs_gaochengmin/checkpoints/dinov3/'
    'dinov3-vitl16-pretrain-lvd1689m/model.safetensors')

model = dict(
    type='LatentDiffusionFlux2KleinImageEditAlphaSplitStageDinoGAN',
    vae=dict(
        type='PretrainedVAEFlux2',
        from_pretrained=klein_model,
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='ArcFlowEditImitationSplitStageGAN',
        policy_type='ArcFlowEditNewAlpha',
        denoising=dict(
            type='ArcFlux2EditAlphaSoftsign01Transformer2DModel',
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
            pretrained=klein_transformer,
            num_gaussians=16,
            logweights_channels=4,
            in_channels=128,
            num_layers=8,
            num_single_layers=24,
            attention_head_dim=128,
            num_attention_heads=32,
            joint_attention_dim=12288,
            axes_dims_rope=(32, 32, 32, 32),
            guidance_embeds=False,
            torch_dtype='bfloat16',
            checkpointing=True,
            use_lora=True,
            lora_target_modules=[
                'to_q',
                'to_k',
                'to_v',
                'to_out.0',
                'to_out',
                'add_q_proj',
                'add_k_proj',
                'add_v_proj',
                'to_add_out',
                'ff.linear_in',
                'ff.linear_out',
                'ff_context.linear_in',
                'ff_context.linear_out',
                'to_qkv_mlp_proj',
            ],
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
            type='Flux2Transformer2DModel',
            patch_size=2,
            freeze=True,
            pretrained=klein_transformer,
            in_channels=128,
            num_layers=8,
            num_single_layers=24,
            attention_head_dim=128,
            num_attention_heads=32,
            joint_attention_dim=12288,
            axes_dims_rope=(32, 32, 32, 32),
            guidance_embeds=False,
            torch_dtype='bfloat16'),
        num_timesteps=1,
        denoising_mean_mode='U'),
    tie_teacher=True,
    text_encoder=dict(
        type='PretrainedFlux2KleinTextEncoder',
        from_pretrained=klein_model,
        torch_dtype='bfloat16',
        max_sequence_length=512,
        text_encoder_out_layers=(9, 18, 27),
    ),
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
        condition_on_source=True,
        use_mask_local_crop=True,
        p_disable_local=0.3,
        edit_is_low_alpha=True,
        alpha_smooth_sigma=2.0,
        mass_threshold_percentile=30.0,
        mass_coverage_min=0.85,
        mass_coverage_max=0.90,
        hot_mass_frac=0.40,
        union_area_max_ratio=0.35,
        bbox_expand_factor=1.1,
        min_crop_area_ratio=0.01,
        max_crop_area_ratio=0.35,
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
    split_stage_diffusion_loss_weight=0.5,
    split_stage_teacher_loss_weight=0.5,
    split_stage_step2_x_ref_scale=1.0,
    split_stage_gan_warmup_iters=0,
    split_stage_gan_ramp_iters=0,
    split_stage_gan_loss_weight=0.05,
    direct_delta_loss_weight=0.0,
    # GAN grads flow through both NFE steps (do not detach step-1 state).
    gan_grad_step2_only=False,
    # Teacher: official klein-base CFG=4.0 (cond + uncond two DiT forwards)
    teacher_guidance_scale=4.0,
    teacher_negative_prompt='',
    teacher_guidance_norm_rescale=False,
    num_decay_iters=0,
    window_substeps=3,
    gm_dropout=0.1,
    num_intermediate_states=4,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
)
test_cfg = dict(
    fixed_path_epsilon=True,
    split_stage_step2_x_ref_scale=1.0,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
    latent_size=(32, 128, 128),
    guidance_scale=1.0,
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
        'diffusers.models.transformers.transformer_flux2.Flux2TransformerBlock',
        'diffusers.models.transformers.transformer_flux2.Flux2SingleTransformerBlock',
    ],
    exclude_keys=['vae', 'discriminator'],
    tie_key_mappings=['teacher->diffusion', 'teacher->diffusion_ema'],
)

sample_eval = dict(
    type='EditFlowSampleImagesHook',
    enabled=True,
    dataset=dict(
        type='ImgEditBenchSample',
        annotations_path=(
            '/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17/EditFlow/'
            'evaluation/imgedit_bench/annotations/basic_edit.json'),
        bench_root='/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark',
        categories=[
            'action', 'add', 'adjust', 'background', 'compose',
            'extract', 'remove', 'replace', 'style'],
        samples_per_category=5,
        seed=42,
        resize_mode='flux2',
        latent_channels=32,
    ),
    interval=save_interval,
    must_save_interval=must_save_interval,
    output_dir='samples',
    max_samples=None,
    priority='LOW',
)

data = dict(
    workers_per_gpu=1,
    # Paired source/edit only: source-conditional GAN labels D(ref, edit)=1.
    train=dict(
        resize_mode='flux2',
        latent_size=(32, 128, 128),
        load_unpaired_edited=False),
    val=dict(resize_mode='flux2', latent_size=(32, 128, 128)),
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

total_iters = 50000
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
