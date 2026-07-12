_base_ = ['./_fsdp_train.py', './_data_trainval_data.py']

# DMD2 (Tianwei Yin et al.) on pico-banana-400k image editing, 2-NFE.
# `train_flux_edit_dmd2_fixedeps_data.sh` -> gmkontext_dmd2_uedit_fixedeps_k16_2nfe_pico400k
#
# Generator: ArcFlowEdit residual student (official backward simulation:
#            random step index, no-grad bootstrap, single-step gradient)
# Real score: frozen Kontext teacher (distilled guidance embed)
# Fake score: LoRA-tuned Flux flow model (tied backbone with teacher)
# GAN (DMD2-native): fake score run in classify_mode + FluxDMD2ClsHead on
#            latents (official cls-on-clean-image + diffusion-GAN), softplus losses
name = 'gmkontext_dmd2_uedit_fixedeps_k16_2nfe_pico400k'
kontext_model = '/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev'
kontext_transformer = f'{kontext_model}/transformer/diffusion_pytorch_model.safetensors.index.json'

_lora_targets = [
    'proj_mlp',
    'proj_out',
    'ff.net.0.proj',
    'ff.net.2',
    'ff_context.net.0.proj',
    'ff_context.net.2',
    'timestep_embedder.linear_1',
    'timestep_embedder.linear_2',
]

model = dict(
    type='LatentDiffusionImageEditDMD2',
    tie_fake_score=True,
    vae=dict(
        type='PretrainedVAE',
        from_pretrained=kontext_model,
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='ArcFlowEditDMD2Imitation',
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
            lora_target_modules=_lora_targets,
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
    fake_score=dict(
        type='GaussianFlow',
        denoising=dict(
            type='FluxTransformer2DModel',
            patch_size=2,
            freeze=True,
            freeze_exclude=['lora'],
            pretrained=kontext_transformer,
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
            lora_target_modules=_lora_targets,
            lora_rank=64),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=3.2,
            logit_normal_enable=False),
        denoising_mean_mode='U'),
    # DMD2-native GAN head: shares the fake-score backbone (classify_mode);
    # only this small cls head is a separate trainable module.
    # 3072 = num_attention_heads (24) * attention_head_dim (128).
    discriminator=dict(
        type='FluxDMD2ClsHead',
        feature_dim=3072,
        hidden_dim=1024),
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
    # DMD2 knobs (aligned with the official SDXL 4-step backsim script)
    dmd2_gen_update_ratio=5,
    dmd2_dm_loss_weight=1.0,
    dmd2_fake_loss_weight=1.0,
    dmd2_reg_loss_weight=0.0,
    dmd2_real_guidance_scale=1.0,
    dmd2_fake_guidance_scale=1.0,
    dmd2_fake_train_guidance_scale=1.0,
    dmd2_fake_distilled_guidance_scale=1.0,
    dmd2_max_step_percent=0.98,
    dmd2_min_step_percent=0.02,
    dmd2_backward_simulation=True,
    # Official: gen_cls_loss_weight=5e-3, guidance_cls_loss_weight=1e-2,
    # --diffusion_gan --diffusion_gan_max_timestep 1000 (full range).
    dmd2_gan_loss_weight=5e-3,
    dmd2_guidance_cls_loss_weight=1e-2,
    dmd2_diffusion_gan=True,
    dmd2_diffusion_gan_max_step_percent=1.0,
    split_stage_step2_x_ref_scale=1.0,
    num_decay_iters=0,
    window_substeps=3,
    gm_dropout=0.1,
    num_intermediate_states=4,
    distilled_guidance_scale=3.5,
    teacher_distilled_guidance_scale=3.5,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
    diffusion_grad_clip=50.0,
    diffusion_grad_clip_begin_iter=100,
    fake_score_grad_clip=50.0,
    fake_score_grad_clip_begin_iter=100,
    discriminator_grad_clip=50.0,
    discriminator_grad_clip_begin_iter=100,
)
test_cfg = dict(
    distilled_guidance_scale=3.5,
    fixed_path_epsilon=True,
    split_stage_step2_x_ref_scale=1.0,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
    latent_size=(16, 128, 128),
)
# yapf: enable

optimizer = {
    'diffusion': dict(
        type='AdamW', lr=5e-7, betas=(0.9, 0.95), weight_decay=0.0,
        paramwise_cfg=dict(
            custom_keys={
                'proj_out_loggamma': dict(lr_mult=0.1),
            }),
    ),
    'fake_score': dict(
        type='AdamW', lr=5e-7, betas=(0.9, 0.95), weight_decay=0.0,
    ),
    # Official DMD2: the cls head belongs to the guidance model and shares
    # its optimizer (guidance_lr=5e-7).
    'discriminator': dict(
        type='AdamW', lr=5e-7, betas=(0.9, 0.95), weight_decay=0.0,
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

total_iters = 20000
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

# Extend FSDP ties: share frozen Kontext weights across teacher / student / fake_score.
fsdp_kwargs = dict(
    wrap_frozen_modules=True,
    ignore_frozen_parameters=False,
    fsdp_modules=[
        'diffusers.models.transformers.transformer_flux.FluxTransformerBlock',
        'diffusers.models.transformers.transformer_flux.FluxSingleTransformerBlock',
    ],
    exclude_keys=['vae'],
    tie_key_mappings=[
        'teacher->diffusion',
        'teacher->diffusion_ema',
        'teacher->fake_score',
    ],
)

load_from = None
resume_from = f'checkpoints/{name}/latest.pth'
workflow = [('train', save_interval)]
