_base_ = ['./_ddp_train.py', './_data_trainval_data.py']

# `train_flux_edit_fixedeps_alpha_data_lpips.sh`
# Alpha PIID + same-segment analytic x0 LPIPS (20% of PIID) + DINOv3 feature loss.
name = 'gmkontext_uedit_fixedeps_alpha_k16_2nfe_pico400k_lpips'
kontext_model = '/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev'
kontext_transformer = f'{kontext_model}/transformer/diffusion_pytorch_model.safetensors.index.json'
lpips_weights = '/mnt/afs_zhangyunzhe/pretrained_models/lpips/vgg.pth'
lpips_vgg16 = '/mnt/afs_zhangyunzhe/pretrained_models/lpips/vgg16-397923af.pth'
dinov3_model = (
    '/mnt/afs_zhangyunzhe/pretrained_models/'
    'dinov3-vitl16-pretrain-lvd1689m/model.safetensors')

model = dict(
    type='LatentDiffusionImageEditAlphaLpips',
    vae=dict(
        type='PretrainedVAE',
        from_pretrained=kontext_model,
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    lpips=dict(
        weights_path=lpips_weights,
        vgg_weights_path=lpips_vgg16,
        spatial=False),
    dino_loss=dict(
        checkpoint_path=dinov3_model,
        input_size=224,
        feature_layers=(23,),
        backbone_dtype='bf16',
        loss_type='cosine'),
    diffusion=dict(
        type='ArcFlowEditAlphaLpipsImitation',
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
    # LPIPS / DINO on the same random-segment x0_hat (no 2-NFE rollout).
    lpips_loss_weight=0.2,
    dino_loss_weight=0.1,
    # Downsample decoded images before LPIPS/DINO.
    perceptual_image_size=256,
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

sample_eval = dict(
    type='EditFlowSampleImagesHook',
    enabled=True,
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

load_from = None
resume_from = f'checkpoints/{name}/latest.pth'
workflow = [('train', save_interval)]
