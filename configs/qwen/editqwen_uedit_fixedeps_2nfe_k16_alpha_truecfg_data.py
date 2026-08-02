_base_ = ['./_fsdp_train_edit.py', './_data_trainval_data.py']

# `train_flux_edit_fixedeps_alpha_truecfg_data_qwen.sh`
# -> gmqwen_uedit_fixedeps_alpha_softsign01_truecfg_k16_2nfe_pico400k
#
# Softsign-01 alpha + student true-CFG aligned with teacher:
#   alpha = 0.5 * (x / (1 + abs(x)) + 1)
#   student / teacher both run cond + uncond
#   same scale=4.0 and Qwen norm rescale
#   loss matches CFG-combined velocities
name = 'gmqwen_uedit_fixedeps_alpha_softsign01_truecfg_k16_2nfe_pico400k'
qwen_model = '/mnt/afs_zhangyunzhe/pretrained_models/Qwen-Image-Edit-2511'
qwen_transformer = f'{qwen_model}/transformer/diffusion_pytorch_model.safetensors.index.json'

model = dict(
    type='LatentDiffusionQwenImageEdit',
    vae=dict(
        type='PretrainedVAEQwenImage',
        from_pretrained=qwen_model,
        subfolder='vae',
        freeze=True,
        use_slicing=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='ArcFlowEditAlphaImitation',
        policy_type='ArcFlowEditNewAlpha',
        denoising=dict(
            type='ArcQwenEditAlphaSoftsign01ImageTransformer2DModel',
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
            pretrained=qwen_transformer,
            num_gaussians=16,
            logweights_channels=4,
            in_channels=64,
            out_channels=64,
            num_layers=60,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=3584,
            axes_dims_rope=(16, 56, 56),
            zero_cond_t=True,
            torch_dtype='bfloat16',
            checkpointing=True,
            use_lora=True,
            lora_target_modules=[
                'img_mlp.net.0.proj',
                'img_mlp.net.2',
                'timestep_embedder.linear_1',
                'timestep_embedder.linear_2'
            ] + [
                f'transformer_blocks.{i}.txt_mlp.net.0.proj' for i in range(59)
            ] + [
                f'transformer_blocks.{i}.txt_mlp.net.2' for i in range(59)
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
            use_dynamic_shifting=True,
            base_seq_len=1024,
            max_seq_len=32768,
            base_logshift=0.5,
            max_logshift=0.9,
            logit_normal_enable=False),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    teacher=dict(
        type='GaussianFlow',
        denoising=dict(
            type='QwenImageEditTransformer2DModel',
            patch_size=2,
            freeze=True,
            pretrained=qwen_transformer,
            in_channels=64,
            out_channels=64,
            num_layers=60,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=3584,
            axes_dims_rope=(16, 56, 56),
            zero_cond_t=True,
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
    # Teacher true-CFG
    teacher_guidance_scale=4.0,
    teacher_negative_prompt=' ',
    teacher_guidance_norm_rescale=True,
    # Student true-CFG (aligned with teacher)
    student_guidance_scale=4.0,
    student_negative_prompt=' ',
    student_guidance_norm_rescale=True,
    student_guidance_norm_rescale_patch_size=2,
    num_decay_iters=2000,
    window_substeps=3,
    gm_dropout=0.1,
    num_intermediate_states=4,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
)
test_cfg = dict(
    fixed_path_epsilon=True,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
    latent_size=(16, 128, 128),
    # Infer with the same true-CFG as training.
    guidance_scale=4.0,
    negative_prompt=' ',
    guidance_norm_rescale=True,
    guidance_norm_rescale_patch_size=2,
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
        resize_mode='qwen',
    ),
    interval=save_interval,
    must_save_interval=must_save_interval,
    output_dir='samples',
    max_samples=None,
    priority='LOW',
)

data = dict(
    workers_per_gpu=1,
    train=dict(resize_mode='qwen'),
    val=dict(resize_mode='qwen'),
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
resume_from = f'checkpoints/{name}/latest.pth'
workflow = [('train', save_interval)]
