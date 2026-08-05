_base_ = ['./_fsdp_train.py', './_data_trainval_data.py']

# `train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh`
# -> gmklein_base_uedit_fixedeps_alpha_softsign01_k16_2nfe_pico400k
#
# Distill FLUX.2-klein-base-9B -> 2-NFE Softsign-01 alpha student (no GAN).
# Goal: beat the official step-distilled FLUX.2-klein-9B with our 2-step student.
#
# Distillation architecture mirrors Kontext fixedeps+alpha
# (`train_flux_edit_fixedeps_alpha_data.sh`):
#   pred_delta = mixture(deltax_k; weights, gammas) ~ x0_tgt - x_ref
#   student_u = path_epsilon - alpha * x_ref - pred_delta
#   fixed_path_epsilon + use_uedit + 2-NFE ArcFlow imitation
#
# Softsign-01 alpha (same as Qwen softsign; NOT Kontext sigmoid):
#   alpha = 0.5 * (raw / (1 + abs(raw)) + 1)
# deliberately NO DINO/GAN.
#
# Guidance:
#   - Teacher (klein-base): official classic CFG scale=4.0
#     (Flux2KleinPipeline: cond forward + uncond forward, then combine;
#      guidance_embeds=false — NOT Flux2-dev / Kontext single-pass embeds)
#   - Student: guidance_scale=1.0 — cond only, no uncond forward
#
# Data: resize_mode='flux2' matches Flux2KleinPipeline
#   (area-cap ~1MP + multiple-of-16; NOT Kontext buckets)

name = 'gmklein_base_uedit_fixedeps_alpha_softsign01_k16_2nfe_pico400k'
klein_model = '/mnt/afs_zhangyunzhe/pretrained_models/FLUX.2-klein-base-9B'
klein_transformer = f'{klein_model}/transformer/diffusion_pytorch_model.safetensors.index.json'

model = dict(
    type='LatentDiffusionFlux2KleinImageEdit',
    vae=dict(
        type='PretrainedVAEFlux2',
        from_pretrained=klein_model,
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='ArcFlowEditAlphaImitation',
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
            # Patterns are resolved to nn.Linear paths in Flux2Transformer2DModel.
            # Keep both to_out.0 (double-stream ModuleList[0]) and to_out (single-stream Linear).
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
            shift=3.2,  # match Kontext fixedeps (train_flux_edit_fixedeps_*.sh)
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
    # Teacher: official klein-base CFG=4.0 (cond + uncond two DiT forwards)
    teacher_guidance_scale=4.0,
    teacher_negative_prompt='',
    teacher_guidance_norm_rescale=False,
    # Student: cond-only (no uncond / no CFG)
    # omit student_guidance_scale or keep <=1 so ArcFlow skips TrueCFGEditPolicyPair
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
    latent_size=(32, 128, 128),
    guidance_scale=1.0,
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
    train=dict(resize_mode='flux2', latent_size=(32, 128, 128)),
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
