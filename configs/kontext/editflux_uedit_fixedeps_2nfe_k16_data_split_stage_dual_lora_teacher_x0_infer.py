_base_ = ['./_ddp_train.py', './_data_trainval_data.py']

# Inference-only config for dual-LoRA teacher-x0 checkpoints.
# Used by:
#   evaluation/run_gmkontext_uedit_fixedeps_split_stage_dual_lora_teacher_x0_infer.sh
#
# Matches train_flux_edit_fixedeps_data_split_stage_dual_lora_teacher_x0.sh student:
#   ArcFlowEditImitationSplitStageDualLoraTeacherX0 + dual_stage_lora
#   step-1: integrate to mid; step-2: x0 = x_ref + pred_delta
# No LPIPS/DINO wrapper (train-only).

name = 'gmkontext_uedit_fixedeps_k16_2nfe_pico400k_split_stage_dual_lora_teacher_x0'
kontext_model = '/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev'
kontext_transformer = f'{kontext_model}/transformer/diffusion_pytorch_model.safetensors.index.json'

model = dict(
    type='LatentDiffusionImageEdit',
    vae=dict(
        type='PretrainedVAE',
        from_pretrained=kontext_model,
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='ArcFlowEditImitationSplitStageDualLoraTeacherX0',
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
            checkpointing=True,
            use_lora=True,
            dual_stage_lora=True,
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

train_cfg = dict(
    use_edited_x0=True,
    use_uedit=True,
    fixed_path_epsilon=True,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
    distilled_guidance_scale=3.5,
)
test_cfg = dict(
    distilled_guidance_scale=3.5,
    fixed_path_epsilon=True,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
    latent_size=(16, 128, 128),
)

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHookMod',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='karras',
        momentum_cfg=dict(gamma=7.0),
        priority='VERY_HIGH'),
]

load_from = None
resume_from = None
