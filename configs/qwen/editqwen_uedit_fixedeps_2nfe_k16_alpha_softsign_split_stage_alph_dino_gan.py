_base_ = ['./editqwen_uedit_fixedeps_2nfe_k16_alpha_split_stage_alph_dino_gan.py']

# Softsign-01 alpha + split-stage PIID + source-cond DINO GAN.
#   alpha = 0.5 * (raw / (1 + abs(raw)) + 1)
#   student_u = path_epsilon - alpha * x_ref - pred_delta
# Zero-logit initialization still gives alpha=0.5.
name = 'gmqwen_uedit_fixedeps_alpha_softsign01_k16_2nfe_pico400k_split_stage_alph_dino_gan'

model = dict(
    diffusion=dict(
        denoising=dict(
            type='ArcQwenEditAlphaSoftsign01ImageTransformer2DModel',
        ),
    ),
)

work_dir = f'work_dirs/{name}'
checkpoint_config = dict(out_dir=f'checkpoints/{name}')
resume_from = f'checkpoints/{name}/latest.pth'
