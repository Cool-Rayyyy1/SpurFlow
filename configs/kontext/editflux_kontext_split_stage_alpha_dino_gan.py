_base_ = ['./editflux_uedit_fixedeps_2nfe_k16_data_step2_alph_dino_gan.py']

# Formal FLUX.1 Kontext training.
# One shared 2-NFE split-stage rollout (not a second independent rollout):
#   step 1, t=1: teacher imitation
#   step 2, student endpoint -> t=0: teacher imitation, direct flow, and the GAN
# The GAN fake is the VAE decode of that step-2 endpoint. The step-2 alpha map
# picks the local crop. Real pairs are (reference image, dataset edit).
# Paths, data mix, and the warmup checkpoint are set by train_flux_kontext.sh.

name = 'flux_kontext'

model = dict(
    type='LatentDiffusionImageEditSplitStageAlphaDinoFeatureGAN',
    diffusion=dict(
        type='ArcFlowEditImitationSplitStageGAN',
        policy_type='ArcFlowEditNewAlpha',
    ),
)
