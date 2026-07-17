_base_ = ['./editflux_uedit_fixedeps_2nfe_k16_data.py']

# Inference-only variant of fixedeps 2-NFE:
#   step-1: same velocity integration  x <- x - v dt
#           v = path_epsilon - x_ref - pred_delta
#   step-2: skip integration; decode x0 = x_ref + pred_delta
#
# Used by:
#   evaluation/run_gmkontext_uedit_fixedeps_step2_direct_x0_infer.sh
#
# Does NOT change training. Default ArcFlowEditImitation.forward_test stays
# unchanged unless test_cfg.step2_direct_x0=True.

name = 'gmkontext_uedit_fixedeps_k16_2nfe_pico400k_step2_direct_x0'

test_cfg = dict(
    distilled_guidance_scale=3.5,
    fixed_path_epsilon=True,
    nfe=2,
    timestep_ratio=1.0,
    total_substeps=128,
    latent_size=(16, 128, 128),
    step2_direct_x0=True,
)
