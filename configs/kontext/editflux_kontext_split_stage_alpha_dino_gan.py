_base_ = ['./editflux_uedit_fixedeps_2nfe_k16_data_step2_alph_dino_gan.py']

# Formal SpurFlow training.
# One shared 2-NFE split-stage rollout (not a second independent rollout):
#   step 1, t=1: teacher imitation
#   step 2, student endpoint -> t=0: teacher imitation, direct flow, and the GAN
# The GAN fake is the VAE decode of that step-2 endpoint. The step-2 alpha map
# picks the local crop. Real pairs are (reference image, dataset edit).
# Paths, data mix, and the warmup checkpoint are set by train_flux_kontext.sh.

name = 'spurflow'

data_a_root = '/path/to/paired_edit_set_a'
data_b_root = '/path/to/paired_edit_set_b'

data = dict(
    train=dict(
        _delete_=True,
        type='ProbMixDataset',
        probs=[0.3, 0.7],
        datasets=[
            dict(
                type='PairEdit',
                data_root=data_a_root,
                jsonl_path='metadata.jsonl',
                require_edited=True,
                resize_mode='kontext',
                image_size=1024,
                load_unpaired_edited=False,
            ),
            dict(
                type='ImageEdit',
                data_root=data_b_root,
                jsonl_path='metadata.jsonl',
                edited_images_dir='edited_images',
                image_size=1024,
                require_edited=True,
                resize_mode='kontext',
                load_unpaired_edited=False,
                end_ind=-128,
            ),
        ],
    ),
    val=dict(
        _delete_=True,
        type='ImageEdit',
        data_root=data_b_root,
        jsonl_path='metadata.jsonl',
        edited_images_dir='edited_images',
        image_size=1024,
        resize_mode='kontext',
        start_ind=-128,
        repeat=2,
        test_mode=True,
    ),
)

model = dict(
    type='LatentDiffusionImageEditSplitStageAlphaDinoFeatureGAN',
    diffusion=dict(
        type='ArcFlowEditImitationSplitStageGAN',
        policy_type='ArcFlowEditNewAlpha',
    ),
)
