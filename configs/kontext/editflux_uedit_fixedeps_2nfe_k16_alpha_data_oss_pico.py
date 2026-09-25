_base_ = ['./editflux_uedit_fixedeps_2nfe_k16_alpha_data.py']

# `train_flux_edit_fixedeps_alpha_data.sh`
# Same non-split ArcFlowEditAlphaImitation sigmoid-alpha recipe as the pico-only
# baseline (ArcFluxEditNewAlphaTransformer2DModel), but:
#   - init from FLUX.1-Kontext-dev (no EditFlow student pretrain by default)
#   - train mix 30% oss_edit + 70% pico-banana-400k
#
# Student remains cond-only 2-NFE. Teacher distilled CFG=3.5. No GAN.

pico_root = '/path/to/data/pico-banana-400k'
oss_root = '/path/to/data/oss_edit'
kontext_model = '/path/to/checkpoints/FLUX.1-Kontext-dev'
kontext_transformer = f'{kontext_model}/transformer/diffusion_pytorch_model.safetensors.index.json'
name = 'gmkontext_uedit_fixedeps_alpha_k16_2nfe_oss30_pico70'
work_dir = f'work_dirs/{name}'
resume_from = None
load_from = None

model = dict(
    vae=dict(from_pretrained=kontext_model),
    text_encoder=dict(from_pretrained=kontext_model),
    diffusion=dict(
        denoising=dict(pretrained=kontext_transformer)),
    teacher=dict(
        denoising=dict(pretrained=kontext_transformer)),
)

data = dict(
    train=dict(
        _delete_=True,
        type='ProbMixDataset',
        probs=[0.3, 0.7],
        datasets=[
            dict(
                type='OssEdit',
                data_root=oss_root,
                jsonl_path='metadata.jsonl',
                require_edited=True,
                resize_mode='kontext',
                image_size=1024,
                load_unpaired_edited=False,
            ),
            dict(
                type='ImageEdit',
                data_root=pico_root,
                jsonl_path='jsonl/sft_with_local_source_image_path.jsonl',
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
        data_root=pico_root,
        jsonl_path='jsonl/sft_with_local_source_image_path.jsonl',
        edited_images_dir='edited_images',
        image_size=1024,
        resize_mode='kontext',
        start_ind=-128,
        repeat=2,
        test_mode=True,
    ),
)

sample_eval = dict(
    _delete_=True,
    type='EditFlowSampleImagesHook',
    enabled=True,
    dataset=dict(
        type='GEditV2Sample',
        split='gedit_v2',
        annotations_path='/path/to/data/benchmark/GEdit_v2/gedit_v2_meta.json',
        bench_root='/path/to/data/benchmark/GEdit_v2',
        samples_per_category=2,
        language=None,
        seed=42,
        resize_mode='kontext',
    ),
    interval=500,
    must_save_interval=0,
    output_dir='samples',
    max_samples=None,
    priority='LOW',
)
