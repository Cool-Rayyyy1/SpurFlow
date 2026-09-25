_base_ = ['./editqwen_uedit_fixedeps_2nfe_k16_alpha_data.py']

# `train_flux_edit_fixedeps_alpha_data_qwen.sh`
# Same non-split ArcFlowEditAlphaImitation recipe as the sigmoid pico-only
# baseline, but:
#   - alpha head is Softsign-01
#   - init from Qwen-Image-Edit-2511 (no EditFlow student pretrain by default)
#   - train mix 30% oss_edit + 70% pico-banana-400k
#
# Student remains cond-only 2-NFE (no true-CFG). Teacher CFG=4 + Qwen norm rescale.

pico_root = '/mnt/afs_gaochengmin/data/pico-banana-400k'
oss_root = '/mnt/afs_gaochengmin/data/oss_edit'
qwen_model = '/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511'
qwen_transformer = f'{qwen_model}/transformer/diffusion_pytorch_model.safetensors.index.json'
name = 'gmqwen_uedit_fixedeps_alpha_softsign01_k16_2nfe_oss30_pico70'
work_dir = f'work_dirs/{name}'
resume_from = None
load_from = None
load_ignore_key_prefixes = ['discriminator.']

model = dict(
    vae=dict(from_pretrained=qwen_model),
    text_encoder=dict(from_pretrained=qwen_model),
    diffusion=dict(
        denoising=dict(
            type='ArcQwenEditAlphaSoftsign01ImageTransformer2DModel',
            pretrained=qwen_transformer)),
    teacher=dict(
        denoising=dict(pretrained=qwen_transformer)),
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
                resize_mode='qwen',
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
                resize_mode='qwen',
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
        resize_mode='qwen',
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
        annotations_path='/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json',
        bench_root='/mnt/afs_caiqi/data/benchmark/GEdit_v2',
        samples_per_category=2,
        language=None,
        seed=42,
        resize_mode='qwen',
    ),
    interval=500,
    must_save_interval=0,
    output_dir='samples',
    max_samples=None,
    priority='LOW',
)
