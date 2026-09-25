_base_ = ['./editflux_2nfe_k16_data.py']

# `train_flux_data.sh`
# Vanilla ArcFlowImitation (policy_type='ArcFlow'), NO uedit / fixedeps / alpha / GAN.
# Only the training data changes: default 30% oss_edit + 70% pico-banana-400k.
# Override at launch with OSS_PROB / PICO_PROB.

pico_root = '/mnt/afs_gaochengmin/data/pico-banana-400k'
oss_root = '/mnt/afs_gaochengmin/data/oss_edit'
name = 'gmkontext_k16_2nfe_oss30_pico70_data'
work_dir = f'work_dirs/{name}'
resume_from = f'checkpoints/{name}/latest.pth'
kontext_model = '/mnt/afs_gaochengmin/checkpoints/FLUX.1-Kontext-dev'
kontext_transformer = f'{kontext_model}/transformer/diffusion_pytorch_model.safetensors.index.json'

model = dict(
    vae=dict(from_pretrained=kontext_model),
    text_encoder=dict(from_pretrained=kontext_model),
    diffusion=dict(denoising=dict(pretrained=kontext_transformer)),
    teacher=dict(denoising=dict(pretrained=kontext_transformer)),
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
    train_dataloader=dict(samples_per_gpu=1),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
)
