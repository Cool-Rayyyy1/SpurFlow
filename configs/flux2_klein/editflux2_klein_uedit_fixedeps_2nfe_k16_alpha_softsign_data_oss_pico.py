_base_ = ['./editflux2_klein_uedit_fixedeps_2nfe_k16_alpha_softsign_data.py']

# `train_flux2_klein_edit_fixedeps_alpha_softsign_data.sh`
# FLUX.2-klein-base Softsign-01 alpha distillation, NO GAN, non-split-stage
# (random-segment ArcFlowEditAlphaImitation, same as the pico-only baseline).
# Only the training data changes: 70% oss_edit + 30% pico-banana-400k.
#
# Teacher: official klein-base CFG=4.0 (cond + uncond, no norm rescale).
# Student: cond-only single forward.
# Child datasets share the ImageEdit flux2 resize (area-cap ~1MP, multiple-of-16).

pico_root = '/mnt/afs_gaochengmin/data/pico-banana-400k'
oss_root = '/mnt/afs_gaochengmin/data/oss_edit'
name = 'gmklein_base_uedit_fixedeps_alpha_softsign01_k16_2nfe_oss70_pico30'
work_dir = f'work_dirs/{name}'
resume_from = f'checkpoints/{name}/latest.pth'

data = dict(
    train=dict(
        _delete_=True,
        type='ProbMixDataset',
        probs=[0.7, 0.3],
        datasets=[
            dict(
                type='OssEdit',
                data_root=oss_root,
                jsonl_path='metadata.jsonl',
                require_edited=True,
                resize_mode='flux2',
                image_size=1024,
                latent_size=(32, 128, 128),
                load_unpaired_edited=False,
            ),
            dict(
                type='ImageEdit',
                data_root=pico_root,
                jsonl_path='jsonl/sft_with_local_source_image_path.jsonl',
                edited_images_dir='edited_images',
                image_size=1024,
                require_edited=True,
                resize_mode='flux2',
                latent_size=(32, 128, 128),
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
        resize_mode='flux2',
        latent_size=(32, 128, 128),
        start_ind=-128,
        repeat=2,
        test_mode=True,
    ),
)

# Training-time vis: ImgEdit-Bench (9 cats × 5) + GEdit-v2 (23 types × 2).
# Dumped as samples/iter_N/{imgedit,gedit_v2}/...
sample_eval = dict(
    _delete_=True,
    type='EditFlowSampleImagesHook',
    enabled=True,
    dataset=dict(
        type='ConcatEditValSample',
        datasets=[
            dict(
                type='ImgEditBenchSample',
                split='imgedit',
                annotations_path=(
                    '/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17/EditFlow/'
                    'evaluation/imgedit_bench/annotations/basic_edit.json'),
                bench_root='/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark',
                categories=[
                    'action', 'add', 'adjust', 'background', 'compose',
                    'extract', 'remove', 'replace', 'style'],
                samples_per_category=5,
                seed=42,
                resize_mode='flux2',
                latent_channels=32,
            ),
            dict(
                type='GEditV2Sample',
                split='gedit_v2',
                annotations_path=(
                    '/mnt/afs_caiqi/data/benchmark/GEdit_v2/gedit_v2_meta.json'),
                bench_root='/mnt/afs_caiqi/data/benchmark/GEdit_v2',
                samples_per_category=2,
                language=None,
                seed=42,
                resize_mode='flux2',
                latent_channels=32,
            ),
        ],
    ),
    interval=10,
    must_save_interval=0,
    output_dir='samples',
    max_samples=None,
    priority='LOW',
)
