_base_ = ['./editflux_uedit_fixedeps_2nfe_k16_data_step2_dino_gan.py']

# 70% oss_edit banana-task pairs + 30% pico-banana-400k.
# Child datasets share ImageEdit kontext resize (preferred-resolution buckets).

pico_root = '/path/to/data/pico-banana-400k'
oss_root = '/path/to/data/oss_edit'
name = 'gmkontext_uedit_fixedeps_k16_2nfe_oss70_pico30_step2_dino_gan'
work_dir = f'work_dirs/{name}'

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
                resize_mode='kontext',
                image_size=1024,
            ),
            dict(
                type='ImageEdit',
                data_root=pico_root,
                jsonl_path='jsonl/sft_with_local_source_image_path.jsonl',
                edited_images_dir='edited_images',
                image_size=1024,
                require_edited=True,
                resize_mode='kontext',
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

# Same ImgEdit-Bench vis subset as klein alph_dino_gan (9 cats x 5, seed=42).
# Kontext buckets instead of flux2.
sample_eval = dict(
    _delete_=True,
    type='EditFlowSampleImagesHook',
    enabled=True,
    dataset=dict(
        type='ImgEditBenchSample',
        annotations_path=(
            ''
            'evaluation/imgedit_bench/annotations/basic_edit.json'),
        bench_root='/path/to/data/imgedit/benchmark/Benchmark',
        categories=[
            'action', 'add', 'adjust', 'background', 'compose',
            'extract', 'remove', 'replace', 'style'],
        samples_per_category=5,
        seed=42,
        resize_mode='kontext',
    ),
    interval=10,
    must_save_interval=0,
    output_dir='samples',
    max_samples=None,
    priority='LOW',
)
