_base_ = ['./editqwen_2nfe_k16_data.py']

# `train_qwen_data.sh`
# Vanilla ArcFlowImitation (policy_type='ArcFlow'), NO uedit / fixedeps / alpha / GAN.
# Only the training data changes: default 30% oss_edit + 70% pico-banana-400k.
# Override at launch with OSS_PROB / PICO_PROB.
# Child datasets share ImageEdit qwen resize (not Kontext buckets).

pico_root = '/mnt/afs_gaochengmin/data/pico-banana-400k'
oss_root = '/mnt/afs_gaochengmin/data/oss_edit'
name = 'gmqwen_k16_2nfe_oss30_pico70_data'
work_dir = f'work_dirs/{name}'
resume_from = None

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
        type='ImgEditBenchSample',
        annotations_path=(
            '/mnt/afs_gaochengmin/projects/zhangyunzhe/EditFlow_8.17/EditFlow/'
            'evaluation/imgedit_bench/annotations/basic_edit.json'),
        bench_root='/mnt/afs_gaochengmin/data/imgedit/benchmark/Benchmark',
        categories=[
            'action', 'add', 'adjust', 'background', 'compose',
            'extract', 'remove', 'replace', 'style'],
        samples_per_category=5,
        seed=42,
        resize_mode='qwen',
    ),
    interval=10,
    must_save_interval=0,
    output_dir='samples',
    max_samples=None,
    priority='LOW',
)
