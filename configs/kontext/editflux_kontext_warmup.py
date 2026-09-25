_base_ = ['./editflux_uedit_fixedeps_2nfe_k16_alpha_data.py']

# SpurFlow warmup. Sigmoid alpha, 2-NFE, no GAN.
# One paired edit dataset. Paths are overridden by train_flux_kontext_warmup.sh.
# Student is cond-only. Teacher uses distilled CFG.

data_root = '/path/to/dataset'
kontext_model = '/path/to/checkpoints/FLUX.1-Kontext-dev'
kontext_transformer = f'{kontext_model}/transformer/diffusion_pytorch_model.safetensors.index.json'
name = 'spurflow_warmup'
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
        type='PairEdit',
        data_root=data_root,
        jsonl_path='metadata.jsonl',
        require_edited=True,
        resize_mode='kontext',
        image_size=1024,
        load_unpaired_edited=False,
        end_ind=-128,
    ),
    val=dict(
        _delete_=True,
        type='PairEdit',
        data_root=data_root,
        jsonl_path='metadata.jsonl',
        image_size=1024,
        resize_mode='kontext',
        start_ind=-128,
        repeat=2,
        test_mode=True,
    ),
)

sample_eval = dict(
    _delete_=True,
    type='SpurFlowSampleImagesHook',
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
