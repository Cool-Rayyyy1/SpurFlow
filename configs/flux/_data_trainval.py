data_root = '/mnt/afs_zhangyunzhe/dataset/t2i_prompts_3m'

# TDM-style online prompts:
# - lazy-read prompt text from local parquet (no HF datasets, no embedding cache)
# - encode with FLUX text_encoder inside the training loop
# - fixed latent size (no bucketize scan over 3M rows at startup)
data = dict(
    train=dict(
        type='ImagePrompt',
        data_root=data_root,
        lazy_prompt=True,
        ignore_prompt_size=True,
        bucketize=False,
        end_ind=-128),
    val=dict(
        type='ImagePrompt',
        data_root=data_root,
        lazy_prompt=True,
        ignore_prompt_size=True,
        bucketize=False,
        latent_size=(16, 128, 128),
        start_ind=-128,
        repeat=2,
        test_mode=True,
    ),
)
