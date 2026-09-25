data_root = '/mnt/afs_gaochengmin/data/pico-banana-400k'

# Data-based distillation: x0 = edited target latent (forward-diffused + noise).
# Rows without a valid edited image on disk are skipped.
data = dict(
    train=dict(
        type='ImageEdit',
        data_root=data_root,
        jsonl_path='jsonl/sft_with_local_source_image_path.jsonl',
        edited_images_dir='edited_images',
        image_size=1024,
        require_edited=True,
        end_ind=-128),
    val=dict(
        type='ImageEdit',
        data_root=data_root,
        jsonl_path='jsonl/sft_with_local_source_image_path.jsonl',
        edited_images_dir='edited_images',
        image_size=1024,
        start_ind=-128,
        repeat=2,
        test_mode=True,
    ),
)
