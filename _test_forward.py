import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
torch.cuda.set_device(rank)

from mmcv import Config
from mmgen.models import build_model
import lakonlab.models
from lakonlab.parallel.fsdp_wrapper import FSDPWrapper

cfg = Config.fromfile(
    'configs/kontext/editflux_uedit_fixedeps_2nfe_k16_data_split_stage_dual_lora_dino_gan.py')
model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
wrapped = FSDPWrapper(model, device_id=rank, **cfg.fsdp_kwargs)
d = wrapped.module.diffusion
d.train()
lin = d.denoising.time_text_embed.timestep_embedder.linear_1
lora_a = lin.lora_A['step1'].weight
print(f'rank{rank} before', lora_a.shape, lora_a.dtype)
dt = torch.bfloat16
bs, dev = 1, 'cuda'
x_0 = torch.randn(bs, 64, 64, 64, device=dev, dtype=dt)
kw = dict(
    encoder_hidden_states=torch.randn(bs, 512, 4096, device=dev, dtype=dt),
    pooled_projections=torch.randn(bs, 768, device=dev, dtype=dt),
    guidance=torch.ones(bs, device=dev, dtype=dt),
    image_latents=torch.randn(bs, 64, 64, 64, device=dev, dtype=dt),
    x_ref=torch.randn(bs, 64, 64, 64, device=dev, dtype=dt),
)
try:
    loss, _ = d(x_0, return_loss=True, running_status=dict(iteration=0), **kw)
    if rank == 0:
        print('forward ok', float(loss))
except Exception as e:
    print(f'rank{rank} fail', e)
dist.destroy_process_group()
