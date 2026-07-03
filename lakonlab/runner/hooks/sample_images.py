# Copyright (c) 2026 ArcFlow contributors

import os.path as osp
import re

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import DataContainer
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import get_dist_info
from torchvision.utils import save_image


def _dataloader_batch_size(dataloader):
    batch_size = getattr(dataloader, 'batch_size', None)
    if batch_size is None or batch_size <= 0:
        batch_size = 1
    return int(batch_size)


def _max_val_steps(dataloader, max_samples):
    batch_size = _dataloader_batch_size(dataloader)
    return max(1, (int(max_samples) + batch_size - 1) // batch_size)


def _should_sample(hook, runner, interval, must_save_interval, save_last):
    if interval > 0 and hook.every_n_iters(runner, interval):
        return True
    if must_save_interval > 0 and hook.every_n_iters(runner, must_save_interval):
        return True
    if save_last and hook.is_last_iter(runner):
        return True
    return False


def _unwrap_dc(value):
    if isinstance(value, DataContainer):
        return value.data
    if isinstance(value, dict):
        return {k: _unwrap_dc(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap_dc(v) for v in value)
    return value


def _slugify(text, max_len=48):
    text = str(text).strip().lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    if not text:
        text = 'prompt'
    return text[:max_len]


@HOOKS.register_module()
class ArcFlowSampleImagesHook(Hook):
    """Generate and save images with the current student (EMA) weights.

    Triggered on the same iterations as checkpoint saving. This hook only saves
    images; it does not compute FID/IS or other metrics.
    """

    def __init__(
            self,
            dataloader,
            interval=500,
            must_save_interval=1000000000,
            save_last=True,
            output_dir='samples',
            max_samples=8,
            nrow=4,
            padding=2,
            save_grid=True,
            save_individual=True,
            start_iter=0):
        self.dataloader = dataloader
        self.interval = interval
        self.must_save_interval = must_save_interval
        self.save_last = save_last
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.nrow = nrow
        self.padding = padding
        self.save_grid = save_grid
        self.save_individual = save_individual
        self.start_iter = start_iter

    def after_train_iter(self, runner):
        if runner.iter < self.start_iter:
            return
        if not _should_sample(
                self, runner, self.interval, self.must_save_interval, self.save_last):
            return

        rank, world_size = get_dist_info()
        if world_size > 1:
            dist.barrier()
        self._sample_and_save(runner, rank)
        if world_size > 1:
            dist.barrier()

    def _sample_and_save(self, runner, rank):
        iter_tag = runner.iter + 1
        out_dir = osp.join(runner.work_dir, self.output_dir, f'iter_{iter_tag}')
        if rank == 0:
            mmcv.mkdir_or_exist(out_dir)

        runner.model.eval()
        pred_imgs = []
        names = []
        saved = 0
        max_steps = _max_val_steps(self.dataloader, self.max_samples)

        with torch.no_grad():
            for step_idx, data_batch in enumerate(self.dataloader):
                if step_idx >= max_steps:
                    break

                # FSDP forward requires every rank to enter val_step together.
                outputs = runner.model.val_step(data_batch)
                if rank != 0:
                    continue

                batch_imgs = outputs['pred_imgs']
                batch_size = min(batch_imgs.size(0), self.max_samples - saved)
                batch_imgs = batch_imgs[:batch_size].detach().float().cpu().clamp(0, 1)

                data = _unwrap_dc(data_batch)
                batch_names = data.get('name', [f'sample_{saved + i}' for i in range(batch_size)])
                if not isinstance(batch_names, list):
                    batch_names = [batch_names] * batch_size

                for i in range(batch_size):
                    name = batch_names[i] if i < len(batch_names) else f'sample_{saved + i}'
                    names.append(str(name))
                    pred_imgs.append(batch_imgs[i])
                    if self.save_individual:
                        img_path = osp.join(
                            out_dir, f'{saved:03d}_{_slugify(name)}.png')
                        save_image(batch_imgs[i], img_path)
                    saved += 1
                    if saved >= self.max_samples:
                        break

        if rank == 0:
            if self.save_grid and pred_imgs:
                grid = torch.stack(pred_imgs, dim=0)
                save_image(
                    grid,
                    osp.join(out_dir, 'grid.png'),
                    nrow=min(self.nrow, len(pred_imgs)),
                    padding=self.padding)

            if names:
                with open(osp.join(out_dir, 'prompts.txt'), 'w', encoding='utf-8') as f:
                    for idx, name in enumerate(names):
                        f.write(f'{idx:03d}\t{name}\n')

            runner.logger.info(
                f'Saved {len(pred_imgs)} sample image(s) to {out_dir}')
        runner.model.train()


def _batch_item(value, index):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


@HOOKS.register_module()
class EditFlowSampleImagesHook(Hook):
    """Save edit validation samples as one folder per example.

    Each folder under ``iter_<N>/`` contains:
      - src.png     source / reference image
      - pred.png    current model output
      - target.png  dataset ground-truth edited image
      - prompt.txt  editing instruction
    """

    def __init__(
            self,
            dataloader,
            interval=500,
            must_save_interval=1000000000,
            save_last=True,
            output_dir='samples',
            max_samples=8,
            start_iter=0):
        self.dataloader = dataloader
        self.interval = interval
        self.must_save_interval = must_save_interval
        self.save_last = save_last
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.start_iter = start_iter

    def after_train_iter(self, runner):
        if runner.iter < self.start_iter:
            return
        if not _should_sample(
                self, runner, self.interval, self.must_save_interval, self.save_last):
            return

        rank, world_size = get_dist_info()
        if world_size > 1:
            dist.barrier()
        self._sample_and_save(runner, rank)
        if world_size > 1:
            dist.barrier()

    def _sample_and_save(self, runner, rank):
        iter_tag = runner.iter + 1
        out_dir = osp.join(runner.work_dir, self.output_dir, f'iter_{iter_tag}')
        if rank == 0:
            mmcv.mkdir_or_exist(out_dir)

        runner.model.eval()
        saved = 0
        max_steps = _max_val_steps(self.dataloader, self.max_samples)

        with torch.no_grad():
            for step_idx, data_batch in enumerate(self.dataloader):
                if step_idx >= max_steps:
                    break

                # FSDP forward requires every rank to enter val_step together.
                outputs = runner.model.val_step(data_batch)
                if rank != 0:
                    continue

                pred_imgs = outputs['pred_imgs'].detach().float().cpu().clamp(0, 1)

                data = _unwrap_dc(data_batch)
                batch_size = pred_imgs.size(0)
                batch_names = data.get('name', [f'sample_{saved + i}' for i in range(batch_size)])
                if not isinstance(batch_names, list):
                    batch_names = [batch_names] * batch_size

                source_imgs = data.get('source_images')
                target_imgs = data.get('edited_images')

                for i in range(batch_size):
                    if saved >= self.max_samples:
                        break

                    sample_dir = osp.join(out_dir, f'{saved:03d}')
                    mmcv.mkdir_or_exist(sample_dir)

                    prompt = str(batch_names[i] if i < len(batch_names) else f'sample_{saved}')
                    with open(osp.join(sample_dir, 'prompt.txt'), 'w', encoding='utf-8') as f:
                        f.write(prompt)

                    src = _batch_item(source_imgs, i)
                    if src is not None:
                        save_image(src.detach().float().cpu().clamp(0, 1), osp.join(sample_dir, 'src.png'))

                    save_image(pred_imgs[i], osp.join(sample_dir, 'pred.png'))

                    target = _batch_item(target_imgs, i)
                    if target is not None:
                        save_image(target.detach().float().cpu().clamp(0, 1), osp.join(sample_dir, 'target.png'))

                    saved += 1

        if rank == 0:
            runner.logger.info(
                f'Saved {saved} edit sample folder(s) to {out_dir}')
        runner.model.train()
