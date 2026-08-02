# Copyright (c) 2025 Hansheng Chen

import mmcv
import torch

from copy import deepcopy
try:
    from torch.distributed.fsdp import FSDPModule, FullyShardedDataParallel
except ImportError:
    FSDPModule = None
    FullyShardedDataParallel = None
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS
from mmgen.core import ExponentialMovingAverageHook
from lakonlab.utils import rgetattr, rhasattr


def get_ori_key(key):
    ori_key = key.split('.')
    if ori_key[0].endswith('_ema'):
        ori_key[0] = ori_key[0][:-4]
    elif ori_key[0].endswith('_ema2'):
        ori_key[0] = ori_key[0][:-5]
    else:
        raise ValueError(
            f'Invalid module key {key}, it should be in the format of '
            '<module_name>_ema or <module_name>_ema2, but got {ori_key[0]}')
    ori_key = '.'.join(ori_key)
    return ori_key


@HOOKS.register_module()
class ExponentialMovingAverageHookMod(ExponentialMovingAverageHook):

    _registered_momentum_updaters = ['rampup', 'fixed', 'karras']

    def __init__(self,
                 module_keys,
                 trainable_only=True,
                 interp_mode='lerp',
                 interp_cfg=None,
                 interval=-1,
                 start_iter=0,
                 momentum_policy='fixed',
                 momentum_cfg=None):
        super(ExponentialMovingAverageHook, self).__init__()
        self.trainable_only = trainable_only
        # check args
        assert interp_mode in self._registered_interp_funcs, (
            'Supported '
            f'interpolation functions are {self._registered_interp_funcs}, '
            f'but got {interp_mode}')

        assert momentum_policy in self._registered_momentum_updaters, (
            'Supported momentum policy are'
            f'{self._registered_momentum_updaters},'
            f' but got {momentum_policy}')

        assert isinstance(module_keys, str) or mmcv.is_tuple_of(
            module_keys, str)
        self.module_keys = (module_keys, ) if isinstance(module_keys,
                                                         str) else module_keys
        # sanity check for the format of module keys
        for k in self.module_keys:
            module_name = k.split('.')[0]
            assert module_name.endswith('_ema') or module_name.endswith('_ema2')
        self.interp_mode = interp_mode
        self.interp_cfg = dict() if interp_cfg is None else deepcopy(
            interp_cfg)
        self.interval = interval
        self.start_iter = start_iter

        assert hasattr(
            self, interp_mode
        ), f'Currently, we do not support {self.interp_mode} for EMA.'
        self.interp_func = getattr(self, interp_mode)

        self.momentum_cfg = dict() if momentum_cfg is None else deepcopy(
            momentum_cfg)
        self.momentum_policy = momentum_policy
        if momentum_policy != 'fixed':
            assert hasattr(
                self, momentum_policy
            ), f'Currently, we do not support {self.momentum_policy} for EMA.'
            self.momentum_updater = getattr(self, momentum_policy)

    def karras(self, runner, gamma=7.0, max_momentum=1.0):
        t = max(runner.iter + 1 - self.start_iter, 1)
        ema_beta = min((1 - 1 / t) ** (gamma + 1), max_momentum)
        return dict(momentum=ema_beta)

    @staticmethod
    def _is_fsdp_module(module):
        if FullyShardedDataParallel is not None and isinstance(
                module, FullyShardedDataParallel):
            return True
        return FSDPModule is not None and isinstance(module, FSDPModule)

    def _unwrap_non_fsdp_module(self, module):
        """Remove DDP-style wrappers while preserving FSDP contexts."""
        while is_module_wrapper(module) and not self._is_fsdp_module(module):
            module = module.module
        return module

    def _update_module_pair(
            self, net, ema, runner, interp_cfg):
        ema_params = dict(ema.named_parameters())
        matched_params = 0
        for name, p_net in net.named_parameters():
            p_ema = ema_params.get(name)
            if p_ema is None:
                continue
            matched_params += 1
            if self.trainable_only and not p_net.requires_grad:
                continue
            if runner.iter < self.start_iter:
                p_ema.data.copy_(p_net.data)
            else:
                p_ema.data.copy_(self.interp_func(
                    p_net, p_ema, trainable=p_net.requires_grad, **interp_cfg))
        if ema_params and matched_params == 0:
            raise RuntimeError(
                'EMA update matched zero parameters. Check whether the online '
                'and EMA modules have inconsistent wrapper prefixes.')

        ema_buffers = dict(ema.named_buffers())
        for name, b_net in net.named_buffers():
            b_ema = ema_buffers.get(name)
            if b_ema is not None:
                b_ema.data.copy_(b_net.data)

    def _snapshot_online_tensors(self, net):
        """CPU copies of tensors needed for EMA (keeps GPU peak to one unshard)."""
        param_snaps = {}
        trainable_flags = {}
        for name, p_net in net.named_parameters():
            if self.trainable_only and not p_net.requires_grad:
                continue
            param_snaps[name] = p_net.detach().to('cpu', copy=True)
            trainable_flags[name] = bool(p_net.requires_grad)
        buffer_snaps = {
            name: b.detach().to('cpu', copy=True)
            for name, b in net.named_buffers()
        }
        return param_snaps, trainable_flags, buffer_snaps

    def _apply_snapshots_to_ema(
            self, ema, param_snaps, trainable_flags, buffer_snaps, runner,
            interp_cfg):
        ema_params = dict(ema.named_parameters())
        matched_params = 0
        for name, p_net_cpu in param_snaps.items():
            p_ema = ema_params.get(name)
            if p_ema is None:
                continue
            matched_params += 1
            p_net = p_net_cpu.to(device=p_ema.device, dtype=p_ema.dtype)
            trainable = trainable_flags[name]
            if runner.iter < self.start_iter:
                p_ema.data.copy_(p_net)
            else:
                p_ema.data.copy_(self.interp_func(
                    p_net, p_ema, trainable=trainable, **interp_cfg))
        if param_snaps and matched_params == 0:
            raise RuntimeError(
                'EMA update matched zero parameters. Check whether the online '
                'and EMA modules have inconsistent wrapper prefixes.')

        ema_buffers = dict(ema.named_buffers())
        for name, b_cpu in buffer_snaps.items():
            b_ema = ema_buffers.get(name)
            if b_ema is not None:
                b_ema.data.copy_(
                    b_cpu.to(device=b_ema.device, dtype=b_ema.dtype))

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        with torch.no_grad():
            model = runner.model.module if is_module_wrapper(
                runner.model) else runner.model

            # update momentum
            _interp_cfg = deepcopy(self.interp_cfg)
            if self.momentum_policy != 'fixed':
                _updated_args = self.momentum_updater(runner, **self.momentum_cfg)
                _interp_cfg.update(_updated_args)

            for key in self.module_keys:
                net = rgetattr(model, get_ori_key(key))
                ema = rgetattr(model, key)
                # The custom DDP wrapper wraps trainable children (e.g.
                # ``diffusion``) independently but leaves ``diffusion_ema``
                # unwrapped. Unwrap those children so both sides expose the
                # same parameter names; otherwise every EMA lookup silently
                # misses due to the student's ``module.`` prefix.
                net = self._unwrap_non_fsdp_module(net)
                ema = self._unwrap_non_fsdp_module(ema)
                net_fsdp = self._is_fsdp_module(net)
                ema_fsdp = self._is_fsdp_module(ema)
                if net_fsdp or ema_fsdp:
                    assert FullyShardedDataParallel is not None
                    # Sequential summon: holding net+ema fully unsharded at once
                    # OOMs on Qwen+GAN (~80GB). Snapshot online tensors to CPU,
                    # reshard, then write into EMA under a separate summon.
                    if net_fsdp:
                        with FullyShardedDataParallel.summon_full_params(
                                net, writeback=False, rank0_only=False):
                            param_snaps, trainable_flags, buffer_snaps = (
                                self._snapshot_online_tensors(net))
                    else:
                        param_snaps, trainable_flags, buffer_snaps = (
                            self._snapshot_online_tensors(net))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if ema_fsdp:
                        with FullyShardedDataParallel.summon_full_params(
                                ema, writeback=True, rank0_only=False):
                            self._apply_snapshots_to_ema(
                                ema, param_snaps, trainable_flags,
                                buffer_snaps, runner, _interp_cfg)
                    else:
                        self._apply_snapshots_to_ema(
                            ema, param_snaps, trainable_flags, buffer_snaps,
                            runner, _interp_cfg)
                    del param_snaps, trainable_flags, buffer_snaps
                else:
                    self._update_module_pair(net, ema, runner, _interp_cfg)

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model
        # sanity check for ema model
        for k in self.module_keys:
            if not rhasattr(model, k):
                raise RuntimeError(
                    f'Cannot find {k} network for EMA hook.')
