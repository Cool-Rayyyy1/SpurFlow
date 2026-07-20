# Copyright (c) 2025 Hansheng Chen

import os
import os.path as osp
import logging
import time
import tempfile
import subprocess
import uuid
import shutil
import re
import torch
import torch.nn as nn
import torch.distributed as dist
import mmcv
from contextlib import nullcontext
from typing import Union, Callable, Optional, List, Sequence, Tuple
from collections import OrderedDict
from tempfile import TemporaryDirectory
from torch.optim import Optimizer
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions, get_optimizer_state_dict
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullyShardedDataParallel as FSDP
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:  # pragma: no cover
    FSDPModule = None
from safetensors.torch import load_file, load
from diffusers.utils.hub_utils import _get_checkpoint_shard_files
from mmcv.runner import CheckpointLoader, get_dist_info, _load_checkpoint
from mmcv.parallel import is_module_wrapper
from lakonlab.utils import download_from_huggingface, rgetattr
from lakonlab.utils.io_utils import S3Backend, TMP_DIR
from lakonlab.parallel import FSDP2Wrapper

_TRAINABLE_STATE_MARKERS = (
    'lora',
    'proj_out_means',
    'proj_out_deltax',
    'proj_out_epsilon',
    'proj_out_logweights',
    'proj_out_loggamma',
    'proj_out_alpha',
    'norm_out',
)


def _is_trainable_state_key(key: str) -> bool:
    if not (key.startswith('diffusion.') or key.startswith('diffusion_ema.')):
        return False
    return any(marker in key for marker in _TRAINABLE_STATE_MARKERS)


def _is_redundant_non_ema_key(key: str) -> bool:
    return key.startswith('diffusion.') and not key.startswith('diffusion_ema.')


_DUAL_STAGE_HEAD_NAMES = (
    'norm_out',
    'proj_out_deltax',
    'proj_out_logweights',
    'proj_out_loggamma',
)


def remap_single_heads_to_dual_stage_state_dict(
        state_dict: Union[dict, OrderedDict],
        target_stages: tuple = ('step1', 'step2')) -> Union[dict, OrderedDict]:
    """Duplicate shared output-head weights into dual-stage step1/step2 head bundles."""
    keys = list(state_dict.keys())
    has_dual = any('.head_stages.step1.' in k for k in keys)
    if has_dual:
        return state_dict

    metadata = getattr(state_dict, '_metadata', None)
    new_sd = OrderedDict(state_dict)
    keys_to_drop = []
    for key, value in state_dict.items():
        for head_name in _DUAL_STAGE_HEAD_NAMES:
            needle = f'.denoising.{head_name}.'
            if needle not in key:
                continue
            keys_to_drop.append(key)
            for stage in target_stages:
                new_key = key.replace(
                    needle, f'.denoising.head_stages.{stage}.{head_name}.')
                new_sd[new_key] = value
            break
    for key in keys_to_drop:
        new_sd.pop(key, None)
    if metadata is not None:
        new_sd._metadata = metadata  # type: ignore
    return new_sd


def remap_single_lora_to_dual_stage_state_dict(
        state_dict: Union[dict, OrderedDict],
        source_adapter: str = 'default',
        target_stages: tuple = ('step1', 'step2')) -> Union[dict, OrderedDict]:
    """Duplicate single-adapter LoRA weights into dual-stage step1/step2 adapters.

    Split-stage checkpoints store PEFT keys like ``...lora_A.default.weight``.
    Dual-stage students expect ``...lora_A.step1.weight`` and ``...lora_A.step2.weight``.
    """
    keys = list(state_dict.keys())
    has_dual = any(
        f'.lora_A.{stage}.' in k or f'.lora_B.{stage}.' in k
        for k in keys for stage in target_stages)
    has_single = any(f'.lora_A.{source_adapter}.' in k for k in keys)
    if has_dual or not has_single:
        return state_dict

    metadata = getattr(state_dict, '_metadata', None)
    new_sd = OrderedDict(state_dict)
    keys_to_drop = []
    for key, value in state_dict.items():
        for lora_side in ('lora_A', 'lora_B'):
            needle = f'.{lora_side}.{source_adapter}.'
            if needle not in key:
                continue
            keys_to_drop.append(key)
            for stage in target_stages:
                new_sd[key.replace(needle, f'.{lora_side}.{stage}.')] = value
    for key in keys_to_drop:
        new_sd.pop(key, None)
    if metadata is not None:
        new_sd._metadata = metadata  # type: ignore
    return new_sd


def _filter_checkpoint_key_warnings(
        missing_keys: List[str],
        unexpected_keys: List[str]) -> tuple[List[str], List[str]]:
    """Drop benign mismatches from trainable-only resume checkpoints."""
    missing_keys = [k for k in missing_keys if _is_trainable_state_key(k)]
    unexpected_keys = [
        k for k in unexpected_keys
        if not (
            k.endswith('position_ids')
            or k.startswith('text_encoder.')
            or _is_redundant_non_ema_key(k)
        )
    ]
    return missing_keys, unexpected_keys


def fsdp_load_full_state_dict(module: FSDP,
                              full_sd: dict,
                              prefix: str,
                              fsdp_missing_keys: List[str],
                              err_msg: List[str]) -> None:
    flat_p = module._flat_param
    if flat_p is not None:
        for param_info, shape, shard_info in zip(flat_p._param_infos, flat_p._shapes, flat_p._shard_param_infos):
            k = f'{prefix}{param_info.module_name}.{param_info.param_name}'
            if k not in full_sd:
                fsdp_missing_keys.append(k)
                continue
            t = full_sd[k]
            if t.shape != shape:
                err_msg.append(
                    f'size mismatch for {k}: copying a param with shape {t.shape} from checkpoint, '
                    f'the shape in current model is {shape}.')
            elif shard_info.in_shard:
                flat_p[shard_info.offset_in_shard:shard_info.offset_in_shard + shard_info.numel_in_shard].data.copy_(
                    t.view(-1)[shard_info.intra_param_start_idx:shard_info.intra_param_end_idx + 1])
            del full_sd[k]


def load_full_state_dict(module: nn.Module,
                         state_dict: Union[dict, OrderedDict],
                         strict: bool = False,
                         logger: Optional[logging.Logger] = None,
                         assign: bool = False) -> None:
    unexpected_keys: List[str] = []
    all_missing_keys: List[str] = []
    err_msg: List[str] = []
    fsdp_missing_keys: List[str] = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()  # type: ignore
    if metadata is not None:
        state_dict._metadata = metadata  # type: ignore

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # Load full states to sharded FSDP1
        if isinstance(module, FSDP):
            fsdp_load_full_state_dict(
                module, state_dict, prefix, fsdp_missing_keys, err_msg)
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        if assign:
            local_metadata['assign_to_params_buffers'] = assign
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    # break load->load reference cycle
    load = None  # type: ignore

    # ignore "num_batches_tracked" of BN layers
    missing_keys = fsdp_missing_keys.copy()
    for key in all_missing_keys:
        if not rgetattr(module, key + '._fsdp_flattened', False):
            missing_keys.append(key)

    missing_keys = [
        key for key in missing_keys if 'num_batches_tracked' not in key
    ]
    missing_keys, unexpected_keys = _filter_checkpoint_key_warnings(
        missing_keys, unexpected_keys)

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)  # type: ignore
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def exists_ckpt(filename):
    if not filename:
        return False
    loader_name = CheckpointLoader._get_checkpoint_loader(filename).__name__[10:]
    if loader_name == 'local':
        return os.path.exists(filename)
    elif loader_name == 'tmp':
        src_file = filename[4:]
        return os.path.exists(src_file)
    elif loader_name == 's3':
        return S3Backend().exists(filename)
    else:
        raise NotImplementedError()


@CheckpointLoader.register_scheme(prefixes='s3://', force=True)
def load_from_s3(filename, map_location=None):
    ext = os.path.splitext(filename)[-1].lower()

    rank, ws = get_dist_info()
    if ws > 1:
        local_rank = dist.get_node_local_rank()
    else:
        local_rank = 0

    # get the temporary file path
    if rank == 0:
        tmp_file = os.path.join(TMP_DIR, str(uuid.uuid4()) + ext)
    else:
        tmp_file = None
    if ws > 1:
        object_list = [tmp_file]
        dist.broadcast_object_list(object_list, src=0)
        tmp_file = object_list[0]

    # download the file to temp dir
    if local_rank == 0:
        cmd = ['aws', 's3', 'cp', filename, tmp_file]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    if ws > 1:
        dist.barrier()

    # load the temporary file
    if ext == '.txt':
        # if the file is a text file, it contains the name to the actual checkpoint file
        with open(tmp_file, 'r') as f:
            _filename = f.read().strip()
        filename = os.path.join(
            os.path.dirname(filename), _filename)  # get the actual checkpoint file path
        # remove the temporary file
        if ws > 1:
            dist.barrier()
        if local_rank == 0:
            os.remove(tmp_file)
        return load_from_s3(filename, map_location=map_location)

    elif ext == '.safetensors':
        ckpt = load_file(tmp_file, device=map_location)
    else:
        ckpt = torch.load(tmp_file, map_location=map_location)

    # remove the temporary file
    if ws > 1:
        dist.barrier()
    if local_rank == 0:
        os.remove(tmp_file)

    return ckpt


@CheckpointLoader.register_scheme(prefixes='tmp:')
def load_from_tmp(filename, map_location=None):
    src_file = filename[4:]
    assert os.path.exists(src_file)
    ext = os.path.splitext(src_file)[-1].lower()
    rank, ws = get_dist_info()
    if ws > 1:
        local_rank = dist.get_node_local_rank()
    else:
        local_rank = 0

    # get the temporary file path
    if rank == 0:
        tmp_file = os.path.join(TMP_DIR, str(uuid.uuid4()) + ext)
    else:
        tmp_file = None
    if ws > 1:
        object_list = [tmp_file]
        dist.broadcast_object_list(object_list, src=0)
        tmp_file = object_list[0]

    # copy the file to temp dir
    if local_rank == 0:
        shutil.copy(src_file, tmp_file)
    if ws > 1:
        dist.barrier()

    # load the temporary file
    if ext == '.safetensors':
        ckpt = load_file(tmp_file, device=map_location)
    else:
        ckpt = torch.load(tmp_file, map_location=map_location)

    # remove the temporary file
    if ws > 1:
        dist.barrier()
    if local_rank == 0:
        os.remove(tmp_file)

    return ckpt


@CheckpointLoader.register_scheme(prefixes='huggingface://')
def load_from_huggingface(filename, map_location=None):
    cached_file = download_from_huggingface(filename)
    if cached_file.endswith('.index.json'):  # sharded checkpoint
        filename = filename.replace('huggingface://', '').split('/')
        repo_id = '/'.join(filename[:2])
        repo_subfolder = '/'.join(filename[2:-1])
        is_dist = dist.is_available() and dist.is_initialized()
        if is_dist:
            local_rank = dist.get_node_local_rank()
        else:
            local_rank = 0
        if local_rank == 0:
            sharded_cached_files = _get_checkpoint_shard_files(
                repo_id,
                cached_file,
                subfolder=repo_subfolder)[0]
        if is_dist:
            dist.barrier()
        if local_rank > 0:
            sharded_cached_files = _get_checkpoint_shard_files(
                repo_id,
                cached_file,
                subfolder=repo_subfolder)[0]
        ckpt = OrderedDict()
        for sharded_cached_file in sharded_cached_files:
            ext = os.path.splitext(sharded_cached_file)[-1].lower()
            if ext == '.safetensors':
                ckpt.update(load_file(sharded_cached_file, device=map_location))
            else:
                ckpt.update(torch.load(sharded_cached_file, map_location=map_location))
        return ckpt
    else:
        ext = os.path.splitext(cached_file)[-1].lower()
        if ext == '.safetensors':
            return load_file(cached_file, device=map_location)
        else:
            return torch.load(cached_file, map_location=map_location)


@CheckpointLoader.register_scheme(prefixes='', force=True)
def load_from_local(filename, map_location=None):
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    ext = os.path.splitext(filename)[-1].lower()
    if filename.endswith('.safetensors.index.json'):
        import json
        index_dir = osp.dirname(filename)
        with open(filename) as f:
            index = json.load(f)
        ckpt = OrderedDict()
        for shard_file in sorted(set(index.get('weight_map', {}).values())):
            shard_path = osp.join(index_dir, shard_file)
            ckpt.update(load_file(shard_path, device=map_location))
        return ckpt
    if ext == '.safetensors':
        with open(filename, "rb") as f:  # load_file may fail with FUSE/NFS mmap
            ckpt = load(f.read())
            if map_location is not None:
                for k in ckpt:
                    ckpt[k] = ckpt[k].to(map_location)
    else:
        ckpt = torch.load(filename, map_location=map_location)
    return ckpt


def promote_non_ema_to_ema_state_dict(
        state_dict: Union[dict, OrderedDict],
        ema_module_keys: Sequence[str] = ('diffusion_ema',)) -> OrderedDict:
    """Duplicate ``<ori>.*`` weights into ``<ema>.*`` when the source lacks EMA keys.

    Trainable-only alpha / fixed-eps checkpoints often store student weights under
    ``diffusion.*`` only. Training configs with ``diffusion_use_ema=True`` still
    instantiate a separate ``diffusion_ema`` module that must be populated at load.
    """
    state_dict = OrderedDict(state_dict)
    promoted_any = False
    for ema_key in ema_module_keys:
        if ema_key.endswith('_ema'):
            ori_key = ema_key[:-4]
        elif ema_key.endswith('_ema2'):
            ori_key = ema_key[:-5]
        else:
            continue
        ema_prefix = f'{ema_key}.'
        ori_prefix = f'{ori_key}.'
        if any(k.startswith(ema_prefix) for k in state_dict):
            continue
        promoted = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith(ori_prefix):
                promoted[ema_prefix + key[len(ori_prefix):]] = value
        if promoted:
            state_dict.update(promoted)
            promoted_any = True
    state_dict._promoted_non_ema_to_ema = promoted_any  # type: ignore[attr-defined]
    return state_dict


def _model_expects_dual_stage_remap(model: nn.Module) -> bool:
    """True when the student denoiser uses split-stage step1/step2 LoRA adapters."""
    if is_module_wrapper(model):
        model = model.module
    for module_name in ('diffusion', 'diffusion_ema'):
        submodule = getattr(model, module_name, None)
        if submodule is None:
            continue
        denoising = getattr(submodule, 'denoising', None)
        if denoising is not None and getattr(denoising, 'dual_stage_lora', False):
            return True
    return False


def load_checkpoint(model: torch.nn.Module,
                    filename: str,
                    map_location: Union[str, Callable, None] = None,
                    strict: bool = False,
                    logger: Optional[logging.Logger] = None,
                    revise_keys: list = [(r'^module\.', '')],
                    assign: bool = False) -> Union[dict, OrderedDict]:
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    state_dict = promote_non_ema_to_ema_state_dict(state_dict)
    if getattr(state_dict, '_promoted_non_ema_to_ema', False) and logger is not None:
        logger.info(
            'Checkpoint lacks diffusion_ema.* keys; promoted matching diffusion.* '
            'weights for EMA modules.')

    dual_stage_remap = _model_expects_dual_stage_remap(model)
    if dual_stage_remap:
        state_dict = remap_single_lora_to_dual_stage_state_dict(state_dict)
        state_dict = remap_single_heads_to_dual_stage_state_dict(state_dict)
        if logger is not None and any(
                f'.lora_A.{stage}.' in k or f'.head_stages.{stage}.' in k
                for k in state_dict for stage in ('step1', 'step2')):
            logger.info(
                'Expanded single LoRA / head weights to dual-stage step1/step2 modules.')

    # load state_dict
    if isinstance(model, FSDP2Wrapper):  # FSDP2
        for name, submodule in model.module._modules.items():
            submodule_state_dict = {
                k[len(name) + 1:]: v for k, v in state_dict.items() if k.startswith(name)}
            set_model_state_dict(
                model=submodule,
                model_state_dict=submodule_state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=False,
                    strict=strict))
    else:  # FSDP1, DDP, or non-distributed model
        load_full_state_dict(model, state_dict, strict, logger, assign)
    return checkpoint


def _save_to_state_dict(module, destination, prefix, keep_vars, trainable_only=False, cpu_offload=False):
    for name, param in module._parameters.items():
        if param is not None and (not trainable_only or param.requires_grad):
            if not keep_vars:
                param = param.detach()
            if isinstance(param, DTensor):
                param = param.full_tensor()
                if torch.distributed.get_rank() == 0:  # only save the full tensor on rank 0
                    if cpu_offload:
                        param = param.cpu()
                    destination[prefix + name] = param
            else:
                if cpu_offload:
                    param = param.cpu()
                destination[prefix + name] = param
    for name, buf in module._buffers.items():
        if buf is not None:
            if not keep_vars:
                buf = buf.detach()
            if cpu_offload:
                buf = buf.cpu()
            destination[prefix + name] = buf


def get_state_dict(module,
                   destination=None,
                   prefix='',
                   keep_vars=False,
                   trainable_only=False,
                   cpu_offload=True):
    if isinstance(module, FSDP):  # FSDP1
        if trainable_only and len(module.params) == 1 and not module.params[0].requires_grad:
            return destination  # skip frozen module

        with FSDP.state_dict_type(
                module,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=cpu_offload, rank0_only=True)):
            module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    else:  # FSDP2, DDP, or non-distributed model
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module

        # below is the same as torch.nn.Module.state_dict() except for the trainable_only argument
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()  # type: ignore

        local_metadata = dict(version=module._version)
        if hasattr(destination, '_metadata'):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in module._state_dict_pre_hooks.values():
            hook(module, prefix, keep_vars)
        _save_to_state_dict(
            module, destination, prefix, keep_vars, trainable_only=trainable_only, cpu_offload=cpu_offload)
        for name, child in module._modules.items():
            if child is not None:
                get_state_dict(
                    child, destination, prefix + name + '.',
                    keep_vars=keep_vars, trainable_only=trainable_only, cpu_offload=cpu_offload)
        for hook in module._state_dict_hooks.values():
            hook_result = hook(module, destination, prefix, local_metadata)
            if not getattr(hook, '_from_public_api', False):
                if hook_result is not None:
                    destination = hook_result
            else:
                if hook_result is not None:
                    raise RuntimeError('state_dict post-hook must return None')

    return destination


def get_optim_state_dict(model, optimizer, bf16=False):
    optim_state_dict = get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True))
    if 'state' in optim_state_dict:
        for state_name, state in optim_state_dict['state'].items():
            new_state = dict()
            for k, v in state.items():
                if bf16 and isinstance(v, torch.Tensor) and v.dtype == torch.float32 and v.numel() > 1:
                    v = v.to(dtype=torch.bfloat16)
                new_state[k] = v
            optim_state_dict['state'][state_name] = new_state
    return optim_state_dict


def _is_fsdp_module(module: nn.Module) -> bool:
    if isinstance(module, FSDP):
        return True
    return FSDPModule is not None and isinstance(module, FSDPModule)


def _unwrap_non_fsdp_module(module: nn.Module) -> nn.Module:
    """Strip DDP-style wrappers while keeping FSDP roots intact."""
    while is_module_wrapper(module) and not _is_fsdp_module(module):
        module = module.module
    return module


def _iter_ema_module_pairs(model: nn.Module) -> List[Tuple[str, str, nn.Module]]:
    """Yield ``(online_name, ema_name, ema_module)`` for top-level EMA children."""
    pairs = []
    for name, module in model.named_children():
        if name.endswith('_ema'):
            pairs.append((name[:-4], name, module))
        elif name.endswith('_ema2'):
            pairs.append((name[:-5], name, module))
    return pairs


def append_ema_mirrors_to_state_dict(
        model: nn.Module,
        state_dict: Union[dict, OrderedDict],
        cpu_offload: bool = True) -> Union[dict, OrderedDict]:
    """Copy EMA tensors that mirror already-saved online trainable keys.

    ``ckpt_trainable_only=True`` skips ``requires_grad=False`` params, and EMA
    weights are always non-trainable clones, so they never land in the
    checkpoint. Sampling / resume both rely on ``diffusion_ema``; without these
    mirrors, resume promotes noisy online weights into EMA and sample quality
    drops sharply.

    Only keys already present under the online module (e.g. LoRA / proj heads)
    are mirrored, so frozen backbones tied into EMA are not materialized.
    """
    if is_module_wrapper(model) and not _is_fsdp_module(model):
        model = model.module
    if isinstance(model, FSDP2Wrapper):
        model = model.module

    ema_pairs = _iter_ema_module_pairs(model)
    if not ema_pairs:
        return state_dict

    rank, world_size = get_dist_info()
    for online_name, ema_name, ema_mod in ema_pairs:
        online_prefix = f'{online_name}.'
        suffixes = [
            key[len(online_prefix):]
            for key in state_dict
            if key.startswith(online_prefix)
        ]
        if not suffixes:
            continue

        ema_mod = _unwrap_non_fsdp_module(ema_mod)
        if isinstance(ema_mod, FSDP):
            # All ranks must enter the collective; mirrors are only kept on rank0.
            ema_ctx = FSDP.summon_full_params(
                ema_mod, writeback=False, rank0_only=False)
        else:
            ema_ctx = nullcontext()

        with ema_ctx:
            if rank != 0 and world_size > 1:
                continue
            ema_params = dict(ema_mod.named_parameters())
            ema_buffers = dict(ema_mod.named_buffers())
            for suffix in suffixes:
                tensor = ema_params.get(suffix)
                if tensor is None:
                    tensor = ema_buffers.get(suffix)
                if tensor is None:
                    continue
                tensor = tensor.detach()
                if isinstance(tensor, DTensor):
                    tensor = tensor.full_tensor()
                if cpu_offload:
                    tensor = tensor.cpu()
                state_dict[f'{ema_name}.{suffix}'] = tensor

    return state_dict


def write_checkpoint_to_file(checkpoint, filepath, create_symlink=False, after_save_hook=None):
    if filepath.startswith('pavi://'):
        try:
            from pavi import modelcloud
            from pavi.exception import NodeNotFoundError
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filepath[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)

    elif filepath.startswith('s3://'):
        ext = os.path.splitext(filepath)[-1].lower()
        with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=ext, delete=False) as tmp:
            cached_file = tmp.name
            torch.save(checkpoint, tmp)
            tmp.flush()
        try:
            cmd = ['aws', 's3', 'cp', cached_file, filepath]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        finally:
            os.remove(cached_file)

        if create_symlink:
            # S3 does not support real symlinks, so we create a 'latest.txt'
            # containing the relative path of the latest checkpoint
            dst_file = osp.join(osp.dirname(filepath), 'latest.txt')
            S3Backend().put_text(osp.basename(filepath), dst_file)

    else:
        mmcv.mkdir_or_exist(osp.dirname(filepath))
        # immediately flush buffer
        with open(filepath, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()

        if create_symlink:
            dst_file = osp.join(osp.dirname(filepath), 'latest.pth')
            mmcv.symlink(osp.basename(filepath), dst_file)

    if after_save_hook is not None:
        after_save_hook()


def get_checkpoint(model,
                   optimizer=None,
                   loss_scaler=None,
                   meta=None,
                   trainable_only=False,
                   fp16=False,
                   fp16_ema=False,
                   bf16_optim=False):
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    state_dict = get_state_dict(
        model, trainable_only=trainable_only, cpu_offload=True)
    if trainable_only:
        # EMA params are requires_grad=False clones and would otherwise be
        # dropped; mirror the saved online trainable keys into *_ema*.
        append_ema_mirrors_to_state_dict(
            model, state_dict, cpu_offload=True)
    checkpoint = {
        'meta': meta,
        'state_dict': state_dict}
    if fp16 or fp16_ema:
        for k, v in checkpoint['state_dict'].items():
            if ((fp16 and '_ema.' not in k and '_ema2.' not in k) or (fp16_ema and ('_ema.' in k or '_ema2.' in k))) \
                    and v.dtype == torch.float32:
                checkpoint['state_dict'][k] = v.half()

    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = get_optim_state_dict(model, optimizer, bf16_optim)
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            submodule = getattr(model, name)
            checkpoint['optimizer'][name] = get_optim_state_dict(submodule, optim, bf16_optim)

    # save loss scaler for mixed-precision (FP16) training
    if loss_scaler is not None:
        checkpoint['loss_scaler'] = loss_scaler.state_dict()

    return checkpoint
