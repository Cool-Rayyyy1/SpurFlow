import torch
import mmcv
from collections import OrderedDict
from typing import Union
from mmgen.models import build_model
from lakonlab.runner.checkpoint import _load_checkpoint, load_full_state_dict
from lakonlab.runner.hooks.ema_hook import get_ori_key


def _ema_module_keys(config) -> list[str]:
    module_keys: list[str] = []
    for hook in config.get('custom_hooks', []):
        if hook['type'] in ('ExponentialMovingAverageHookMod', 'ExponentialMovingAverageHook'):
            if isinstance(hook['module_keys'], str):
                module_keys.append(hook['module_keys'])
            else:
                module_keys.extend(hook['module_keys'])
    return module_keys


def _drop_non_ema_state_dict_keys(state_dict, module_keys: list[str]) -> OrderedDict:
    drop_prefixes = tuple(f"{get_ori_key(key)}." for key in module_keys)
    if not drop_prefixes:
        return state_dict
    return OrderedDict(
        (key, value) for key, value in state_dict.items() if not key.startswith(drop_prefixes)
    )


def _prepare_ema_state_dict(
        state_dict: Union[dict, OrderedDict],
        module_keys: list[str]) -> OrderedDict:
    """Normalize checkpoint keys for ema_only inference.

    Training checkpoints often save trainable weights under ``diffusion.*`` only.
    Inference uses ``diffusion_ema``; promote non-EMA keys when EMA keys are absent.
    """
    state_dict = OrderedDict(state_dict)
    for ema_key in module_keys:
        ori_key = get_ori_key(ema_key)
        ema_prefix = f"{ema_key}."
        ori_prefix = f"{ori_key}."
        has_ema = any(k.startswith(ema_prefix) for k in state_dict)
        if has_ema:
            state_dict = _drop_non_ema_state_dict_keys(state_dict, [ema_key])
            continue

        promoted = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith(ori_prefix):
                promoted[ema_prefix + key[len(ori_prefix):]] = value
            elif not key.startswith(ema_prefix):
                promoted[key] = value
        state_dict = promoted
    return state_dict


def init_model(
        config, checkpoint=None, device='cuda:0', cfg_options=None,
        ema_only=True, use_fp16=False, use_bf16=False):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model = build_model(
        config.model, train_cfg=config.train_cfg, test_cfg=config.test_cfg)

    ema_module_keys: list[str] = []
    if ema_only:
        ema_module_keys = _ema_module_keys(config)
        for key in ema_module_keys:
            ori_key = get_ori_key(key)
            del model._modules[ori_key]

    if checkpoint is not None:
        if ema_only and ema_module_keys:
            ckpt = _load_checkpoint(checkpoint, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            state_dict = _prepare_ema_state_dict(state_dict, ema_module_keys)
            load_full_state_dict(model, state_dict, strict=False)
        else:
            from mmcv.runner import load_checkpoint
            load_checkpoint(model, checkpoint, map_location='cpu')

    model._cfg = config  # save the config in the model for convenience

    for module in model.modules():
        if hasattr(module, 'bake_lora_weights'):
            module.bake_lora_weights()

    if use_fp16 or use_bf16:
        for m in model.modules():
            if hasattr(m, 'autocast_dtype'):
                setattr(m, 'autocast_dtype', None)
        if use_fp16:
            assert not use_bf16
            model.to(dtype=torch.float16)
        elif use_bf16:
            model.to(dtype=torch.bfloat16)

    model.to(device)
    model.eval()

    return model
