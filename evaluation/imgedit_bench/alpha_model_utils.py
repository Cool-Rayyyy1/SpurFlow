"""Load four-channel sigmoid-alpha checkpoints for inference."""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
EDITFLOW_ROOT = EVAL_ROOT.parents[1]
DEFAULT_ALPHA_CONFIG = (
    EDITFLOW_ROOT / 'configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data.py'
)


def _ensure_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def peek_alpha_head_dim(ckpt_path: Path) -> int:
    """Return proj_out_alpha output dim (expected 4 for sigmoid alpha)."""
    import torch

    payload = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload)
    for key in (
        'diffusion_ema.denoising.proj_out_alpha.weight',
        'diffusion.denoising.proj_out_alpha.weight',
    ):
        if key in state:
            return int(state[key].shape[0])
    raise KeyError(f'proj_out_alpha not found in checkpoint: {ckpt_path}')


def validate_continuous_alpha_ckpt(ckpt_path: Path) -> None:
    alpha_dim = peek_alpha_head_dim(ckpt_path)
    if alpha_dim != 4:
        raise ValueError(
            f'Checkpoint {ckpt_path} has proj_out_alpha dim={alpha_dim}; '
            'expected 4 for sigmoid alpha (ArcFluxEditNewAlphaTransformer2DModel). '
            'Use a checkpoint trained with train_flux_edit_fixedeps_alpha_data.sh.'
        )


def resolve_alpha_config(config_path: Path | None = None) -> Path:
    if config_path is not None:
        return Path(config_path)
    return DEFAULT_ALPHA_CONFIG


def build_alpha_vis_model(config_path: Path, ckpt_path: Path, device: str):
    ckpt_path = Path(ckpt_path)
    config_path = resolve_alpha_config(config_path)
    validate_continuous_alpha_ckpt(ckpt_path)
    _ensure_path(str(EDITFLOW_ROOT))

    from lakonlab.apis.inference import init_model

    return init_model(
        str(config_path),
        str(ckpt_path),
        device=device,
        use_bf16=True,
        cfg_options={'model.inference_only': True},
    )
