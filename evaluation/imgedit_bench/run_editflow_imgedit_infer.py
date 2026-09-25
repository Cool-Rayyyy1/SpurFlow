#!/usr/bin/env python3
"""ImgEdit-Bench inference for EditFlow and supported baseline pipelines."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

EDITFLOW_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = Path(__file__).resolve().parent
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))

DEFAULT_BENCH_ROOT = Path(
    os.environ.get(
        "IMGEDIT_BENCH_ROOT",
        "/path/to/data/imgedit/benchmark/Benchmark",
    )
)
DEFAULT_KONTEXT_MODEL = os.environ.get(
    "KONTEXT_MODEL_PATH", "/path/to/checkpoints/FLUX.1-Kontext-dev"
)
DEFAULT_KLEIN_MODEL = os.environ.get(
    "KLEIN_MODEL_PATH", "/path/to/checkpoints/FLUX.2-klein-base-9B"
)
DEFAULT_OUTPUT_ROOT = EVAL_ROOT / "outputs"

MULTITURN_SUBS = ("content_memory", "content_understand", "version_backtrace")
TEACHER_DEFAULT_STEPS = 28
TEACHER_DEFAULT_GUIDANCE = 2.5
STUDENT_DEFAULT_GUIDANCE = TEACHER_DEFAULT_GUIDANCE
KLEIN_DEFAULT_STEPS = 4
KLEIN_DEFAULT_GUIDANCE = 1.0
QWEN_DEFAULT_STEPS = 40
QWEN_DEFAULT_GUIDANCE = 1.0
QWEN_TRUE_CFG_SCALE = float(os.environ.get("QWEN_TRUE_CFG_SCALE", "4.0"))
QWEN_NEGATIVE_PROMPT = os.environ.get("QWEN_NEGATIVE_PROMPT", " ")

BASIC_CATEGORY_DIRS = (
    "Action",
    "Add",
    "Adjust",
    "Background",
    "Compose",
    "Extract",
    "Remove",
    "Replace",
    "Style",
)


def category_dir_name(edit_type: Optional[str]) -> str:
    raw = str(edit_type or "unknown").strip()
    titled = raw[:1].upper() + raw[1:].lower() if raw else "Unknown"
    for name in BASIC_CATEGORY_DIRS:
        if name.lower() == titled.lower():
            return name
    return titled


def basic_case_dir(output_dir: Path, item: Dict, sample_key: str) -> Path:
    return output_dir / "basic" / category_dir_name(item.get("edit_type")) / sample_key


def case_bundle_ready(case_dir: Path) -> bool:
    return all(
        (case_dir / name).is_file()
        for name in ("src.png", "pred.png", "prompt.txt")
    )


def write_basic_case_bundle(
    case_dir: Path,
    src_image: Image.Image,
    pred_image: Image.Image,
    prompt: str,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    src_image.save(case_dir / "src.png")
    pred_image.save(case_dir / "pred.png")
    (case_dir / "prompt.txt").write_text(
        (prompt or "").rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ImgEdit-Bench inference for EditFlow and baselines.")
    p.add_argument("--suite", choices=("basic", "uge", "multiturn", "all"), default="all")
    p.add_argument("--role", choices=("student", "teacher", "klein", "qwen"), default="student")
    p.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT)
    p.add_argument("--annotations_dir", type=Path, default=EVAL_ROOT / "annotations")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--model_path", type=str, default=DEFAULT_KONTEXT_MODEL)
    p.add_argument("--config", type=Path, default=None, help="EditFlow config for student val_step inference.")
    p.add_argument("--ckpt", type=Path, default=None, help="EditFlow checkpoint for student inference.")
    p.add_argument("--adapter_dir", type=str, default="", help="Deprecated; student uses --config/--ckpt.")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_gpus", type=int, default=int(os.environ.get("NUM_GPUS", "2")))
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument(
        "--write_case_bundles",
        action="store_true",
        help=(
            "For basic suite, also write student/basic/<Category>/<key>/"
            "{src.png,pred.png,prompt.txt} alongside flat scoring pngs."
        ),
    )
    p.add_argument(
        "--mixture_reduce",
        type=str,
        default=os.environ.get("STUDENT_MIXTURE_REDUCE", "mean"),
        choices=("mean", "mode"),
        help=(
            "How to reduce the K-Gaussian residual mixture at inference. "
            "'mean' = weighted sum (default); 'mode' = argmax component only."
        ),
    )
    p.add_argument(
        "--dump_mixture_stats",
        action="store_true",
        help=(
            "For student runs, write per-image K mixture weight stats under "
            "mixture_stats/{key}.txt (and case folder if enabled)."
        ),
    )
    p.add_argument("--check_only", action="store_true", help="Only report suite completeness.")
    return p.parse_args()


def infer_nfe(run_name: str, explicit: Optional[int]) -> int:
    if explicit is not None:
        return explicit
    if run_name:
        m = re.search(r"(\d+)nfe", run_name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 2


def load_basic_tasks(annotations_dir: Path) -> List[Tuple[str, Dict]]:
    basic = json.loads((annotations_dir / "basic_edit.json").read_text(encoding="utf-8"))
    return [(f"basic:{key}", item) for key, item in basic.items()]


def load_uge_tasks(annotations_dir: Path) -> List[Tuple[str, Dict]]:
    uge = json.loads((annotations_dir / "UGE_edit.json").read_text(encoding="utf-8"))
    return [(f"uge:{key}", item) for key, item in uge.items()]


def load_multiturn_tasks(bench_root: Path) -> List[Tuple[str, Dict]]:
    tasks: List[Tuple[str, Dict]] = []
    for sub in MULTITURN_SUBS:
        ann_path = bench_root / "multiturn" / sub / "annotation.json"
        if not ann_path.is_file():
            continue
        for idx, line in enumerate(ann_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = f"{sub}:{idx}"
            item["_multiturn_sub"] = sub
            item["_multiturn_key"] = str(idx)
            tasks.append((f"multiturn:{key}", item))
    return tasks


def load_tasks(suite: str, annotations_dir: Path, bench_root: Path) -> List[Tuple[str, Dict]]:
    tasks: List[Tuple[str, Dict]] = []
    if suite in ("basic", "all"):
        tasks.extend(load_basic_tasks(annotations_dir))
    if suite in ("uge", "all"):
        tasks.extend(load_uge_tasks(annotations_dir))
    if suite in ("multiturn", "all"):
        tasks.extend(load_multiturn_tasks(bench_root))
    return tasks


def resolve_source_path(bench_root: Path, item: Dict, task_prefix: str) -> Path:
    if task_prefix.startswith("basic:"):
        path = bench_root / "singleturn" / item["id"]
    elif task_prefix.startswith("uge:"):
        path = bench_root / "hard" / item["id"]
    else:
        sub = item["_multiturn_sub"]
        path = bench_root / "multiturn" / sub / item["id"]
    if path.is_file():
        return path
    fname = Path(item["id"]).name
    for sub in MULTITURN_SUBS:
        cand = bench_root / "multiturn" / sub / fname
        if cand.is_file():
            return cand
    return path


def multiturn_prompts(item: Dict) -> List[str]:
    return [item[k] for k in sorted(item) if re.fullmatch(r"turn\d+", k)]


def try_open_rgb_image(path: Path) -> Optional[Image.Image]:
    """Open an RGB image, or return None if missing/truncated/unreadable."""
    if not path.is_file() or path.stat().st_size <= 32:
        return None
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return None


def save_png_atomic(image: Image.Image, path: Path) -> None:
    """Write via temp + replace so a crash cannot leave a truncated PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=path.suffix or ".png", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        image.save(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def expected_outputs(task_key: str, item: Dict) -> List[Path]:
    suite_name, sample_key = task_key.split(":", 1)
    if suite_name == "multiturn":
        sub, idx = sample_key.split(":", 1)
        turn_dir = Path("multiturn") / sub
        return [turn_dir / f"{idx}_turn{t}.png" for t in range(1, len(multiturn_prompts(item)) + 1)]
    return [Path(suite_name) / f"{sample_key}.png"]


def suite_complete(output_dir: Path, tasks: List[Tuple[str, Dict]]) -> Tuple[bool, Dict[str, int]]:
    expected = 0
    found = 0
    for task_key, item in tasks:
        rel_paths = expected_outputs(task_key, item)
        expected += len(rel_paths)
        for rel in rel_paths:
            if (output_dir / rel).is_file():
                found += 1
    return found == expected and expected > 0, {"expected": expected, "found": found}


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.role == "teacher":
        return DEFAULT_OUTPUT_ROOT / "teacher"
    if args.role == "klein":
        run_name = args.run_name or "flux2_klein_9b"
        return DEFAULT_OUTPUT_ROOT / "runs" / f"{run_name}_{KLEIN_DEFAULT_STEPS}step" / "model"
    if args.role == "qwen":
        run_name = args.run_name or "qwen_image_edit_2511"
        return DEFAULT_OUTPUT_ROOT / "runs" / f"{run_name}_{QWEN_DEFAULT_STEPS}step" / "model"
    run_name = args.run_name or "editflow"
    return DEFAULT_OUTPUT_ROOT / "student" / run_name


def resolve_inference_settings(args: argparse.Namespace) -> Tuple[int, float]:
    if args.role == "teacher":
        steps = args.num_inference_steps or int(
            os.environ.get("TEACHER_STEPS", TEACHER_DEFAULT_STEPS))
        guidance = args.guidance_scale if args.guidance_scale is not None else float(
            os.environ.get("TEACHER_GUIDANCE", TEACHER_DEFAULT_GUIDANCE))
        return steps, guidance
    if args.role == "klein":
        steps = args.num_inference_steps or int(
            os.environ.get("KLEIN_STEPS", KLEIN_DEFAULT_STEPS))
        guidance = args.guidance_scale if args.guidance_scale is not None else float(
            os.environ.get("KLEIN_GUIDANCE", KLEIN_DEFAULT_GUIDANCE))
        return steps, guidance
    if args.role == "qwen":
        steps = args.num_inference_steps or int(
            os.environ.get("QWEN_STEPS", QWEN_DEFAULT_STEPS))
        guidance = args.guidance_scale if args.guidance_scale is not None else float(
            os.environ.get("QWEN_GUIDANCE", QWEN_DEFAULT_GUIDANCE))
        return steps, guidance
    steps = infer_nfe(args.run_name, args.num_inference_steps)
    guidance = args.guidance_scale if args.guidance_scale is not None else float(
        os.environ.get(
            "STUDENT_GUIDANCE",
            os.environ.get("TEACHER_GUIDANCE", STUDENT_DEFAULT_GUIDANCE),
        ))
    return steps, guidance


def build_teacher_pipeline(model_path: str, device: str, cpu_offload: bool):
    from diffusers import FluxKontextPipeline

    pipe = FluxKontextPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def build_klein_pipeline(model_path: str, device: str, cpu_offload: bool):
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise ImportError(
            "Flux2KleinPipeline requires diffusers>=0.37. "
            "Run: pip install 'diffusers>=0.37.0'"
        ) from exc

    pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def _parse_qwen_lora_specs() -> List[Tuple[Path, str]]:
    """LoRA files from QWEN_LORA_PATHS (comma-separated) or QWEN_LORA_PATH."""
    raw = os.environ.get("QWEN_LORA_PATHS", "").strip() or os.environ.get("QWEN_LORA_PATH", "").strip()
    if not raw:
        return []
    paths = [Path(item.strip()) for item in raw.split(",") if item.strip()]
    names_raw = os.environ.get("QWEN_LORA_ADAPTER_NAMES", "").strip()
    if names_raw:
        names = [item.strip() for item in names_raw.split(",") if item.strip()]
        if len(names) != len(paths):
            raise ValueError(
                f"QWEN_LORA_ADAPTER_NAMES has {len(names)} names but {len(paths)} LoRA paths."
            )
    elif len(paths) == 2:
        names = ["style", "dmd"]
    elif len(paths) == 1:
        names = ["default"]
    else:
        names = [f"lora{i}" for i in range(len(paths))]
    return list(zip(paths, names))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    return default


def qwen_lightning_scheduler():
    """Official LightX2V / Qwen-Image-Lightning scheduler (shift=3, 4-step)."""
    import math
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    })


def build_qwen_pipeline(model_path: str, device: str, cpu_offload: bool):
    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError as exc:
        raise ImportError(
            "QwenImageEditPlusPipeline is unavailable. "
            "Install the latest diffusers from GitHub."
        ) from exc

    lora_specs = _parse_qwen_lora_specs()
    use_lightning_sched = _env_flag(
        "QWEN_LIGHTNING_SCHEDULER",
        default=(len(lora_specs) == 1),
    )
    kwargs = {"torch_dtype": torch.bfloat16}
    if lora_specs and use_lightning_sched:
        kwargs["scheduler"] = qwen_lightning_scheduler()
        print("[qwen] scheduler: FlowMatchEulerDiscreteScheduler exponential shift=3", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_path, **kwargs)
    adapter_names = []
    for lora_file, adapter_name in lora_specs:
        if not lora_file.exists():
            raise FileNotFoundError(f"Qwen LoRA not found: {lora_file}")
        load_kwargs = {}
        if adapter_name != "default":
            load_kwargs["adapter_name"] = adapter_name
        if lora_file.is_file():
            pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name, **load_kwargs)
        else:
            pipe.load_lora_weights(str(lora_file), **load_kwargs)
        adapter_names.append(adapter_name)
        print(f"[qwen] LoRA ({adapter_name}): {lora_file}", flush=True)
    if len(adapter_names) > 1:
        pipe.set_adapters(adapter_names, adapter_weights=[1.0] * len(adapter_names))
        print(f"[qwen] adapters={adapter_names} weights=1.0", flush=True)
    if lora_specs and _env_flag("QWEN_FUSE_LORA", default=False):
        fuse_names = None if adapter_names == ["default"] else adapter_names
        pipe.fuse_lora(adapter_names=fuse_names, lora_scale=1.0)
        pipe.unload_lora_weights()
        print("[qwen] fused LoRAs at scale=1.0", flush=True)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_student_model(config_path: Path, ckpt_path: Path, device: str):
    from lakonlab.apis.inference import init_model

    model = init_model(
        str(config_path),
        str(ckpt_path),
        device=device,
        use_bf16=True,
        cfg_options={"model.inference_only": True},
    )
    install_mixture_stats_hook(model)
    return model


VAE_SCALE_FACTOR = 8
STUDENT_PATCH_SIZE = 2
STUDENT_SPATIAL_MULTIPLE = VAE_SCALE_FACTOR * STUDENT_PATCH_SIZE
# Match training ImageEdit preprocessing. kontext: pick FLUX Kontext bucket;
# qwen: VAE-area resize; flux2: ~1MP area cap aligned to 16 px; center_crop: 1024 square.
STUDENT_IMAGE_SIZE = int(os.environ.get("STUDENT_IMAGE_SIZE", "1024"))
STUDENT_RESIZE_MODE = os.environ.get("STUDENT_RESIZE_MODE", "center_crop").strip().lower()


def snap_image_for_student(image: Image.Image, size: int = STUDENT_IMAGE_SIZE) -> Image.Image:
    """Resize the shorter side to `size`, then center-crop a square."""
    width, height = image.size
    if width == size and height == size:
        return image
    scale = size / min(width, height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    left = max((new_width - size) // 2, 0)
    top = max((new_height - size) // 2, 0)
    return resized.crop((left, top, left + size, top + size))


def preprocess_image_for_student(image: Image.Image) -> Image.Image:
    """Align benchmark input with training ImageEdit preprocessing."""
    if STUDENT_RESIZE_MODE in {"kontext", "flux2", "qwen"}:
        from lakonlab.datasets.image_edit import (
            _pick_flux2_resolution,
            _pick_kontext_resolution,
            _pick_qwen_vae_resolution,
            _resize_to,
        )

        width, height = image.size
        if STUDENT_RESIZE_MODE == "kontext":
            bucket_w, bucket_h = _pick_kontext_resolution(width, height)
        elif STUDENT_RESIZE_MODE == "flux2":
            bucket_w, bucket_h = _pick_flux2_resolution(width, height)
        else:
            bucket_w, bucket_h = _pick_qwen_vae_resolution(width, height)
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        arr = _resize_to(arr, bucket_w, bucket_h)
        return Image.fromarray(arr)
    if STUDENT_RESIZE_MODE != "center_crop":
        raise ValueError(
            f"Unsupported STUDENT_RESIZE_MODE={STUDENT_RESIZE_MODE!r}; "
            "expected 'kontext', 'qwen', 'flux2', or 'center_crop'.")
    return snap_image_for_student(image)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).astype(np.uint8)
    return Image.fromarray(arr)


@torch.inference_mode()
def run_one_teacher(
    pipe,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> Image.Image:
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    out = pipe(
        image=image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return out.images[0]


@torch.inference_mode()
def run_one_klein(
    pipe,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> Image.Image:
    device = getattr(pipe, "_execution_device", "cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        image=image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return out.images[0]


@torch.inference_mode()
def run_one_qwen(
    pipe,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> Image.Image:
    device = getattr(pipe, "_execution_device", "cuda")
    out = pipe(
        image=[image.convert("RGB")],
        prompt=prompt,
        generator=torch.Generator(device=device).manual_seed(seed),
        true_cfg_scale=QWEN_TRUE_CFG_SCALE,
        guidance_scale=guidance_scale,
        negative_prompt=QWEN_NEGATIVE_PROMPT,
        num_inference_steps=num_inference_steps,
    )
    return out.images[0]


def get_student_diffusion(model):
    """Return the diffusion module used at inference (EMA if present)."""
    if model is None:
        return None
    if getattr(model, "diffusion_use_ema", False) and hasattr(model, "diffusion_ema"):
        return model.diffusion_ema
    if hasattr(model, "diffusion"):
        return model.diffusion
    if hasattr(model, "diffusion_ema"):
        return model.diffusion_ema
    return None


def get_student_mixture_stats(model) -> Optional[List[Dict]]:
    """Read mixture weight stats collected by the last student forward_test."""
    diffusion = get_student_diffusion(model)
    if diffusion is None:
        return None
    stats = getattr(diffusion, "_last_mixture_stats", None)
    if stats is None:
        return None
    return list(stats)


def _mixture_stats_from_logweights(logweights: torch.Tensor) -> Dict:
    """Per-step stats for one sample: logweights (bs, K, ...) with K at dim=1."""
    lw = logweights.detach().float()[-1:]  # last batch element = positive half under CFG
    weights = torch.softmax(lw, dim=1)
    k = weights.size(1)
    reduce_dims = [d for d in range(weights.dim()) if d != 1]
    mean_weights = weights.mean(dim=reduce_dims).flatten()
    argmax = weights.argmax(dim=1).flatten()
    mode_counts = torch.bincount(argmax, minlength=k)
    entropy = -(mean_weights * mean_weights.clamp_min(1e-12).log()).sum()
    return {
        "num_gaussians": int(k),
        "mean_weights": [float(v) for v in mean_weights],
        "mode_counts": [int(v) for v in mode_counts],
        "mode_frac": [float(v) / max(int(argmax.numel()), 1) for v in mode_counts],
        "dominant_k": int(mode_counts.argmax()),
        "mean_weight_entropy": float(entropy),
    }


def install_mixture_stats_hook(model) -> None:
    """Wrap diffusion.pred so each NFE step appends pi stats to _last_mixture_stats.

    lakonlab never populates _last_mixture_stats itself; this instance-level hook
    fills it without modifying lakonlab code. Recording only happens while
    run_one_student has set _dump_mixture_stats, so the hook is free otherwise.
    """
    diffusion = get_student_diffusion(model)
    if diffusion is None or getattr(diffusion, "_mixture_stats_hooked", False):
        return
    orig_pred = diffusion.pred

    def pred_with_stats(x_t=None, t=None, **kwargs):
        out = orig_pred(x_t=x_t, t=t, **kwargs)
        if (
            getattr(diffusion, "_dump_mixture_stats", False)
            and isinstance(out, dict)
            and isinstance(out.get("logweights"), torch.Tensor)
        ):
            steps = getattr(diffusion, "_last_mixture_stats", None) or []
            stats = _mixture_stats_from_logweights(out["logweights"])
            stats["step"] = len(steps)
            steps.append(stats)
            object.__setattr__(diffusion, "_last_mixture_stats", steps)
        return out

    object.__setattr__(diffusion, "pred", pred_with_stats)
    object.__setattr__(diffusion, "_mixture_stats_hooked", True)


@torch.inference_mode()
def add_qwen_condition_source_images(data, orig_image: Image.Image, device: str) -> None:
    """Match training: VL encoder sees 384-area condition, VAE sees 1024-area source."""
    if STUDENT_RESIZE_MODE != "qwen":
        return
    from lakonlab.datasets.image_edit import (
        _pick_qwen_condition_resolution,
        _resize_to,
    )

    orig = orig_image.convert("RGB")
    orig_w, orig_h = orig.size
    cond_w, cond_h = _pick_qwen_condition_resolution(orig_w, orig_h)
    cond_arr = _resize_to(np.array(orig, dtype=np.uint8), cond_w, cond_h)
    data["condition_source_images"] = pil_to_tensor(Image.fromarray(cond_arr)).to(device)


def run_one_student(
    model,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
    mixture_reduce: str = "mean",
    dump_mixture_stats: bool = False,
) -> Image.Image:
    orig = image.convert("RGB")
    image = preprocess_image_for_student(orig)
    source = pil_to_tensor(image).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    if hasattr(model.vae, "dtype"):
        vae_dtype = model.vae.dtype
    else:
        vae_dtype = next(model.vae.parameters()).dtype
    with torch.no_grad():
        latents = model.vae.encode((source * 2 - 1).to(vae_dtype)).float()
        noise = torch.randn(latents.shape, generator=gen, device=device, dtype=latents.dtype)

    data = {
        "prompt_kwargs": {"prompt": [prompt]},
        "source_images": source,
        "noise": noise,
    }
    add_qwen_condition_source_images(data, orig, device)
    test_cfg_override = {
        "nfe": num_inference_steps,
        "distilled_guidance_scale": guidance_scale,
        "guidance_scale": 1.0,
        "mixture_reduce": mixture_reduce,
        "dump_mixture_stats": dump_mixture_stats,
    }
    # Also pin flags on the diffusion module — more reliable than test_cfg alone.
    diffusion = get_student_diffusion(model)
    if diffusion is not None:
        object.__setattr__(diffusion, "_dump_mixture_stats", bool(dump_mixture_stats))
        object.__setattr__(diffusion, "_mixture_reduce", str(mixture_reduce))
        object.__setattr__(diffusion, "_last_mixture_stats", None)

    outputs = model.val_step(data, test_cfg_override=test_cfg_override)
    return tensor_to_pil(outputs["pred_imgs"][0])


def format_mixture_stats_txt(sample_key: str, steps: List[Dict]) -> str:
    lines = [
        f"# mixture weight stats  key={sample_key}",
        f"# num_steps={len(steps)}",
        "",
    ]
    for step in steps:
        k = int(step.get("num_gaussians", len(step.get("mean_weights", []))))
        mean_w = step.get("mean_weights", [])
        mode_f = step.get("mode_frac", [])
        mode_c = step.get("mode_counts", [])
        lines.append(f"## step {step.get('step')}")
        lines.append(f"dominant_k: {step.get('dominant_k')}")
        lines.append(f"mean_weight_entropy: {step.get('mean_weight_entropy'):.6f}")
        lines.append("k\tmean_weight\tmode_frac\tmode_count")
        for i in range(k):
            lines.append(
                f"{i}\t{mean_w[i]:.6f}\t{mode_f[i]:.6f}\t{mode_c[i]}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_mixture_stats_files(
    output_dir: Path,
    sample_key: str,
    steps: List[Dict],
    case_dir: Optional[Path] = None,
) -> Path:
    stats_dir = output_dir / "mixture_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    txt_path = stats_dir / f"{sample_key}.txt"
    txt = format_mixture_stats_txt(sample_key, steps)
    txt_path.write_text(txt, encoding="utf-8")
    json_path = stats_dir / f"{sample_key}.json"
    json_path.write_text(
        json.dumps({"key": sample_key, "steps": steps}, indent=2),
        encoding="utf-8",
    )
    if case_dir is not None:
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "mixture_weights.txt").write_text(txt, encoding="utf-8")
    return txt_path


def run_one(
    runner: Union[object, str],
    role: str,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
    mixture_reduce: str = "mean",
    dump_mixture_stats: bool = False,
) -> Image.Image:
    if role == "teacher":
        return run_one_teacher(
            runner, image, prompt, num_inference_steps, guidance_scale, seed)
    if role == "klein":
        return run_one_klein(
            runner, image, prompt, num_inference_steps, guidance_scale, seed)
    if role == "qwen":
        return run_one_qwen(
            runner, image, prompt, num_inference_steps, guidance_scale, seed)
    return run_one_student(
        runner, image, prompt, num_inference_steps, guidance_scale, seed, device,
        mixture_reduce=mixture_reduce,
        dump_mixture_stats=dump_mixture_stats)


def run_multiturn_chain(
    runner,
    role: str,
    source: Image.Image,
    prompts: List[str],
    out_paths: List[Path],
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    skip_existing: bool,
    device: str,
    mixture_reduce: str = "mean",
    dump_mixture_stats: bool = False,
) -> None:
    current = source
    for turn_idx, (prompt, out_path) in enumerate(zip(prompts, out_paths), start=1):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing:
            existing = try_open_rgb_image(out_path)
            if existing is not None:
                current = existing
                continue
            if out_path.is_file():
                print(
                    f"[warn] unreadable existing turn, regenerating: {out_path}",
                    flush=True)
                out_path.unlink()
        current = run_one(
            runner,
            role,
            image=current,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed + turn_idx,
            device=device,
            mixture_reduce=mixture_reduce,
            dump_mixture_stats=dump_mixture_stats,
        )
        save_png_atomic(current, out_path)


def resolve_gpu_ids(num_gpus: int) -> List[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [int(x) for x in visible.split(",") if x.strip()]
    else:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        ids = list(range(n))
    if not ids:
        return [0]
    if num_gpus > len(ids):
        raise ValueError(
            f"--num_gpus={num_gpus} but only {len(ids)} GPU(s) visible ({visible!r})")
    return ids[:num_gpus]


def split_tasks_round_robin(tasks: List[Tuple[str, Dict]], num_workers: int) -> List[List[Tuple[str, Dict]]]:
    buckets: List[List[Tuple[str, Dict]]] = [[] for _ in range(num_workers)]
    for idx, task in enumerate(tasks):
        buckets[idx % num_workers].append(task)
    return [bucket for bucket in buckets if bucket]


def process_tasks(
    runner,
    role: str,
    tasks: List[Tuple[str, Dict]],
    bench_root: Path,
    output_dir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    skip_existing: bool,
    device: str,
    desc: str,
    write_case_bundles: bool = False,
    mixture_reduce: str = "mean",
    dump_mixture_stats: bool = False,
) -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    for task_key, item in tqdm(tasks, desc=desc):
        suite_name, sample_key = task_key.split(":", 1)
        rel_paths = expected_outputs(task_key, item)
        abs_paths = [output_dir / rel for rel in rel_paths]
        case_dir = (
            basic_case_dir(output_dir, item, sample_key)
            if write_case_bundles and suite_name == "basic"
            else None
        )
        stats_path = output_dir / "mixture_stats" / f"{sample_key}.txt"
        score_ready = all(try_open_rgb_image(p) is not None for p in abs_paths)
        case_ready = case_dir is None or case_bundle_ready(case_dir)
        stats_ready = (not dump_mixture_stats) or stats_path.is_file()
        if skip_existing and score_ready and case_ready and stats_ready:
            continue

        src_path = resolve_source_path(bench_root, item, task_key)
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_path}")

        if role == "qwen":
            task_seed = seed
        else:
            task_seed = (
                seed + int(sample_key.split(":")[-1])
                if sample_key.split(":")[-1].isdigit()
                else seed
            )
        image = try_open_rgb_image(src_path)
        if image is None:
            print(f"[skip] unreadable source: {src_path}", flush=True)
            manifest[task_key] = {
                "suite": suite_name,
                "key": sample_key,
                "source": str(src_path),
                "error": f"unreadable source: {src_path}",
            }
            continue

        ran_model = False
        if suite_name == "multiturn":
            run_multiturn_chain(
                runner,
                role,
                source=image,
                prompts=multiturn_prompts(item),
                out_paths=abs_paths,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=task_seed,
                skip_existing=skip_existing,
                device=device,
                mixture_reduce=mixture_reduce,
                dump_mixture_stats=dump_mixture_stats,
            )
            ran_model = True
        else:
            abs_paths[0].parent.mkdir(parents=True, exist_ok=True)
            # Prefer rebuilding the flat score png from an existing case bundle.
            if (
                not abs_paths[0].is_file()
                and case_dir is not None
                and case_bundle_ready(case_dir)
            ):
                shutil.copy2(case_dir / "pred.png", abs_paths[0])
            if not abs_paths[0].is_file() or (
                dump_mixture_stats and role == "student" and not stats_path.is_file()
            ):
                result = run_one(
                    runner,
                    role,
                    image=image,
                    prompt=item["prompt"],
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=task_seed,
                    device=device,
                    mixture_reduce=mixture_reduce,
                    dump_mixture_stats=dump_mixture_stats,
                )
                result.save(abs_paths[0])
                ran_model = True
            else:
                result = Image.open(abs_paths[0]).convert("RGB")
            if case_dir is not None:
                write_basic_case_bundle(
                    case_dir, image, result, item.get("prompt", ""))

        mixture_stats_file = None
        if dump_mixture_stats and role == "student" and ran_model:
            steps = get_student_mixture_stats(runner)
            if steps:
                mixture_stats_file = str(
                    write_mixture_stats_files(
                        output_dir, sample_key, steps, case_dir=case_dir))
            else:
                print(
                    f"[warn] dump_mixture_stats enabled but no stats for key={sample_key}; "
                    f"diffusion={type(get_student_diffusion(runner)).__name__ if get_student_diffusion(runner) else None}"
                )

        manifest[task_key] = {
            "suite": suite_name,
            "key": sample_key,
            "source": str(src_path),
            "prompt": item.get("prompt"),
            "turns": multiturn_prompts(item) if suite_name == "multiturn" else None,
            "outputs": [str(p) for p in abs_paths],
            "case_dir": str(case_dir) if case_dir is not None else None,
            "edit_type": item.get("edit_type"),
            "mixture_reduce": mixture_reduce,
            "mixture_stats": mixture_stats_file,
        }
    if dump_mixture_stats and role == "student":
        _write_mixture_stats_summary(output_dir)
    return manifest


def _write_mixture_stats_summary(output_dir: Path) -> None:
    """Aggregate per-image mixture json into a short distribution summary."""
    stats_dir = output_dir / "mixture_stats"
    if not stats_dir.is_dir():
        return
    json_files = sorted(stats_dir.glob("*.json"))
    if not json_files:
        return
    # Aggregate final-step dominant_k and mean_weights across images.
    k_hist: Dict[int, int] = {}
    sum_mean: Optional[List[float]] = None
    n = 0
    for jf in json_files:
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        steps = payload.get("steps") or []
        if not steps:
            continue
        step = steps[-1]
        dk = int(step.get("dominant_k", -1))
        k_hist[dk] = k_hist.get(dk, 0) + 1
        mw = [float(x) for x in step.get("mean_weights", [])]
        if not mw:
            continue
        if sum_mean is None:
            sum_mean = [0.0] * len(mw)
        if len(mw) != len(sum_mean):
            continue
        for i, v in enumerate(mw):
            sum_mean[i] += v
        n += 1
    lines = [
        "# mixture_stats summary (final NFE step across images)",
        f"num_images_with_stats: {n}",
        "",
        "## dominant_k histogram (which component wins most pixels)",
    ]
    for k in sorted(k_hist):
        lines.append(f"k={k}: {k_hist[k]}")
    if sum_mean and n > 0:
        lines.append("")
        lines.append("## average mean_weights over images")
        lines.append("k\tavg_mean_weight")
        for i, v in enumerate(sum_mean):
            lines.append(f"{i}\t{v / n:.6f}")
    out = stats_dir / "summary.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _infer_worker(gpu_id: int, tasks: List[Tuple[str, Dict]], worker_cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"
    if worker_cfg["role"] == "teacher":
        runner = build_teacher_pipeline(
            worker_cfg["model_path"], device, worker_cfg["cpu_offload"])
    elif worker_cfg["role"] == "klein":
        runner = build_klein_pipeline(
            worker_cfg["model_path"], device, worker_cfg["cpu_offload"])
    elif worker_cfg["role"] == "qwen":
        runner = build_qwen_pipeline(
            worker_cfg["model_path"], device, worker_cfg["cpu_offload"])
    else:
        runner = build_student_model(
            Path(worker_cfg["config"]),
            Path(worker_cfg["ckpt"]),
            device,
        )
    manifest = process_tasks(
        runner,
        worker_cfg["role"],
        tasks,
        Path(worker_cfg["bench_root"]),
        Path(worker_cfg["output_dir"]),
        worker_cfg["num_steps"],
        worker_cfg["guidance_scale"],
        worker_cfg["seed"],
        worker_cfg["skip_existing"],
        device,
        desc=f"GPU {gpu_id}",
        write_case_bundles=worker_cfg.get("write_case_bundles", False),
        mixture_reduce=worker_cfg.get("mixture_reduce", "mean"),
        dump_mixture_stats=worker_cfg.get("dump_mixture_stats", False),
    )
    part_path = Path(worker_cfg["output_dir"]) / f"manifest.gpu{gpu_id}.json"
    part_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    del runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def merge_manifest_parts(output_dir: Path, base_manifest: Dict[str, Dict]) -> Dict[str, Dict]:
    manifest = dict(base_manifest)
    for part in sorted(output_dir.glob("manifest.gpu*.json")):
        manifest.update(json.loads(part.read_text(encoding="utf-8")))
        part.unlink()
    return manifest


def run_parallel_inference(
    args: argparse.Namespace,
    tasks: List[Tuple[str, Dict]],
    output_dir: Path,
    num_steps: int,
    guidance_scale: float,
    gpu_ids: List[int],
) -> Dict[str, Dict]:
    buckets = split_tasks_round_robin(tasks, len(gpu_ids))
    worker_cfg = {
        "role": args.role,
        "model_path": args.model_path,
        "config": str(args.config) if args.config is not None else "",
        "ckpt": str(args.ckpt) if args.ckpt is not None else "",
        "bench_root": str(args.bench_root),
        "output_dir": str(output_dir),
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "seed": args.seed,
        "skip_existing": args.skip_existing,
        "cpu_offload": args.cpu_offload,
        "write_case_bundles": bool(getattr(args, "write_case_bundles", False)),
        "mixture_reduce": str(getattr(args, "mixture_reduce", "mean")),
        "dump_mixture_stats": bool(getattr(args, "dump_mixture_stats", False)),
    }
    if len(buckets) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        device = args.device
        if args.role == "teacher":
            runner = build_teacher_pipeline(args.model_path, device, args.cpu_offload)
        elif args.role == "klein":
            runner = build_klein_pipeline(args.model_path, device, args.cpu_offload)
        elif args.role == "qwen":
            runner = build_qwen_pipeline(args.model_path, device, args.cpu_offload)
        else:
            runner = build_student_model(args.config, args.ckpt, device)
        return process_tasks(
            runner,
            args.role,
            buckets[0],
            args.bench_root,
            output_dir,
            num_steps,
            guidance_scale,
            args.seed,
            args.skip_existing,
            device,
            desc=f"ImgEdit {args.role}/{args.suite}",
            write_case_bundles=worker_cfg["write_case_bundles"],
            mixture_reduce=worker_cfg["mixture_reduce"],
            dump_mixture_stats=worker_cfg["dump_mixture_stats"],
        )

    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, bucket in zip(gpu_ids, buckets):
        p = ctx.Process(target=_infer_worker, args=(gpu_id, bucket, worker_cfg))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Inference worker failed with exit code {p.exitcode}")
    return merge_manifest_parts(output_dir, {})


def main() -> None:
    args = parse_args()
    if args.role == "student":
        if args.config is None or args.ckpt is None:
            raise ValueError("Student inference requires --config and --ckpt.")
        if not args.config.is_file():
            raise FileNotFoundError(f"Config not found: {args.config}")
        if not args.ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    elif args.role in ("klein", "qwen"):
        model_dir = Path(args.model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"{args.role} model not found: {model_dir}")

    num_steps, guidance_scale = resolve_inference_settings(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.suite, args.annotations_dir, args.bench_root)
    if args.max_samples is not None:
        tasks = tasks[: args.max_samples]

    complete, report = suite_complete(output_dir, tasks)
    if args.check_only:
        print(json.dumps({"output_dir": str(output_dir), "complete": complete, **report}, indent=2))
        raise SystemExit(0 if complete else 1)
    if complete and args.skip_existing:
        print(f"[skip] {args.role} outputs already complete at {output_dir} ({report['found']} files)")
        return

    manifest_path = output_dir / "manifest.json"
    manifest: Dict[str, Dict] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    gpu_ids = resolve_gpu_ids(args.num_gpus)
    use_parallel = len(gpu_ids) > 1 and not args.cpu_offload

    if use_parallel:
        print(f"[infer] parallel on GPUs: {gpu_ids}")
        new_manifest = run_parallel_inference(args, tasks, output_dir, num_steps, guidance_scale, gpu_ids)
        manifest.update(new_manifest)
    else:
        if len(gpu_ids) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        device = args.device
        if args.role == "teacher":
            runner = build_teacher_pipeline(args.model_path, device, args.cpu_offload)
        elif args.role == "klein":
            runner = build_klein_pipeline(args.model_path, device, args.cpu_offload)
        elif args.role == "qwen":
            runner = build_qwen_pipeline(args.model_path, device, args.cpu_offload)
        else:
            runner = build_student_model(args.config, args.ckpt, device)
        new_manifest = process_tasks(
            runner,
            args.role,
            tasks,
            args.bench_root,
            output_dir,
            num_steps,
            guidance_scale,
            args.seed,
            args.skip_existing,
            device,
            desc=f"ImgEdit {args.role}/{args.suite}",
            write_case_bundles=bool(getattr(args, "write_case_bundles", False)),
            mixture_reduce=str(getattr(args, "mixture_reduce", "mean")),
            dump_mixture_stats=bool(getattr(args, "dump_mixture_stats", False)),
        )
        manifest.update(new_manifest)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "role": args.role,
        "model_path": args.model_path,
        "config": str(args.config) if args.config is not None else "",
        "ckpt": str(args.ckpt) if args.ckpt is not None else "",
        "run_name": args.run_name,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": QWEN_TRUE_CFG_SCALE if args.role == "qwen" else None,
        "negative_prompt": QWEN_NEGATIVE_PROMPT if args.role == "qwen" else None,
        "lora_path": os.environ.get("QWEN_LORA_PATH", "").strip() or None,
        "lora_paths": os.environ.get("QWEN_LORA_PATHS", "").strip() or None,
        "fuse_lora": _env_flag("QWEN_FUSE_LORA", default=False),
        "suite": args.suite,
        "num_tasks": len(manifest),
        "num_gpus": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "student_resize_mode": STUDENT_RESIZE_MODE,
        "student_image_size": STUDENT_IMAGE_SIZE,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(manifest)} task records to {output_dir}")


if __name__ == "__main__":
    main()
