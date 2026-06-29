#!/usr/bin/env python3
"""ImgEdit-Bench inference for EditFlow (student) and FLUX Kontext teacher."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

EDITFLOW_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = Path(__file__).resolve().parent
if str(EDITFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITFLOW_ROOT))

DEFAULT_BENCH_ROOT = Path(
    os.environ.get(
        "IMGEDIT_BENCH_ROOT",
        "/mnt/afs_zhangyunzhe/dataset/imgedit/benchmark/Benchmark",
    )
)
DEFAULT_KONTEXT_MODEL = os.environ.get(
    "KONTEXT_MODEL_PATH", "/mnt/afs_zhangyunzhe/pretrained_models/FLUX.1-Kontext-dev"
)
DEFAULT_KLEIN_MODEL = os.environ.get(
    "KLEIN_MODEL_PATH", "/mnt/afs_zhangyunzhe/pretrained_models/FLUX.2-klein-9B"
)
DEFAULT_OUTPUT_ROOT = EVAL_ROOT / "outputs"

MULTITURN_SUBS = ("content_memory", "content_understand", "version_backtrace")
TEACHER_DEFAULT_STEPS = 28
TEACHER_DEFAULT_GUIDANCE = 2.5
STUDENT_DEFAULT_GUIDANCE = TEACHER_DEFAULT_GUIDANCE
KLEIN_DEFAULT_STEPS = 4
KLEIN_DEFAULT_GUIDANCE = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ImgEdit-Bench inference for EditFlow/Kontext.")
    p.add_argument("--suite", choices=("basic", "uge", "multiturn", "all"), default="all")
    p.add_argument("--role", choices=("student", "teacher", "klein"), default="student")
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
    steps = infer_nfe(args.run_name, args.num_inference_steps)
    guidance = args.guidance_scale if args.guidance_scale is not None else float(
        os.environ.get(
            "STUDENT_GUIDANCE",
            os.environ.get("TEACHER_GUIDANCE", STUDENT_DEFAULT_GUIDANCE),
        ))
    return steps, guidance


def build_teacher_pipeline(model_path: str, device: str, cpu_offload: bool):
    from diffusers import FluxKontextPipeline, FlowMatchEulerDiscreteScheduler

    pipe = FluxKontextPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config, shift=3.2, shift_terminal=None, use_dynamic_shifting=False)
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


def build_student_model(config_path: Path, ckpt_path: Path, device: str):
    from lakonlab.apis.inference import init_model

    return init_model(
        str(config_path),
        str(ckpt_path),
        device=device,
        use_bf16=True,
        cfg_options={"model.inference_only": True},
    )


VAE_SCALE_FACTOR = 8
STUDENT_PATCH_SIZE = 2
STUDENT_SPATIAL_MULTIPLE = VAE_SCALE_FACTOR * STUDENT_PATCH_SIZE
# Match training ImageEdit preprocessing. kontext: pick FLUX Kontext bucket from source
# aspect ratio and bicubic resize (no crop). center_crop: 1024 square center crop.
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
    if STUDENT_RESIZE_MODE == "kontext":
        from lakonlab.datasets.image_edit import _pick_kontext_resolution, _resize_to

        width, height = image.size
        bucket_w, bucket_h = _pick_kontext_resolution(width, height)
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        arr = _resize_to(arr, bucket_w, bucket_h)
        return Image.fromarray(arr)
    if STUDENT_RESIZE_MODE != "center_crop":
        raise ValueError(
            f"Unsupported STUDENT_RESIZE_MODE={STUDENT_RESIZE_MODE!r}; "
            "expected 'kontext' or 'center_crop'.")
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
def run_one_student(
    model,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
) -> Image.Image:
    image = preprocess_image_for_student(image)
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
    test_cfg_override = {
        "nfe": num_inference_steps,
        "distilled_guidance_scale": guidance_scale,
        "guidance_scale": 1.0,
    }
    outputs = model.val_step(data, test_cfg_override=test_cfg_override)
    return tensor_to_pil(outputs["pred_imgs"][0])


def run_one(
    runner: Union[object, str],
    role: str,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
) -> Image.Image:
    if role == "teacher":
        return run_one_teacher(
            runner, image, prompt, num_inference_steps, guidance_scale, seed)
    if role == "klein":
        return run_one_klein(
            runner, image, prompt, num_inference_steps, guidance_scale, seed)
    return run_one_student(
        runner, image, prompt, num_inference_steps, guidance_scale, seed, device)


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
) -> None:
    current = source
    for turn_idx, (prompt, out_path) in enumerate(zip(prompts, out_paths), start=1):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and out_path.is_file():
            current = Image.open(out_path).convert("RGB")
            continue
        current = run_one(
            runner,
            role,
            image=current,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed + turn_idx,
            device=device,
        )
        current.save(out_path)


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
) -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    for task_key, item in tqdm(tasks, desc=desc):
        suite_name, sample_key = task_key.split(":", 1)
        rel_paths = expected_outputs(task_key, item)
        abs_paths = [output_dir / rel for rel in rel_paths]
        if skip_existing and all(p.is_file() for p in abs_paths):
            continue

        src_path = resolve_source_path(bench_root, item, task_key)
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image: {src_path}")

        task_seed = seed + int(sample_key.split(":")[-1]) if sample_key.split(":")[-1].isdigit() else seed
        image = Image.open(src_path).convert("RGB")

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
            )
        else:
            abs_paths[0].parent.mkdir(parents=True, exist_ok=True)
            result = run_one(
                runner,
                role,
                image=image,
                prompt=item["prompt"],
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=task_seed,
                device=device,
            )
            result.save(abs_paths[0])

        manifest[task_key] = {
            "suite": suite_name,
            "key": sample_key,
            "source": str(src_path),
            "prompt": item.get("prompt"),
            "turns": multiturn_prompts(item) if suite_name == "multiturn" else None,
            "outputs": [str(p) for p in abs_paths],
            "edit_type": item.get("edit_type"),
        }
    return manifest


def _infer_worker(gpu_id: int, tasks: List[Tuple[str, Dict]], worker_cfg: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"
    if worker_cfg["role"] == "teacher":
        runner = build_teacher_pipeline(
            worker_cfg["model_path"], device, worker_cfg["cpu_offload"])
    elif worker_cfg["role"] == "klein":
        runner = build_klein_pipeline(
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
    }
    if len(buckets) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        device = args.device
        if args.role == "teacher":
            runner = build_teacher_pipeline(args.model_path, device, args.cpu_offload)
        elif args.role == "klein":
            runner = build_klein_pipeline(args.model_path, device, args.cpu_offload)
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
    elif args.role == "klein":
        model_dir = Path(args.model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Klein model not found: {model_dir}")

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
