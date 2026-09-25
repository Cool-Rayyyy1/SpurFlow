#!/usr/bin/env python3
"""Render the display images in assets/EditFlow_images with our students and the
official LoRA baselines.

Targets only: no alpha capture, no heatmaps. Every source is sampled under many
seeds, so one output directory per source, per run:

  {out}/{tag}/{category}/{name}/src.png            qwen-resized source
  {out}/{tag}/{category}/{name}/tgt_seed0000.png   one per seed
  {out}/{tag}/{category}/{name}/prompt.txt
  {out}/manifest.json

The seed set is the same for every source and every run, so tgt_seed0007.png is
directly comparable across all of them.

src is the output of preprocess_image_for_student, i.e. exactly the tensor our
students saw, so src and tgt are the same size by construction rather than by a
separate resize that could disagree with the bucket the model picked. The
baseline runs are fed that same resized src and are told its height/width
explicitly, so their targets line up pixel-for-pixel too -- QwenImageEditPlus
would otherwise pick its own bucket from the raw file and land elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

EDITFLOW_ROOT = Path(__file__).resolve().parents[2]
for path in (str(EDITFLOW_ROOT / "evaluation" / "imgedit_bench"), str(EDITFLOW_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_editflow_imgedit_infer import (  # noqa: E402
    build_qwen_pipeline,
    build_student_model,
    preprocess_image_for_student,
    run_one_student,
    save_png_atomic,
)

# --- our students -----------------------------------------------------------
# The two oss_pico iters share a config; the GAN pretrain one is the split-stage
# recipe; arcflow_qwen is the plain ArcFlow baseline (no uedit/fixedeps/alpha/
# GAN, same oss30+pico70 data mix).
OSS_PICO_CFG = "configs/qwen/editqwen_uedit_fixedeps_2nfe_k16_alpha_softsign_step2_alph_dino_gan_oss_pico.py"
SPLIT_STAGE_CFG = "configs/qwen/editqwen_uedit_fixedeps_2nfe_k16_alpha_softsign_split_stage_alph_dino_gan.py"
ARCFLOW_CFG = "configs/qwen/editqwen_2nfe_k16_data_oss_pico.py"

# --- official LoRA baselines ------------------------------------------------
# Same knobs as evaluation/run_qwen_image_edit_2511_lightning_4step_*.sh and
# run_telestyle_qie2511_*.sh; the LoRA env vars are read by build_qwen_pipeline
# at call time, so setting them per run inside this process is enough.
QWEN_BASE = "/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511"
LIGHTNING_LORA = ("/mnt/afs_gaochengmin/checkpoints/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16/"
                  "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
TELESTYLE_DIR = "/mnt/afs_gaochengmin/checkpoints/TeleStyleV2"
TELESTYLE_STYLE_LORA = f"{TELESTYLE_DIR}/diffusers-TeleStyleV2-QIE-2511-Lora-bf16.safetensors"
TELESTYLE_LIGHTNING_LORA = f"{TELESTYLE_DIR}/QIE-2511-Lightning-4steps-V1.0-bf16.safetensors"

# Both baselines run CFG-free with a blank negative prompt. Hardcoded rather
# than read from QWEN_TRUE_CFG_SCALE because that env var is bound at import
# time by the inference module, which is too easy to get wrong from a launcher.
QWEN_TRUE_CFG = 1.0
QWEN_NEGATIVE = " "

DEFAULT_RUNS: List[Dict] = [
    {"tag": "oss_pico_iter2400", "kind": "student", "config": OSS_PICO_CFG,
     "ckpt": "checkpoints/gmqwen_uedit_fixedeps_alpha_softsign01_k16_2nfe_oss30_pico70_step2_alph_dino_gan/20260829_011114/iter_2400.pth"},
    {"tag": "oss_pico_iter2800", "kind": "student", "config": OSS_PICO_CFG,
     "ckpt": "checkpoints/gmqwen_uedit_fixedeps_alpha_softsign01_k16_2nfe_oss30_pico70_step2_alph_dino_gan/20260829_011114/iter_2800.pth"},
    {"tag": "gan_pretrain_iter2800", "kind": "student", "config": SPLIT_STAGE_CFG,
     "ckpt": "pretrain/qwen/gan/iter_2800.pth"},
    {"tag": "arcflow_qwen", "kind": "student", "config": ARCFLOW_CFG,
     "ckpt": "checkpoints/gmqwen_k16_2nfe_oss30_pico70_data/20260817_111001/iter_20000.pth"},
    # Single LoRA, unfused: build_qwen_pipeline then defaults to the Lightning
    # exponential shift=3 scheduler, matching the official 4-step recipe.
    {"tag": "qwen_lightning_4step", "kind": "qwen", "steps": 4,
     "model_path": QWEN_BASE,
     "env": {"QWEN_LORA_PATHS": LIGHTNING_LORA,
             "QWEN_LORA_ADAPTER_NAMES": "",
             "QWEN_LIGHTNING_SCHEDULER": "1",
             "QWEN_FUSE_LORA": "0"}},
    # Style + Lightning LoRA fused at scale 1.0, stock scheduler, per the
    # TeleStyleV2 demo.
    {"tag": "telestyle_qie2511_4step", "kind": "qwen", "steps": 4,
     "model_path": QWEN_BASE,
     "env": {"QWEN_LORA_PATHS": f"{TELESTYLE_STYLE_LORA},{TELESTYLE_LIGHTNING_LORA}",
             "QWEN_LORA_ADAPTER_NAMES": "style,dmd",
             "QWEN_LIGHTNING_SCHEDULER": "0",
             "QWEN_FUSE_LORA": "1"}},
]


@torch.inference_mode()
def run_one_qwen_sized(pipe, image: Image.Image, prompt: str, steps: int,
                       guidance_scale: float, seed: int) -> Image.Image:
    """run_one_qwen, but pinned to the size of the image we hand it.

    Without height/width the pipeline re-derives its own 1024^2-area bucket from
    the input, which need not equal the bucket our students used.
    """
    device = getattr(pipe, "_execution_device", "cuda")
    width, height = image.size
    out = pipe(
        image=[image.convert("RGB")],
        prompt=prompt,
        height=height,
        width=width,
        generator=torch.Generator(device=device).manual_seed(seed),
        true_cfg_scale=QWEN_TRUE_CFG,
        guidance_scale=guidance_scale,
        negative_prompt=QWEN_NEGATIVE,
        num_inference_steps=steps,
    )
    return out.images[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render display assets with Qwen students.")
    p.add_argument("--assets", type=Path, default=EDITFLOW_ROOT / "assets" / "EditFlow_images")
    p.add_argument("--output_dir", type=Path, default=EDITFLOW_ROOT / "work_dirs" / "display_new")
    p.add_argument("--num_inference_steps", type=int, default=2,
                   help="Students only; the LoRA baselines carry their own "
                        "step count (4) in DEFAULT_RUNS.")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--num_seeds", type=int, default=100,
                   help="Seeds per source: seed_start .. seed_start+num_seeds-1. "
                        "The same seed set is used for every source and every "
                        "checkpoint, so tgt_seed0007.png is comparable across all of them.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--only_tags", type=str, nargs="+", default=None,
                   help="Restrict to these checkpoint tags.")
    p.add_argument("--only_ids", type=str, nargs="+", default=None,
                   help="Restrict to these task ids, e.g. replace/yoga_cat.")
    return p.parse_args()


def load_tasks(assets: Path) -> List[Dict]:
    tasks = json.loads((assets / "tasks.json").read_text(encoding="utf-8"))
    for item in tasks:
        src = assets / item["image"]
        if not src.is_file():
            raise FileNotFoundError(f"asset image missing: {src}")
        item["_src_path"] = src
    return tasks


def case_dir_for(out_root: Path, item: Dict) -> Path:
    """One directory per source, since it now holds a hundred targets."""
    category, name = item["id"].split("/", 1)
    return out_root / category / name


def tgt_name(seed: int) -> str:
    return f"tgt_seed{seed:04d}.png"


def build_run(run: Dict, device: str, steps_default: int, guidance: float):
    """Load a run and return (handle, describe_lines, infer(image, prompt, seed))."""
    if run["kind"] == "student":
        ckpt_path = EDITFLOW_ROOT / run["ckpt"]
        config_path = EDITFLOW_ROOT / run["config"]
        for path in (ckpt_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        lines = [f"  ckpt   {run['ckpt']}", f"  config {run['config']}",
                 f"  steps  {steps_default}  guidance {guidance}"]
        model = build_student_model(config_path, ckpt_path, device)

        def infer(image: Image.Image, prompt: str, seed: int) -> Image.Image:
            return run_one_student(model, image, prompt, steps_default,
                                   guidance, seed, device)

        return model, lines, infer

    steps = run.get("steps", steps_default)
    env = run.get("env", {})
    for key, value in env.items():
        os.environ[key] = value  # read at call time by build_qwen_pipeline
    for lora in env.get("QWEN_LORA_PATHS", "").split(","):
        if lora.strip() and not Path(lora.strip()).exists():
            raise FileNotFoundError(f"LoRA not found: {lora.strip()}")
    lines = [f"  base   {run['model_path']}",
             f"  loras  {env.get('QWEN_LORA_PATHS', '(none)')}",
             f"  steps  {steps}  guidance {guidance}  true_cfg {QWEN_TRUE_CFG}"]
    pipe = build_qwen_pipeline(run["model_path"], device, cpu_offload=False)

    def infer(image: Image.Image, prompt: str, seed: int) -> Image.Image:
        # The baseline sees the same qwen-resized src our students did.
        return run_one_qwen_sized(pipe, preprocess_image_for_student(image),
                                  prompt, steps, guidance, seed)

    return pipe, lines, infer


def scan_manifest(out_dir: Path, tasks: List[Dict], seeds: List[int]) -> Dict[str, Dict]:
    """Build the manifest from what is on disk, over every run in DEFAULT_RUNS.

    Deliberately not accumulated during rendering: a tag that --skip_existing
    skipped entirely never enters the render loop, and a crash partway through
    a multi-tag run used to lose the tags that had already finished.
    """
    manifest: Dict[str, Dict] = {}
    for item in tasks:
        category, _ = item["id"].split("/", 1)
        for run in DEFAULT_RUNS:
            case_dir = case_dir_for(out_dir / run["tag"], item)
            src = case_dir / "src.png"
            if not src.is_file():
                continue
            rec = manifest.setdefault(item["id"], {
                "category": category,
                "prompt": item["prompt"],
                "source": str(item["_src_path"]),
                "size": list(Image.open(src).size),
                "seeds": seeds,
                "case_dirs": {},
            })
            rec["case_dirs"][run["tag"]] = str(case_dir)
    return manifest


def main() -> None:
    args = parse_args()
    if os.environ.get("STUDENT_RESIZE_MODE", "").strip().lower() != "qwen":
        raise SystemExit("set STUDENT_RESIZE_MODE=qwen before running this "
                         "(it is read at import time by the inference module).")

    tasks = load_tasks(args.assets)
    if args.only_ids is not None:
        wanted = set(args.only_ids)
        tasks = [t for t in tasks if t["id"] in wanted]
        missing = wanted - {t["id"] for t in tasks}
        if missing:
            raise SystemExit(f"--only_ids not in tasks.json: {sorted(missing)}")
    runs = [r for r in DEFAULT_RUNS
            if args.only_tags is None or r["tag"] in args.only_tags]
    if not runs:
        known = [r["tag"] for r in DEFAULT_RUNS]
        raise SystemExit(f"no run matches --only_tags {args.only_tags}; known: {known}")

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    for run in runs:
        tag = run["tag"]
        out_root = args.output_dir / tag
        todo = [(item, seed) for item in tasks for seed in seeds
                if not (args.skip_existing
                        and case_dir_for(out_root, item).joinpath(tgt_name(seed)).is_file())]
        if not todo:
            print(f"[{tag}] all {len(tasks) * len(seeds)} targets present, skipping", flush=True)
            continue

        print(f"\n=========== {tag}  ({len(todo)}/{len(tasks) * len(seeds)} to render) "
              f"===========", flush=True)
        handle, lines, infer = build_run(
            run, args.device, args.num_inference_steps, args.guidance_scale)
        for line in lines:
            print(line, flush=True)

        start = time.time()
        for done, (item, seed) in enumerate(todo, start=1):
            case_dir = case_dir_for(out_root, item)
            case_dir.mkdir(parents=True, exist_ok=True)
            src_out, tgt_out = case_dir / "src.png", case_dir / tgt_name(seed)

            image = Image.open(item["_src_path"]).convert("RGB")
            edited = infer(image, item["prompt"], seed)
            # Same call the students make internally, so sizes agree exactly.
            src_pil = preprocess_image_for_student(image)
            if src_pil.size != edited.size:
                raise RuntimeError(
                    f"{item['id']}: src {src_pil.size} != tgt {edited.size}")

            if not src_out.is_file():
                save_png_atomic(src_pil, src_out)
                (case_dir / "prompt.txt").write_text(item["prompt"] + "\n", encoding="utf-8")
            save_png_atomic(edited, tgt_out)
            if done % 25 == 0 or done == len(todo):
                rate = (time.time() - start) / done
                eta = rate * (len(todo) - done) / 60
                print(f"  [{done}/{len(todo)}] {item['id']} seed={seed}  "
                      f"{rate:.2f}s/img  eta {eta:.1f} min", flush=True)

        del handle, infer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "manifest.json"
    manifest = scan_manifest(args.output_dir, load_tasks(args.assets), seeds)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.output_dir}")


if __name__ == "__main__":
    main()
