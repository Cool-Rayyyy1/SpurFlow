#!/usr/bin/env python3
"""ImgEdit-Bench crop-mask vis: 9 categories × N cases.

Layout:
  <output_dir>/<category>/<case_key>/
    prompt.txt
    src.png
    edit.png
    crop_boxes.png      # global / random-local / mask-local boxes on src
    alpha_overlay.png   # step2 continuous alpha overlay on src
    crop_specs.json
  <output_dir>/<category>/overview.png   # N rows × 4 cols summary strip
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
EVAL_PARENT = EVAL_ROOT.parent
EDITFLOW_ROOT = EVAL_ROOT.parents[1]
for path in (EDITFLOW_ROOT, EVAL_PARENT, EVAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha_model_utils import build_alpha_vis_model  # noqa: E402
from alpha_vis import capture_student_alphas, render_continuous_alpha_on_src  # noqa: E402
from crop_mask_vis import (  # noqa: E402
    build_crop_specs_for_alpha,
    render_crop_boxes_overlay,
)
from run_editflow_imgedit_infer import (  # noqa: E402
    DEFAULT_BENCH_ROOT,
    pil_to_tensor,
    preprocess_image_for_student,
    tensor_to_pil,
)

DEFAULT_CATEGORIES = (
    'action',
    'add',
    'adjust',
    'background',
    'compose',
    'extract',
    'remove',
    'replace',
    'style',
)

COL_LABELS = ('src', 'edit', 'crop_boxes', 'alpha_overlay')
COL_FILES = ('src.png', 'edit.png', 'crop_boxes.png', 'alpha_overlay.png')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='ImgEdit crop-mask vis: category/case folders with src/edit/crop/alpha.')
    p.add_argument('--bench_root', type=Path, default=DEFAULT_BENCH_ROOT)
    p.add_argument(
        '--annotations_path',
        type=Path,
        default=EVAL_ROOT / 'annotations' / 'basic_edit.json',
    )
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--num_inference_steps', type=int, default=2)
    p.add_argument('--guidance_scale', type=float, default=3.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--samples_per_category', type=int, default=5)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--skip_existing', action='store_true')
    p.add_argument('--overlay_blend', type=float, default=0.52)
    p.add_argument('--p_disable_local', type=float, default=0.0)
    p.add_argument('--edit_is_low_alpha', action='store_true', default=True)
    p.add_argument('--alpha_smooth_sigma', type=float, default=2.0)
    p.add_argument('--mass_threshold_percentile', type=float, default=30.0)
    p.add_argument('--mass_coverage_min', type=float, default=0.85)
    p.add_argument('--mass_coverage_max', type=float, default=0.90)
    p.add_argument('--union_area_max_ratio', type=float, default=0.55)
    p.add_argument('--bbox_expand_factor', type=float, default=1.4)
    p.add_argument('--min_crop_area_ratio', type=float, default=0.05)
    p.add_argument('--max_crop_area_ratio', type=float, default=0.55)
    p.add_argument('--min_edit_mass_ratio', type=float, default=0.002)
    p.add_argument('--min_component_pixels', type=int, default=16)
    return p.parse_args()


def load_category_examples(
        annotations_path: Path,
        bench_root: Path,
        samples_per_category: int,
        seed: int,
        categories: Sequence[str] = DEFAULT_CATEGORIES) -> List[Dict]:
    raw = json.loads(annotations_path.read_text(encoding='utf-8'))
    by_cat: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
    for key, item in raw.items():
        cat = str(item.get('edit_type', '')).lower()
        if cat in categories:
            by_cat[cat].append((key, item))

    rng = random.Random(seed)
    samples: List[Dict] = []
    for cat in categories:
        pool = sorted(by_cat.get(cat, []), key=lambda x: x[0])
        if not pool:
            continue
        if len(pool) <= samples_per_category:
            chosen = pool
        else:
            chosen = sorted(rng.sample(pool, samples_per_category), key=lambda x: x[0])
        for key, item in chosen:
            samples.append(dict(
                key=key,
                category=cat,
                example_name=str(key),
                prompt=item['prompt'],
                source_path=str(bench_root / 'singleturn' / item['id']),
            ))
    return samples


def case_dir(output_dir: Path, category: str, example_name: str) -> Path:
    return output_dir / category / example_name


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _resize_thumb(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.BICUBIC)


def render_category_overview(
        category_dir: Path,
        case_names: Sequence[str],
        thumb_size: int = 256,
) -> Image.Image:
    """Build category overview: rows=cases, cols=src/edit/crop_boxes/alpha_overlay."""
    header_h = 36
    label_w = 120
    pad = 8
    font = _load_font(14)
    small = _load_font(11)

    rows = []
    for case_name in case_names:
        case_path = category_dir / case_name
        row_imgs = []
        for fname in COL_FILES:
            path = case_path / fname
            if path.is_file():
                row_imgs.append(_resize_thumb(Image.open(path).convert('RGB'), thumb_size))
            else:
                blank = Image.new('RGB', (thumb_size, thumb_size), (32, 36, 44))
                ImageDraw.Draw(blank).text((12, 12), 'missing', fill=(180, 180, 180))
                row_imgs.append(blank)
        row_w = label_w + len(row_imgs) * (thumb_size + pad)
        row_h = thumb_size + pad
        row_canvas = Image.new('RGB', (row_w, row_h + header_h), (18, 22, 28))
        draw = ImageDraw.Draw(row_canvas)
        draw.text((8, header_h + thumb_size // 2 - 8), case_name, fill=(220, 220, 220), font=small)
        x = label_w
        for img in row_imgs:
            row_canvas.paste(img, (x, header_h))
            x += thumb_size + pad
        rows.append(row_canvas)

    col_header_w = label_w + len(COL_FILES) * (thumb_size + pad)
    header = Image.new('RGB', (col_header_w, header_h), (18, 22, 28))
    draw = ImageDraw.Draw(header)
    x = label_w
    for label in COL_LABELS:
        draw.text((x + 8, 10), label, fill=(125, 211, 252), font=font)
        x += thumb_size + pad

    total_h = header_h + sum(r.height for r in rows)
    canvas = Image.new('RGB', (col_header_w, total_h), (18, 22, 28))
    y = 0
    canvas.paste(header, (0, y))
    y += header.height
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    title_bar = Image.new('RGB', (col_header_w, 40), (12, 16, 22))
    ImageDraw.Draw(title_bar).text(
        (12, 10), f'{category_dir.name.upper()}  ·  crop-mask vis', fill=(241, 245, 249), font=font)
    out = Image.new('RGB', (col_header_w, total_h + 40), (12, 16, 22))
    out.paste(title_bar, (0, 0))
    out.paste(canvas, (0, 40))
    return out


def process_samples(
        model,
        samples: List[Dict],
        output_dir: Path,
        args: argparse.Namespace) -> None:
    if args.num_inference_steps != 2:
        raise ValueError('Crop-mask bench vis expects nfe=2 for step2 alpha/crops.')

    by_category: Dict[str, List[str]] = defaultdict(list)

    for idx, sample in enumerate(tqdm(samples, desc='crop-mask-bench')):
        out_case = case_dir(output_dir, sample['category'], sample['example_name'])
        done_flag = out_case / 'alpha_overlay.png'
        if args.skip_existing and done_flag.is_file():
            by_category[sample['category']].append(sample['example_name'])
            continue

        src_path = Path(sample['source_path'])
        if not src_path.is_file():
            raise FileNotFoundError(f'Missing source image: {src_path}')

        task_seed = args.seed + idx
        image = Image.open(src_path).convert('RGB')
        src_pil, alphas, edit_pil = capture_student_alphas(
            model,
            image,
            sample['prompt'],
            args.num_inference_steps,
            args.guidance_scale,
            task_seed,
            args.device,
            preprocess_image_for_student,
            pil_to_tensor,
            return_edited=True,
            tensor_to_pil_fn=tensor_to_pil,
        )
        if len(alphas) < 2:
            raise RuntimeError(
                f'Expected 2 alpha maps, got {len(alphas)} for {sample["category"]}/{sample["key"]}')

        step2_alpha = alphas[1]
        img_h, img_w = src_pil.height, src_pil.width
        crop_specs, meta = build_crop_specs_for_alpha(
            step2_alpha,
            img_h,
            img_w,
            p_disable_local=args.p_disable_local,
            edit_is_low_alpha=args.edit_is_low_alpha,
            alpha_smooth_sigma=args.alpha_smooth_sigma,
            mass_threshold_percentile=args.mass_threshold_percentile,
            mass_coverage_min=args.mass_coverage_min,
            mass_coverage_max=args.mass_coverage_max,
            union_area_max_ratio=args.union_area_max_ratio,
            bbox_expand_factor=args.bbox_expand_factor,
            min_crop_area_ratio=args.min_crop_area_ratio,
            max_crop_area_ratio=args.max_crop_area_ratio,
            min_edit_mass_ratio=args.min_edit_mass_ratio,
            min_component_pixels=args.min_component_pixels,
            seed=task_seed,
        )

        local_enabled = bool(meta['local_enabled'][0].item())
        mask_fallback = bool(meta['mask_fallback'][0].item())

        crop_overlay = render_crop_boxes_overlay(
            src_pil,
            crop_specs,
            batch_idx=0,
            mask_fallback=mask_fallback,
            local_enabled=local_enabled,
            title='GAN crop boxes (step2 alpha mask)',
        )
        alpha_overlay = render_continuous_alpha_on_src(
            src_pil,
            step2_alpha,
            blend=args.overlay_blend,
            step_label='step 2/2',
            title_prefix='Alpha overlay',
            draw_grid=True,
        )

        out_case.mkdir(parents=True, exist_ok=True)
        (out_case / 'prompt.txt').write_text(sample['prompt'].rstrip() + '\n', encoding='utf-8')
        src_pil.save(out_case / 'src.png')
        edit_pil.save(out_case / 'edit.png')
        crop_overlay.save(out_case / 'crop_boxes.png')
        alpha_overlay.save(out_case / 'alpha_overlay.png')

        spec_json = {
            'local_enabled': local_enabled,
            'mask_fallback': mask_fallback,
            'global': crop_specs[0][0].tolist(),
        }
        if local_enabled and len(crop_specs) >= 2:
            spec_json['mask_local'] = crop_specs[1][0].tolist()
        else:
            spec_json['random_local'] = crop_specs[1][0].tolist()
        (out_case / 'crop_specs.json').write_text(
            json.dumps(spec_json, indent=2), encoding='utf-8')

        by_category[sample['category']].append(sample['example_name'])

    for category, case_names in sorted(by_category.items()):
        if not case_names:
            continue
        overview = render_category_overview(
            output_dir / category,
            sorted(case_names),
        )
        overview.save(output_dir / category / 'overview.png')


def main() -> None:
    args = parse_args()
    os.environ.setdefault('STUDENT_RESIZE_MODE', 'kontext')
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if not args.annotations_path.is_file():
        raise FileNotFoundError(args.annotations_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_category_examples(
        args.annotations_path,
        args.bench_root,
        args.samples_per_category,
        args.seed,
    )
    print(f'Loaded {len(samples)} cases '
          f'({args.samples_per_category} per category × {len(DEFAULT_CATEGORIES)} categories)')

    model = build_alpha_vis_model(args.config, args.ckpt, args.device)
    model.eval()
    process_samples(model, samples, args.output_dir, args)

    print(f'Done. Output root: {args.output_dir}')
    print('Per case: <category>/<case_key>/{src,edit,crop_boxes,alpha_overlay}.png')
    print('Per category: <category>/overview.png')


if __name__ == '__main__':
    main()
