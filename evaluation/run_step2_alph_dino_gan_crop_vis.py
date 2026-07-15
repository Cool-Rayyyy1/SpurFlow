#!/usr/bin/env python3
"""Smoke visualization: step-2 alpha mask crops on pico-banana val samples.

Outputs per sample under <output_dir>/<sample_id>/:
  prompt.txt, src.png, edit.png, fake.png
  step2_alpha_heatmap.png
  step2_edit_mass_heatmap.png
  crop_boxes_overlay.png
  mask_crop_zoom.png
  crop_panel.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
IMGEDIT_EVAL = EVAL_ROOT / 'imgedit_bench'
EDITFLOW_ROOT = EVAL_ROOT.parent
for path in (EDITFLOW_ROOT, IMGEDIT_EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha_model_utils import build_alpha_vis_model  # noqa: E402
from alpha_vis import capture_student_alphas, render_continuous_alpha_on_src  # noqa: E402
from crop_mask_vis import (  # noqa: E402
    build_crop_specs_for_alpha,
    crop_region_from_spec,
    make_crop_panel,
    render_crop_boxes_overlay,
    render_edit_mass_heatmap,
)
from run_editflow_imgedit_infer import (  # noqa: E402
    pil_to_tensor,
    preprocess_image_for_student,
    tensor_to_pil,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Visualize step-2 alpha mask GAN crops on val set.')
    p.add_argument('--data_root', type=Path,
                   default=Path('/mnt/afs_zhangyunzhe/dataset/pico-banana-400k'))
    p.add_argument('--jsonl', type=str, default='jsonl/sft_with_local_source_image_path.jsonl')
    p.add_argument('--edited_images_dir', type=str, default='edited_images')
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--config', type=Path,
                   default=EDITFLOW_ROOT / 'configs/kontext/editflux_uedit_fixedeps_2nfe_k16_alpha_data.py')
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--num_samples', type=int, default=8)
    p.add_argument('--val_start_ind', type=int, default=-128,
                   help='Same as training val: last 128 rows when -128.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_inference_steps', type=int, default=2)
    p.add_argument('--guidance_scale', type=float, default=3.5)
    p.add_argument('--device', type=str, default='cuda:0')
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
    p.add_argument('--skip_existing', action='store_true')
    return p.parse_args()


def load_val_records(
        data_root: Path,
        jsonl_name: str,
        start_ind: int,
        num_samples: int) -> List[Dict]:
    jsonl_path = data_root / jsonl_name
    records: List[Dict] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if start_ind < 0:
        records = records[start_ind:]
    return records[:num_samples]


def resolve_image_path(data_root: Path, record: Dict, edited_dir: str) -> tuple[Path, Optional[Path]]:
    src = Path(record['local_input_image'])
    if not src.is_file():
        src = data_root / record.get('open_image_input_url', '').split('/')[-1]
    edit_rel = record.get('output_image')
    edit_path = None
    if edit_rel:
        candidate = data_root / edited_dir / Path(edit_rel).name
        if candidate.is_file():
            edit_path = candidate
        else:
            candidate = data_root / edit_rel
            if candidate.is_file():
                edit_path = candidate
    return src, edit_path


def main() -> None:
    args = parse_args()
    os.environ.setdefault('STUDENT_RESIZE_MODE', 'kontext')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_val_records(
        args.data_root, args.jsonl, args.val_start_ind, args.num_samples)
    print(f'Loaded {len(records)} val records from {args.data_root}')

    model = build_alpha_vis_model(args.config, args.ckpt, args.device)
    model.eval()

    for idx, record in enumerate(tqdm(records, desc='crop-vis')):
        sample_id = f'val_{args.val_start_ind + idx:05d}'
        out_dir = args.output_dir / sample_id
        panel_path = out_dir / 'crop_panel.png'
        if args.skip_existing and panel_path.is_file():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        src_path, edit_path = resolve_image_path(
            args.data_root, record, args.edited_images_dir)
        prompt = record.get('text') or record.get('summarized_text') or ''
        (out_dir / 'prompt.txt').write_text(prompt, encoding='utf-8')

        src_raw = Image.open(src_path).convert('RGB')
        src_pil, alphas, fake_pil = capture_student_alphas(
            model,
            src_raw,
            prompt,
            args.num_inference_steps,
            args.guidance_scale,
            args.seed + idx,
            args.device,
            preprocess_image_for_student,
            pil_to_tensor,
            return_edited=True,
            tensor_to_pil_fn=tensor_to_pil,
        )
        src_pil.save(out_dir / 'src.png')
        fake_pil.save(out_dir / 'fake.png')
        if edit_path is not None:
            edit_pil = preprocess_image_for_student(Image.open(edit_path).convert('RGB'))
            edit_pil.save(out_dir / 'edit.png')

        if len(alphas) < 2:
            print(f'WARN {sample_id}: expected 2 NFE alphas, got {len(alphas)}', file=sys.stderr)
            continue
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
            seed=args.seed + idx,
        )

        render_continuous_alpha_on_src(
            src_pil, step2_alpha, step_label='step2',
        ).save(out_dir / 'step2_alpha_heatmap.png')

        render_edit_mass_heatmap(
            src_pil,
            step2_alpha,
            edit_is_low_alpha=args.edit_is_low_alpha,
            mass_threshold_percentile=args.mass_threshold_percentile,
            step_label='step2',
        ).save(out_dir / 'step2_edit_mass_heatmap.png')

        local_enabled = bool(meta['local_enabled'][0].item())
        mask_fallback = bool(meta['mask_fallback'][0].item())
        render_crop_boxes_overlay(
            src_pil,
            crop_specs,
            batch_idx=0,
            mask_fallback=mask_fallback,
            local_enabled=local_enabled,
        ).save(out_dir / 'crop_boxes_overlay.png')

        if local_enabled and len(crop_specs) >= 3:
            crop_region_from_spec(
                src_pil, crop_specs[2][0].tolist(),
            ).save(out_dir / 'mask_crop_zoom.png')

        make_crop_panel(
            src_pil,
            step2_alpha,
            crop_specs,
            meta,
            batch_idx=0,
            edit_is_low_alpha=args.edit_is_low_alpha,
            mass_threshold_percentile=args.mass_threshold_percentile,
        ).save(out_dir / 'crop_panel.png')

        spec_json = {
            'local_enabled': local_enabled,
            'mask_fallback': mask_fallback,
            'global': crop_specs[0][0].tolist(),
        }
        if local_enabled and len(crop_specs) >= 3:
            spec_json['random_local'] = crop_specs[1][0].tolist()
            spec_json['mask_local'] = crop_specs[2][0].tolist()
        (out_dir / 'crop_specs.json').write_text(
            json.dumps(spec_json, indent=2), encoding='utf-8')

    print(f'Done. Outputs: {args.output_dir}')


if __name__ == '__main__':
    main()
