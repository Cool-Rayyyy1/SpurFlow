#!/usr/bin/env python3
"""Official Qwen-Image-Edit-2511 on ImgEdit-Bench.

HF: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
Resize: qwen. Sharding: metadata[split_id::split_num].
Suite: basic + uge only (no multiturn), same layout as flux_kontext_official:

  {output_dir}/basic/{key}.png
  {output_dir}/basic/{Category}/{key}/{src.png,pred.png,prompt.txt}
  {output_dir}/uge/{key}.png

  CUDA_VISIBLE_DEVICES=0 python simple_run/qwen_image_edit/run_imgedit.py --split_id 0 --split_num 8
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIMPLE_RUN = HERE.parent
for path in (str(HERE), str(SIMPLE_RUN)):
    if path not in sys.path:
        sys.path.insert(0, path)

import qwen_common as defaults  # noqa: E402
from official_runners import run_imgedit  # noqa: E402

if __name__ == "__main__":
    run_imgedit("qwen", defaults, extra_sys_path=[str(HERE), str(SIMPLE_RUN)])
