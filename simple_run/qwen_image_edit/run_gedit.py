#!/usr/bin/env python3
"""Official Qwen-Image-Edit-2511 on GEdit-v2.

HF: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
Resize: qwen. Sharding: metadata[split_id::split_num].

  CUDA_VISIBLE_DEVICES=0 python simple_run/qwen_image_edit/run_gedit.py --split_id 0 --split_num 8
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
from official_runners import run_gedit  # noqa: E402

if __name__ == "__main__":
    run_gedit("qwen", defaults, extra_sys_path=[str(HERE), str(SIMPLE_RUN)])
