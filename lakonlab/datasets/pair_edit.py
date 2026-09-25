# Copyright (c) 2026 SpurFlow contributors
"""Paired edit data: source image, edited image, and an instruction.

Same resize / latent logic as ``ImageEdit``.
JSONL is built by ``tools/build_pair_jsonl.py``.
"""

from mmgen.datasets.builder import DATASETS
from mmgen.utils import get_root_logger

from .image_edit import ImageEdit


@DATASETS.register_module()
class PairEdit(ImageEdit):
    """Paired source / edited-image records.

    Expected jsonl fields:
      input_path, output_path, instruction, task, uuid
    """

    def __init__(
            self,
            data_root: str = '/path/to/paired_edit_data',
            jsonl_path: str = 'metadata.jsonl',
            source_column: str = 'input_path',
            target_column: str = 'output_path',
            prompt_column: str = 'instruction',
            require_edited: bool = True,
            **kwargs):
        super().__init__(
            data_root=data_root,
            jsonl_path=jsonl_path,
            source_column=source_column,
            target_column=target_column,
            prompt_column=prompt_column,
            require_edited=require_edited,
            **kwargs)
        # Prompt is copied into jsonl ``instruction`` from Edit_Instruction,
        # falling back to gemma_instruction_separate.
        self.prompt_keys = [
            'instruction',
            'Edit_Instruction',
            'gemma_instruction_separate',
        ]
        get_root_logger().info(
            f'PairEdit: {self.end_ind - self.start_ind} pairs from {self.data_root}')
