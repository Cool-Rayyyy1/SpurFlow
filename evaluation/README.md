# EditFlow Evaluation

Image-editing benchmark only ([ImgEdit-Bench](https://github.com/PKU-YuanGroup/ImgEdit)).

## Two-phase workflow

**Phase 1 — teacher generate + score:**

```bash
SKIP_STUDENT=1 bash evaluation/run_imgedit_eval.sh
```

- Generates teacher images (28 steps, guidance 2.5) if not already done
- Runs GPT-4o scoring on Basic + UGE
- Writes summary to `outputs/teacher/scores.txt`

**Phase 2 — student generate + score:**

```bash
bash evaluation/run_imgedit_eval.sh
```

- Skips teacher generation if already complete
- Does **not** re-score teacher (reads `teacher/scores.txt` at the end)
- Generates + scores student using distilled checkpoint

## Output layout

```
evaluation/imgedit_bench/outputs/
├── teacher/
│   ├── basic/ uge/ multiturn/...
│   ├── scores/          # GPT JSON artifacts
│   └── scores.txt       # human-readable summary (persisted)
└── student/<run_name>/
    ├── basic/ uge/ multiturn/...
    ├── scores/
    └── scores.txt
```

## Flags

| Variable | Meaning |
|---|---|
| `GEN_ONLY=1` | Generation only, skip all GPT scoring |
| `SCORE_ONLY=1` | Skip generation, score only |
| `SKIP_STUDENT=1` | Teacher phase |
| `SKIP_TEACHER=1` | Student only (still prints teacher/scores.txt) |
| `FORCE_SCORE=1` | Re-run GPT scoring |
| `CUDA_VISIBLE_DEVICES` | `0,1` (default) | GPUs for generation |
| `NUM_GPUS` | `2` (default) | Parallel workers (1 task list split across GPUs) |

Generation defaults to **2-GPU parallel** on cards 0 and 1. GPT scoring uses API only (no GPU).

## Examples

```bash
# Teacher smoke test (4 samples, no GPT)
SUITE=basic MAX_SAMPLES=4 GEN_ONLY=1 SKIP_STUDENT=1 bash evaluation/run_imgedit_eval.sh

# Re-score teacher only
SCORE_ONLY=1 SKIP_STUDENT=1 FORCE_SCORE=1 bash evaluation/run_imgedit_eval.sh
```

See also: [imgedit_bench/README.md](imgedit_bench/README.md)
