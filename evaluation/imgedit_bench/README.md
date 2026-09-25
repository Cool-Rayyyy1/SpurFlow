# ImgEdit-Bench for EditFlow

Benchmark from [PKU-YuanGroup/ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit) (NeurIPS 2025 D&B).

**Entry point:** `bash evaluation/run_imgedit_eval.sh` (see [../README.md](../README.md)).

## Data

Downloaded to:

```
/path/to/dataset/imgedit/
├── Benchmark.tar
└── benchmark/Benchmark/
    ├── singleturn/     # Basic-Bench images
    ├── hard/           # UGE-Bench images
    └── multiturn/      # Multi-turn images + annotation.json per sub-suite
```

Repo folder names vs tarball names:

| Suite | Repo (`Benchmark/`) | Images (`Benchmark.tar`) |
|---|---|---|
| Basic | `Basic/` (scripts only) | `singleturn/` |
| UGE | `UGE/` (scripts only) | `hard/` |
| Multiturn | `Multiturn/` (readme) | `multiturn/` |

## Scripts

| Script | Purpose |
|---|---|
| `run_editflow_imgedit_infer.py` | Generate edits (student or teacher) |
| `run_imgedit_score.py` | GPT-4o scoring on student outputs |
| `download_imgedit_bench.sh` | Download + extract benchmark data |

## Multiturn output format

Each sample saves chained turns under `multiturn/<sub_suite>/`:

```
1_turn1.png   # edit from source
1_turn2.png   # edit from turn1 output
1_turn3.png   # ...
```

Multiturn has no automated GPT scorer upstream — outputs are for manual inspection.

## GPT scoring

Credentials in `openai.env`. After generation, scores are written to `<output>/scores.txt`.

**Phase 1 — teacher:**

```bash
SKIP_STUDENT=1 bash evaluation/run_imgedit_eval.sh
# -> outputs/teacher/scores.txt
```

**Phase 2 — student** (teacher scores loaded from txt at the end):

```bash
bash evaluation/run_imgedit_eval.sh
# -> outputs/student/<run_name>/scores.txt
```
