#!/usr/bin/env bash
# Repair corrupt yapf Grammar*.pickle that break `import mmcv` with:
#   EOFError: Ran out of input
# Common on shared AFS/NFS when multi-process torchrun races while regenerating
# the pickle. Safe to run before every training launch.
#
# Usage:
#   bash tools/fix_yapf_grammar_pickle.sh
#   source setup_env.sh && bash tools/fix_yapf_grammar_pickle.sh

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[yapf-fix] python not found (${PYTHON_BIN})" >&2
    exit 1
fi

YAPF_ROOT="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys

spec = importlib.util.find_spec('yapf_third_party')
if spec is None or not spec.submodule_search_locations:
    sys.exit(0)
print(spec.submodule_search_locations[0])
PY
)"

if [[ -z "${YAPF_ROOT}" ]]; then
    echo "[yapf-fix] yapf_third_party not installed; skip"
    exit 0
fi

mapfile -t PICKLES < <(find "${YAPF_ROOT}" -type f -name '*.pickle' 2>/dev/null || true)
REMOVED=0
for pickle_path in "${PICKLES[@]:-}"; do
    [[ -n "${pickle_path}" ]] || continue
    # Empty / truncated pickles cause EOFError on pickle.load.
    if [[ ! -s "${pickle_path}" ]]; then
        echo "[yapf-fix] removing empty pickle: ${pickle_path}"
        rm -f "${pickle_path}"
        REMOVED=$((REMOVED + 1))
        continue
    fi
    if ! "${PYTHON_BIN}" - "${pickle_path}" <<'PY'
import pickle
import sys
path = sys.argv[1]
try:
    with open(path, 'rb') as f:
        pickle.load(f)
except Exception:
    sys.exit(1)
sys.exit(0)
PY
    then
        echo "[yapf-fix] removing corrupt pickle: ${pickle_path}"
        rm -f "${pickle_path}"
        REMOVED=$((REMOVED + 1))
    fi
done

# Single-process regenerate + verify mmcv import before torchrun fans out.
if ! "${PYTHON_BIN}" - <<'PY'
import mmcv
print(f'[yapf-fix] mmcv ok ({mmcv.__version__})')
PY
then
    echo "[yapf-fix] mmcv import still failing; try: pip install --force-reinstall 'yapf<0.43'" >&2
    exit 1
fi

if [[ "${REMOVED}" -gt 0 ]]; then
    echo "[yapf-fix] removed ${REMOVED} bad pickle(s); grammar will be regenerated on import"
else
    echo "[yapf-fix] yapf grammar pickles look healthy"
fi
