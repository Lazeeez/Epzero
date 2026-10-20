#!/usr/bin/env bash
# Python environment for the library comparisons on PACE (Oryx's Python module, hnswlib, FAISS).
# Run once on a login node, which has internet access; jobs then work offline:
#   bash scripts/pace/bench-env.sh
#
# hnswlib compiles itself with -march=native, and the login node's CPU is newer than the
# benchmark nodes', so only its source is fetched here: each job compiles it on its own node.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

VENV="$ORYX_ROOT/.pace-bench"
/usr/bin/python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet numpy faiss-cpu pybind11 setuptools wheel

mkdir -p "$VENV/src"
"$VENV/bin/pip" download --quiet --no-deps --no-binary :all: hnswlib -d "$VENV/src"

# Oryx picks its SIMD kernels at run time, so building it here is fine for every node.
"$VENV/bin/pip" install --quiet "$ORYX_ROOT/engine"

"$VENV/bin/python" -c "import numpy, faiss, oryx; print('numpy', numpy.__version__, '| faiss', faiss.__version__, '| oryx ok')"
ls "$VENV/src"
