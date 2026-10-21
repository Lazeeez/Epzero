#!/usr/bin/env bash
# Python environment for the library comparisons on PACE (Epzero's Python module, hnswlib, FAISS).
# Run once on a login node, which has internet access; jobs then work offline:
#   bash scripts/pace/bench-env.sh
#
# The system Python has no development headers, and PACE's Python modules are hidden under a
# different GCC, so uv provides a complete standalone Python 3.12 (headers included) instead.
# hnswlib compiles itself with -march=native, and the login node's CPU is newer than the
# benchmark nodes', so only its source is fetched here: each job compiles it on its own node.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

"$EPZERO_TOOLS/bin/pip" install --quiet uv
VENV="$EPZERO_ROOT/.pace-bench"
rm -rf "$VENV"
uv venv --seed --python 3.12 "$VENV"   # --seed: include pip, which the jobs use
"$VENV/bin/pip" install --quiet numpy faiss-cpu pybind11 setuptools wheel

# Only hnswlib as source; its build dependencies come as ordinary wheels.
mkdir -p "$VENV/src"
"$VENV/bin/pip" download --quiet --no-deps --no-binary hnswlib hnswlib -d "$VENV/src"

# Epzero picks its SIMD kernels at run time, so building it here is fine for every node.
"$VENV/bin/pip" install --quiet "$EPZERO_ROOT/engine"

"$VENV/bin/python" -c "import sys, numpy, faiss, epzero; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| faiss', faiss.__version__, '| epzero', epzero.simd_level())"
ls "$VENV/src"
