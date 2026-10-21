#!/usr/bin/env bash
# Adds the competitive benchmark's libraries to the PACE benchmark environment (made by
# bench-env.sh). Run once on a login node, which has internet access:
#   bash scripts/pace/compete-env.sh
#
# USearch, VSAG, FAISS and h5py install as wheels. Glass is only published as source and builds
# with -march=native, so, like hnswlib, its source is fetched here and each job compiles it on its
# own node. It needs pybind11 2.x (it does not compile against pybind11 3).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
VENV="$ORYX_ROOT/.pace-bench"
[ -x "$VENV/bin/python" ] || { echo "run scripts/pace/bench-env.sh first"; exit 1; }

"$VENV/bin/pip" install --quiet h5py usearch pyvsag pyarrow huggingface_hub "pybind11<3" setuptools wheel
rm -rf "$VENV/src/pyglass"
git clone --quiet --depth 1 https://github.com/zilliztech/pyglass.git "$VENV/src/pyglass"
"$VENV/bin/python" -c "import h5py, usearch, pyvsag; print('h5py', h5py.__version__, '| usearch', usearch.__version__, '| pyvsag ok')"
