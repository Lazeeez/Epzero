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
VENV="$EPZERO_ROOT/.pace-bench"
[ -x "$VENV/bin/python" ] || { echo "run scripts/pace/bench-env.sh first"; exit 1; }

"$VENV/bin/pip" install --quiet h5py usearch pyvsag pyarrow huggingface_hub "pybind11<3" setuptools wheel
# Pinned to the commit VIBE builds (vibe/algorithms/glass/image.def), so both benchmarks run one Glass.
GLASS_COMMIT=d2296ec447d2374ee8f88c6d3b85be1b1e434ad3
rm -rf "$VENV/src/pyglass"
git init --quiet "$VENV/src/pyglass"
git -C "$VENV/src/pyglass" fetch --quiet --depth 1 https://github.com/zilliztech/pyglass.git "$GLASS_COMMIT"
git -C "$VENV/src/pyglass" checkout --quiet FETCH_HEAD
echo "pyglass $(git -C "$VENV/src/pyglass" rev-parse HEAD)"
"$VENV/bin/python" -c "import h5py, usearch, pyvsag; print('h5py', h5py.__version__, '| usearch', usearch.__version__, '| pyvsag ok')"
