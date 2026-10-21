#!/usr/bin/env bash
# Rebuilds web/public/data/openimages/graph.bin, the prebuilt HNSW graph the site loads instead of
# building one in every visitor's browser. Run after the vectors change (pipeline/build_dataset.py)
# or the engine's graph construction changes. Needs a native engine build (cmake).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
data="$here/public/data/openimages"
build=${BUILD_DIR:-"$here/../engine/build/release"}
cmake -S "$here/../engine" -B "$build" -DCMAKE_BUILD_TYPE=Release -DORYX_BUILD_TESTS=OFF >/dev/null
cmake --build "$build" --target oryx_export_graph -j
# The tool reads row-major int8 vectors and their scales: unpack them from the progressive file
# (pipeline/progressive_vectors.py), which stores them dimension-block by dimension-block.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python3 - "$data/vectors.v2.bin" "$tmp" <<'PY'
import json, os, sys
import numpy as np
path, out = sys.argv[1], sys.argv[2]
manifest = json.load(open(os.path.join(os.path.dirname(path), "manifest.json")))
n, dim = manifest["count"], manifest["dim"]
raw, pos, lo, codes, scales = open(path, "rb").read(), 0, 0, [], []
for stage in manifest["vectors"]["stages"]:
    w = stage["dims"] - lo
    scales.append(np.frombuffer(raw, np.float32, w, pos)); pos += 4 * w + 2 * w * dim
    codes.append(np.frombuffer(raw, np.int8, n * w, pos).reshape(n, w)); pos += n * w
    lo = stage["dims"]
np.hstack(codes).tofile(os.path.join(out, "vectors.i8"))
np.concatenate(scales).tofile(os.path.join(out, "scales.f32"))
PY
# Parameters must match manifest.json ("graph") and the fallback build in collection.worker.ts.
"$build/oryx-export-graph" "$tmp/vectors.i8" "$tmp/scales.f32" 512 16 128 "$data/graph.bin"
