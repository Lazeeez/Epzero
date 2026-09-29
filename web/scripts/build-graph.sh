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
# Parameters must match manifest.json ("graph") and the fallback build in collection.worker.ts.
"$build/oryx-export-graph" "$data/vectors.i8" "$data/scales.f32" 512 16 128 "$data/graph.bin"
