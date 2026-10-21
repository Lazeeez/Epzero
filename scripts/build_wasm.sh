#!/usr/bin/env bash
# Builds the engine to WebAssembly (threaded + single-threaded fallback) and copies the modules
# into web/public/wasm/{mt,st}. Requires an activated Emscripten SDK (emcmake on PATH).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build/wasm}"

for variant in mt st; do
  threads=$([ "$variant" = mt ] && echo ON || echo OFF)
  emcmake cmake -S "$ROOT/engine" -B "$BUILD/$variant" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DEPZERO_BUILD_BENCH=OFF -DEPZERO_BUILD_TESTS=OFF -DEPZERO_WASM_THREADS="$threads" > /dev/null
  cmake --build "$BUILD/$variant" --target epzero_web
  mkdir -p "$ROOT/web/public/wasm/$variant"
  cp "$BUILD/$variant/epzero.js" "$BUILD/$variant/epzero.wasm" "$ROOT/web/public/wasm/$variant/"
done
ls -l "$ROOT"/web/public/wasm/*/
