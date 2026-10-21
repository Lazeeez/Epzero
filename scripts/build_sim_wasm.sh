#!/usr/bin/env bash
# Builds the cluster simulator (the real router with simulated nodes, cluster/cmd/epzero-sim) to
# WebAssembly for the site's cluster page, with Go's JavaScript glue, into web/public/sim/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/web/public/sim"
mkdir -p "$OUT"
cd "$ROOT/cluster"
GOOS=js GOARCH=wasm go build -trimpath -ldflags="-s -w" -o "$OUT/epzero-sim.wasm" ./cmd/epzero-sim
install -m 644 "$(go env GOROOT)/lib/wasm/wasm_exec.js" "$OUT/"
ls -l "$OUT"
