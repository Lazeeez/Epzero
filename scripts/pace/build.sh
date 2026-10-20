#!/usr/bin/env bash
# Builds Oryx on a PACE compute node, runs the test suite, and prints a first look at this CPU:
# which SIMD level is active, the distance kernels at every level, and a quick recall run.
# Run it on a compute node (not a login node), e.g.:
#   srun -N1 --exclusive -t 0:30:00 bash scripts/pace/build.sh 2>&1 | tee build-pace.log
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

echo "== node: $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) cores, $(free -g | awk '/Mem:/ {print $2}') GB"
echo "== compilers: $(gcc --version | head -1); $(cmake --version | head -1)"

cmake -S "$ORYX_ROOT/engine" -B "$ORYX_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DORYX_BUILD_NODE=OFF
cmake --build "$ORYX_BUILD" -j "$(nproc)"

echo "== tests"
ctest --test-dir "$ORYX_BUILD" -j "$(nproc)" --output-on-failure

echo "== distance kernels (every SIMD level this CPU supports)"
"$ORYX_BUILD/bench_distance"

echo "== quick recall check: 200k synthetic vectors, built on all cores, searched on one"
"$ORYX_BUILD/bench_recall" --synthetic --n 200000 --queries 1000 --threads "$(nproc)" --ef 10,40,80,200
