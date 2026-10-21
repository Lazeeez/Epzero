#!/usr/bin/env bash
# Epzero vs hnswlib vs FAISS (IndexHNSWFlat) on SIFT1M, on a dedicated Gold 6226 node, with
# bench/compare_libs.py's fairness rules (same M/efConstruction, one process, interleaved rounds,
# median QPS, single-threaded search).
#   1. Build all three indexes on every core (timed, cached).
#   2. Measure search from the cached indexes, pinned to one socket and its local memory.
# Result: pace-results/gold6226/sift1m-compare.json (untracked; copy it back with scp)
#
# Submit from the repository root, after scripts/download_sift1m.sh and scripts/pace/bench-env.sh:
#   sbatch scripts/pace/compare.sbatch
#SBATCH --job-name=epzero-compare
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=02:00:00
#SBATCH --output=compare-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
VENV="$EPZERO_ROOT/.pace-bench"
[ -x "$VENV/bin/python" ] || { echo "set up first: bash scripts/pace/bench-env.sh"; exit 1; }
[ -f data/sift/sift_base.fvecs ] || { echo "download first: scripts/download_sift1m.sh"; exit 1; }

# hnswlib, compiled for this node's CPU (-march=native), as it would be when installed on it.
"$VENV/bin/pip" install --quiet --no-index --no-build-isolation --no-deps --force-reinstall "$VENV"/src/hnswlib-*.tar.gz

OUT=pace-results/gold6226
CACHE=data/indexes/compare-gold6226
mkdir -p "$OUT" "$CACHE"
COMMON=(--data data/sift --cache "$CACHE" --M 16 --efc 200 --threads "$(nproc)")

echo "== build all three on $(nproc) cores"
"$VENV/bin/python" bench/compare_libs.py "${COMMON[@]}" --rounds 1 --ef 10 --out "$OUT/sift1m-compare-build.json"

echo "== search, pinned to socket 0"
numactl --cpunodebind=0 --membind=0 \
  "$VENV/bin/python" bench/compare_libs.py "${COMMON[@]}" --rounds 5 --out "$OUT/sift1m-compare.json"
