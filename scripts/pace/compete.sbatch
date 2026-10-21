#!/usr/bin/env bash
# Oryx against hnswlib, FAISS (HNSW and tuned IVF-PQ), USearch, VSAG and Glass on one
# ann-benchmarks dataset, on a dedicated Gold 6226 node, with bench/compete.py's fairness rules
# (same graph degree and build beam, one process, interleaved rounds, median QPS, one search thread).
#   1. Build every index on all cores (timed, cached).
#   2. Search from the cached indexes, pinned to one socket and its local memory.
# Result: pace-results/gold6226/<dataset>-compete.json (untracked; copy it back with scp)
#
# After scripts/pace/bench-env.sh, scripts/pace/compete-env.sh and
# scripts/download_ann_dataset.sh <dataset>, submit one job per dataset:
#   sbatch --export=ALL,DATASET=gist-960-euclidean scripts/pace/compete.sbatch
#   sbatch --export=ALL,DATASET=glove-100-angular scripts/pace/compete.sbatch
#   sbatch --export=ALL,DATASET=dbpedia-openai-1000k-angular scripts/pace/compete.sbatch
#   sbatch --export=ALL,DATASET=deep-image-96-angular --time=10:00:00 scripts/pace/compete.sbatch
#SBATCH --job-name=oryx-compete
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=05:00:00
#SBATCH --output=compete-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
: "${DATASET:?pass the dataset: sbatch --export=ALL,DATASET=gist-960-euclidean ...}"
VENV="$ORYX_ROOT/.pace-bench"
DATA="data/ann/$DATASET.hdf5"
[ -x "$VENV/bin/python" ] || { echo "set up first: bash scripts/pace/bench-env.sh && bash scripts/pace/compete-env.sh"; exit 1; }
[ -f "$DATA" ] || { echo "download first: scripts/download_ann_dataset.sh $DATASET"; exit 1; }

# Libraries that compile with -march=native are built here, on this node's CPU.
"$VENV/bin/pip" install --quiet --no-index --no-build-isolation --no-deps --force-reinstall "$VENV"/src/hnswlib-*.tar.gz
if [ -d "$VENV/src/pyglass/python" ]; then
  "$VENV/bin/pip" install --quiet --no-build-isolation --no-deps --force-reinstall "$VENV/src/pyglass/python" || echo "glass did not build: skipped"
fi
# Oryx from this checkout (it picks its SIMD kernels at run time).
"$VENV/bin/pip" install --quiet --no-deps --force-reinstall "$ORYX_ROOT/engine"

OUT=pace-results/gold6226
CACHE=data/indexes/compete-gold6226
mkdir -p "$OUT" "$CACHE"
COMMON=(--data "$DATA" --cache "$CACHE" --M 16 --efc 200 --threads "$(nproc)")

echo "== build every index on $(nproc) cores"
"$VENV/bin/python" bench/compete.py "${COMMON[@]}" --build-only --out "$OUT/$DATASET-compete.json"

echo "== search, one thread, pinned to socket 0"
numactl --cpunodebind=0 --membind=0 \
  "$VENV/bin/python" bench/compete.py "${COMMON[@]}" --rounds 3 --out "$OUT/$DATASET-compete.json"
