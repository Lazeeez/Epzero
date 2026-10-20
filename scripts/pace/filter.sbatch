#!/usr/bin/env bash
# Filtered search on SIFT1M, on a dedicated Gold 6226 node: recall@10 and single-thread QPS for
# every strategy (exact, inline, two-hop) and for Auto, at filter selectivities from 0.1% to 90%,
# against exact filtered ground truth. Uses the index saved by sift1m.sbatch.
# Result: pace-results/gold6226/sift1m-filter.json (untracked; scp back)
#
#   sbatch scripts/pace/filter.sbatch
#SBATCH --job-name=oryx-filter
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=01:30:00
#SBATCH --output=filter-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
INDEX=data/indexes/sift1m-M16-efc200.oryx
[ -f "$INDEX" ] || { echo "run scripts/pace/sift1m.sbatch first (it saves the index)"; exit 1; }

# Pick up engine and benchmark changes since the last build (incremental).
cmake --build "$ORYX_BUILD" --target bench_filter -j "$(nproc)"

OUT=pace-results/gold6226
mkdir -p "$OUT"
# One process computes the filtered ground truth on every core and then times the searches on one
# thread. Its memory is interleaved over both sockets, so wherever that thread runs, about half
# its reads are local: comparable across strategies, a little slower than the socket-pinned runs.
numactl --interleave=all "$ORYX_BUILD/bench_filter" \
  --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --queries 1000 --load "$INDEX" \
  --threads "$(nproc)" --selectivity 0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,0.9 \
  --ef 10,20,40,80,160,320 --json "$OUT/sift1m-filter.json"
