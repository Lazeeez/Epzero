#!/usr/bin/env bash
# Search throughput from 1 to 24 threads on SIFT1M, on a dedicated Gold 6226 node (2 sockets x 12
# cores, no hyperthreading). Uses the index saved by sift1m.sbatch, so nothing is rebuilt.
#   - 1 to 12 threads on one socket, with its local memory: pure core scaling.
#   - 12 to 24 threads over both sockets, memory interleaved across them: the cost of NUMA shows
#     up as the gap between the two 12-thread points.
# Each point is measured three times at ef=80 and ef=200; --repeat grows with the thread count so
# every measurement lasts a couple of seconds.
# Results: pace-results/gold6226/scaling-<placement>-t<threads>-run<n>.json (untracked; scp back)
#
#   sbatch scripts/pace/scaling.sbatch
#SBATCH --job-name=epzero-scaling
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=00:45:00
#SBATCH --output=scaling-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
INDEX=data/indexes/sift1m-M16-efc200.epzero
[ -x "$EPZERO_BUILD/bench_recall" ] || { echo "build first: scripts/pace/build.sh"; exit 1; }
[ -f "$INDEX" ] || { echo "run scripts/pace/sift1m.sbatch first (it saves the index)"; exit 1; }

# Pick up benchmark changes since the last build (incremental, a few seconds).
cmake --build "$EPZERO_BUILD" --target bench_recall -j "$(nproc)"

OUT=pace-results/gold6226
mkdir -p "$OUT"
SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --gt data/sift/sift_groundtruth.ivecs
      --queries 10000 --k 10 --ef 80,200 --load "$INDEX" --no-baseline)

point() {  # placement threads numactl-args...
  local placement=$1 threads=$2
  shift 2
  for run in 1 2 3; do
    echo "== $placement, $threads threads, run $run"
    numactl "$@" "$EPZERO_BUILD/bench_recall" "${SIFT[@]}" --search-threads "$threads" --repeat "$threads" \
      --json "$OUT/scaling-$placement-t$threads-run$run.json" | grep -E "^ +(80|200) "
  done
}

for t in 1 2 4 6 8 10 12; do point socket0 "$t" --cpunodebind=0 --membind=0; done
for t in 12 16 20 24; do point both "$t" --interleave=all; done
