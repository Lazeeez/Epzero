#!/usr/bin/env bash
# SIFT1M on a dedicated Gold 6226 node, with the README's methodology: M=16, efConstruction=200,
# 10k queries, recall@10 against the official ground truth, one search thread.
#   1. Build once on all cores (timed) and save the index.
#   2. Search three times from the saved index, pinned to one socket and its local memory, so a
#      search never reads the other socket's RAM. The README reports the median of the three.
# Results: pace-results/gold6226/sift1m-{build,run1,run2,run3}.json (untracked; copy them back with
# scp, and they're committed to bench/results/pace/ from there)
#
# Submit from the repository root, after scripts/download_sift1m.sh on the login node:
#   sbatch scripts/pace/sift1m.sbatch
#SBATCH --job-name=oryx-sift1m
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --time=01:00:00
#SBATCH --output=sift1m-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
[ -x "$ORYX_BUILD/bench_recall" ] || { echo "build first: scripts/pace/build.sh"; exit 1; }
[ -f data/sift/sift_base.fvecs ] || { echo "download first: scripts/download_sift1m.sh"; exit 1; }

OUT=pace-results/gold6226
mkdir -p "$OUT" data/indexes
INDEX=data/indexes/sift1m-M16-efc200.oryx
SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --gt data/sift/sift_groundtruth.ivecs
      --queries 10000 --k 10 --M 16 --efc 200 --ef 10,40,80,200,400 --search-threads 1)

# Hardware facts only: no user, account or host names.
lscpu | grep -E "^(Model name|Socket|Core|Thread|NUMA node\(s\)|CPU max MHz|L[123])" > "$OUT/hardware.txt"
free -g | awk '/Mem:/ {print "Memory (GB): " $2}' >> "$OUT/hardware.txt"
gcc --version | head -1 >> "$OUT/hardware.txt"

echo "== build (all $(nproc) cores) + brute-force baseline"
"$ORYX_BUILD/bench_recall" "${SIFT[@]}" --threads "$(nproc)" --save "$INDEX" --json "$OUT/sift1m-build.json"

for run in 1 2 3; do
  echo "== search run $run (pinned to socket 0)"
  numactl --cpunodebind=0 --membind=0 \
    "$ORYX_BUILD/bench_recall" "${SIFT[@]}" --load "$INDEX" --no-baseline --json "$OUT/sift1m-run$run.json"
done
