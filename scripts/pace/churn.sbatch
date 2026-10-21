#!/usr/bin/env bash
# Search under writes, on SIFT1M, on a dedicated Gold 6226 node, all on one socket (12 cores):
#   1. Mixed load: 8 reader threads search for 30 s, alone and then with 2 writer threads deleting
#      and re-inserting vectors. Search p50/p99/p99.9, throughput and recall for each.
#   2. Delete/re-insert cycles on a fresh index: 5 cycles of 10%, then compaction. Recall, QPS and
#      slots in use after each.
# Result: pace-results/gold6226/sift1m-churn.json (untracked; scp back)
#
#   sbatch scripts/pace/churn.sbatch
#SBATCH --job-name=epzero-churn
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=01:00:00
#SBATCH --output=churn-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
[ -f data/sift/sift_base.fvecs ] || { echo "download first: scripts/download_sift1m.sh"; exit 1; }
cmake --build "$EPZERO_BUILD" --target bench_churn -j "$(nproc)"

OUT=pace-results/gold6226
mkdir -p "$OUT"
numactl --cpunodebind=0 --membind=0 "$EPZERO_BUILD/bench_churn" \
  --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --gt data/sift/sift_groundtruth.ivecs \
  --queries 10000 --threads 12 --readers 8 --writers 2 --seconds 30 --cycles 5 --churn 0.1 --ef 80 \
  --json "$OUT/sift1m-churn.json"
