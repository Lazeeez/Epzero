#!/usr/bin/env bash
# Recall under drift on SIFT1M, on a dedicated Gold 6226 node: a fixed beam width tuned once, a
# fixed beam width sized for the worst case, and RecallGuard, through steady, harder-query, churn
# and filtered phases, at recall targets 0.90, 0.95 and 0.99.
# Results: pace-results/gold6226/sift1m-guard-<target>.json (untracked; scp back)
#
#   sbatch scripts/pace/guard.sbatch
#SBATCH --job-name=oryx-guard
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=03:00:00
#SBATCH --output=guard-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
[ -f data/sift/sift_base.fvecs ] || { echo "download first: scripts/download_sift1m.sh"; exit 1; }
cmake --build "$ORYX_BUILD" --target bench_guard -j "$(nproc)"

OUT=pace-results/gold6226
mkdir -p "$OUT"
for target in 0.90 0.95 0.99; do
  numactl --interleave=all "$ORYX_BUILD/bench_guard" \
    --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --queries 10000 \
    --batch 2500 --passes 6 --target "$target" --sample-rate 0.02 --threads "$(nproc)" \
    --json "$OUT/sift1m-guard-$target.json"
done
