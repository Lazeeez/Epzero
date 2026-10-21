#!/usr/bin/env bash
# Per-query search effort, on a dedicated Gold 6226 node, on SIFT1M and GloVe-100 (the dataset with
# the widest spread of query difficulty):
#   1. bench_adaptive: a fixed beam width for every query against patience (each query stops once
#      its best 10 have settled), one search thread pinned to socket 0: recall, cost and the share of
#      badly answered queries.
#   2. bench_guard: recall under drift, now with the per-query guard as a fourth policy.
# Needs data/sift (scripts/download_sift1m.sh) and data/ann/glove-100-angular.hdf5
# (scripts/download_ann_dataset.sh glove-100-angular).
# Results: pace-results/gold6226/{sift1m,glove}-adaptive.json and {sift1m,glove}-guard-perquery.json
# (untracked; copy them back with scp)
#
#   sbatch scripts/pace/adaptive.sbatch
#SBATCH --job-name=oryx-adaptive
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=03:00:00
#SBATCH --output=adaptive-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
VENV="$ORYX_ROOT/.pace-bench"
[ -f data/sift/sift_base.fvecs ] || { echo "download first: scripts/download_sift1m.sh"; exit 1; }
[ -f data/ann/glove-100-angular.hdf5 ] || { echo "download first: scripts/download_ann_dataset.sh glove-100-angular"; exit 1; }
cmake --build "$ORYX_BUILD" --target bench_adaptive bench_guard -j "$(nproc)"
[ -f data/ann/glove_base.fvecs ] || "$VENV/bin/python" scripts/hdf5_to_vecs.py data/ann/glove-100-angular.hdf5 data/ann/glove

OUT=pace-results/gold6226
mkdir -p "$OUT" data/indexes
SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs)
GLOVE=(--base data/ann/glove_base.fvecs --query data/ann/glove_query.fvecs --metric cosine)

echo "== patience against a fixed beam"
numactl --cpunodebind=0 --membind=0 "$ORYX_BUILD/bench_adaptive" "${SIFT[@]}" --truth data/sift/sift_groundtruth.ivecs \
  --name sift1m --index data/indexes/sift1m-M16-efc200.oryx --threads "$(nproc)" --rounds 3 \
  --efs 16,24,32,48,64,96,128,192,256 --patience 8,12,16,24,32,48,64 --caps 48,96,192 --json "$OUT/sift1m-adaptive.json"
numactl --cpunodebind=0 --membind=0 "$ORYX_BUILD/bench_adaptive" "${GLOVE[@]}" --truth data/ann/glove_groundtruth.ivecs \
  --name glove-100-angular --index data/indexes/glove-M16-efc200.oryx --threads "$(nproc)" --rounds 3 \
  --efs 32,48,64,96,128,192,256,384,512,768,1024,1536 --patience 16,32,64,96,128,192,256,384,512 --caps 384,768,1536 \
  --json "$OUT/glove-adaptive.json"

echo "== recall under drift, with the per-query guard"
numactl --interleave=all "$ORYX_BUILD/bench_guard" "${SIFT[@]}" --queries 10000 --batch 2500 --passes 6 --target 0.95 \
  --sample-rate 0.02 --threads "$(nproc)" --json "$OUT/sift1m-guard-perquery.json"
numactl --interleave=all "$ORYX_BUILD/bench_guard" "${GLOVE[@]}" --queries 10000 --batch 2500 --passes 6 --target 0.90 \
  --sample-rate 0.02 --threads "$(nproc)" --json "$OUT/glove-guard-perquery.json"
