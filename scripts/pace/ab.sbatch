#!/usr/bin/env bash
# A/B of engine commits on the server: builds bench_recall at each commit (its own git worktree and
# build directory) and searches the saved SIFT1M index with each, interleaved over five rounds,
# pinned to one socket (as sift1m.sbatch). Prints the median QPS per commit and ef.
# Result: pace-results/gold6226/ab-sift1m.txt (copy it back with scp)
#
# After sift1m.sbatch (it saves the index). Default commits: before the visited bitset, the bitset,
# the sorted beam and descent prefetch, and HEAD:
#   sbatch scripts/pace/ab.sbatch
#   sbatch --export=ALL,COMMITS="abc1234 HEAD" scripts/pace/ab.sbatch
#SBATCH --job-name=epzero-ab
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=02:00:00
#SBATCH --output=ab-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
COMMITS="${COMMITS:-7da2db6 7abd47f 94fc4fc HEAD}"
INDEX=data/indexes/sift1m-M16-efc200.epzero
[ -f "$INDEX" ] || { echo "run sift1m.sbatch first (it saves $INDEX)"; exit 1; }
[ -f data/sift/sift_base.fvecs ] || { echo "download first: scripts/download_sift1m.sh"; exit 1; }

WORK="$EPZERO_ROOT/build-ab"
mkdir -p "$WORK"
declare -A BIN
for commit in $COMMITS; do
  sha=$(git -C "$EPZERO_ROOT" rev-parse --short "$commit")
  src="$WORK/src-$sha"
  [ -d "$src" ] || git -C "$EPZERO_ROOT" worktree add --detach --force "$src" "$sha" > /dev/null
  cmake -S "$src/engine" -B "$WORK/build-$sha" -G Ninja -DCMAKE_BUILD_TYPE=Release -DEPZERO_BUILD_NODE=OFF \
    -DEPZERO_BUILD_TESTS=OFF > /dev/null
  cmake --build "$WORK/build-$sha" --target bench_recall -j "$(nproc)" > /dev/null
  BIN[$commit]="$WORK/build-$sha/bench_recall"
  echo "built $commit ($sha)"
done

OUT=pace-results/gold6226
mkdir -p "$OUT"
RAW="$OUT/ab-sift1m.raw"
: > "$RAW"
SIFT=(--load "$INDEX" --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs
      --gt data/sift/sift_groundtruth.ivecs --queries 10000 --k 10 --ef 40,80,200 --search-threads 1 --no-baseline)
for round in 1 2 3 4 5; do
  for commit in $COMMITS; do
    numactl --cpunodebind=0 --membind=0 "${BIN[$commit]}" "${SIFT[@]}" 2>&1 |
      awk -v c="$commit" '/^ +[0-9]+ +0\./ {print c, $1, $2, $3}' >> "$RAW"
  done
  echo "round $round done"
done

python3 - "$RAW" > "$OUT/ab-sift1m.txt" <<'PY'
import collections, statistics, sys
qps = collections.defaultdict(list)
recall = {}
order = []
for line in open(sys.argv[1]):
    commit, ef, r, q = line.split()
    if commit not in order:
        order.append(commit)
    qps[(commit, int(ef))].append(float(q))
    recall[(commit, int(ef))] = r
base = order[0]
for ef in sorted({k[1] for k in qps}):
    cells = []
    for commit in order:
        m = statistics.median(qps[(commit, ef)])
        cells.append(f"{commit} {recall[(commit, ef)]} {m:8.0f} ({m / statistics.median(qps[(base, ef)]) - 1:+.1%})")
    print(f"ef {ef:4d} | " + " | ".join(cells))
PY
cat "$OUT/ab-sift1m.txt"
