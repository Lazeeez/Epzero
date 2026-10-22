#!/usr/bin/env bash
# Where each library's search time goes: hardware counters (cycles, instructions, cache and TLB
# misses) per query for Epzero, Epzero SQ8, hnswlib and Glass, each at the smallest search beam that
# reaches the same recall, on GloVe-100, GIST1M and DBpedia OpenAI. Counting covers only the timed
# search loop (perf stat --control), searched from one thread pinned to socket 0, as in compete.sbatch.
# Needs the indexes cached by compete.sbatch for each dataset (Glass is rebuilt: it isn't saved).
# Result: pace-results/gold6226/profile.json (untracked; copy it back with scp)
#
#   sbatch scripts/pace/profile.sbatch
#SBATCH --job-name=epzero-profile
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=02:00:00
#SBATCH --output=profile-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
VENV="$EPZERO_ROOT/.pace-bench"
[ -x "$VENV/bin/python" ] || { echo "set up first: bash scripts/pace/bench-env.sh && bash scripts/pace/compete-env.sh"; exit 1; }
# perf isn't installed on the compute nodes by default; try the module system before giving up.
command -v perf >/dev/null || module load perf 2>/dev/null || module load linux-perf 2>/dev/null || true
command -v perf >/dev/null || { echo "perf is not available on this node (try: module spider perf)"; exit 1; }
echo "perf $(perf --version); perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"

# The same builds as compete.sbatch: native code for this node's CPU, Epzero from this checkout.
"$VENV/bin/pip" install --quiet --no-index --no-build-isolation --no-deps --force-reinstall "$VENV"/src/hnswlib-*.tar.gz
"$VENV/bin/pip" install --quiet --no-build-isolation --no-deps --force-reinstall "$VENV/src/pyglass/python"
"$VENV/bin/pip" install --quiet --no-deps --force-reinstall "$EPZERO_ROOT/engine"

# User-space counts only (allowed without root). The node has no hyperthreading, so these fit the
# core's counters together without multiplexing; percent_counted in the result says if they didn't.
EVENTS=cycles:u,instructions:u,branch-misses:u,L1-dcache-load-misses:u,LLC-load-misses:u,dTLB-load-misses:u,cycle_activity.stalls_l3_miss:u
CACHE=data/indexes/compete-gold6226
OUT=pace-results/gold6226
WORK="$OUT/profile"
mkdir -p "$WORK"
rm -f "$WORK"/*.csv "$WORK"/*.info.json

profile() {  # dataset target
  local dataset=$1 target=$2 data="data/ann/$1.hdf5"
  [ -f "$data" ] || { echo "skipping $dataset: $data is missing"; return; }
  "$VENV/bin/python" bench/profile_search.py pick --results "bench/results/pace/gold6226/$dataset-compete.json" --target "$target" |
  while read -r lib param; do
    echo "== $dataset $lib (beam $param)"
    local ctl="$WORK/ctl" ack="$WORK/ack"
    rm -f "$ctl" "$ack" && mkfifo "$ctl" "$ack"
    numactl --cpunodebind=0 --membind=0 \
      perf stat -x, -e "$EVENTS" --control "fifo:$ctl,$ack" -D -1 -o "$WORK/$dataset-$lib.csv" -- \
      "$VENV/bin/python" bench/profile_search.py run --data "$data" --cache "$CACHE" --lib "$lib" --param "$param" \
        --rounds 3 --threads "$(nproc)" --ctl "$ctl" --ack "$ack" --info "$WORK/$dataset-$lib.info.json" < /dev/null  # the loop reads its list from stdin
  done
}

profile glove-100-angular 0.90
profile gist-960-euclidean 0.95
profile dbpedia-openai-1000k-angular 0.95

"$VENV/bin/python" bench/profile_search.py report --dir "$WORK" --out "$OUT/profile.json"
