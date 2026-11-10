#!/usr/bin/env bash
# Queries per second on many threads: Epzero's best settings for a dataset against Glass, at 1 thread,
# a quarter, half and all of a dedicated node's cores (bench/compete.py's rules otherwise). Memory is
# interleaved across the node, as a server sharing one index across sockets would run it.
# Result: pace-results/<hw>/<dataset>-throughput-t<threads>.json (untracked; copy it back with scp)
#
#   sbatch --export=ALL,DATASET=gist-960-euclidean scripts/pace/throughput.sbatch
#   sbatch -p ice-cpu -C epyc7513 --export=ALL,DATASET=gist-960-euclidean,HW=epyc7513 scripts/pace/throughput.sbatch
#   sbatch -p ice-bw-cpu -C core96 --export=ALL,DATASET=gist-960-euclidean,HW=xeon6740p scripts/pace/throughput.sbatch
#   sbatch -p ice-cpu -C cpu-gnr --export=ALL,DATASET=gist-960-euclidean,HW=xeon6972p scripts/pace/throughput.sbatch
# LIBS=a,b,c overrides the libraries; THREADS="1 8" the thread counts. The job deletes the experiment
# indexes it built when it ends, unless KEEP=1.
#SBATCH --job-name=epzero-throughput
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=08:00:00
#SBATCH --output=throughput-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
: "${DATASET:?pass the dataset: sbatch --export=ALL,DATASET=gist-960-euclidean ...}"
VENV="$EPZERO_ROOT/.pace-bench"
DATA="data/ann/$DATASET.hdf5"
HW="${HW:-gold6226}"
CACHE="data/indexes/compete-$HW"
if [ -n "${LOCAL:-}" ] && [ -n "${TMPDIR:-}" ] && [ -w "$TMPDIR" ]; then
  # LOCAL=1: indexes on the node's own disk, which keeps them off scratch. On PACE that is the node's
  # shared /tmp, not a per-job folder, so the job makes its own and deletes it when it ends.
  CACHE="${TMPDIR:?}/epzero-${SLURM_JOB_ID:?}"
  trap 'rm -rf "${TMPDIR:?}/epzero-${SLURM_JOB_ID:?}"' EXIT
  echo "index cache: $CACHE ($(df -h "$TMPDIR" | awk 'NR==2{print $4" free"}'))"
fi
[ -x "$VENV/bin/python" ] || { echo "set up first: bash scripts/pace/bench-env.sh && bash scripts/pace/compete-env.sh"; exit 1; }
[ -f "$DATA" ] || { echo "download first: scripts/download_ann_dataset.sh $DATASET"; exit 1; }

case "$DATASET" in
  gist-*)    BEST=epzero-sq8-nav192-r400-z3-fast-refine-after ;;
  dbpedia-*) BEST=epzero-sq8-nav384-r300-z3-fast-refine-after ;;
  *)         BEST=epzero-sq8-fast-refine ;;
esac
LIBS="${LIBS:-glass,$BEST}"
CORES=$(nproc)
THREADS="${THREADS:-1 $((CORES / 4)) $((CORES / 2)) $CORES}"
echo "libraries: $LIBS; threads: $THREADS; $(lscpu | grep 'Model name')"

# Glass compiles with -march=native: build it on this node's CPU (one job at a time, see compete.sbatch).
exec 9>"$VENV/.install.lock"
flock 9
if [ -d "$VENV/src/pyglass/python" ]; then
  # Compiled for this node: a build left from another CPU (AVX-512 on the Gold) crashed the AMD nodes with
  # an illegal instruction, so drop old objects and pip's cached wheel.
  rm -rf "${VENV:?}/src/pyglass/python/build"
  echo "glass flags: [$(glass_cflags)]"; CFLAGS="$(glass_cflags)" CXXFLAGS="$(glass_cflags)" "$VENV/bin/pip" install --quiet --no-cache-dir --no-build-isolation --no-deps --force-reinstall "$VENV/src/pyglass/python"
fi
"$VENV/bin/pip" install --quiet --no-deps --force-reinstall "$EPZERO_ROOT/engine"
flock -u 9

OUT="pace-results/$HW"
mkdir -p "$OUT" "$CACHE"
lscpu > "$OUT.lscpu.txt" 2>/dev/null || true
COMMON=(--data "$DATA" --cache "$CACHE" --M 16 --efc 200 --threads "$CORES" --libs "$LIBS")
cleanup() {
  [ -n "${KEEP:-}" ] || rm -f "${CACHE:?}/${DATASET:?}-epzero-sq8-"{nav,fast,fill,after,refine}*
}
if [ "$CACHE" = "data/indexes/compete-$HW" ]; then
  trap cleanup EXIT
fi

echo "== build on $CORES cores"
"$VENV/bin/python" bench/compete.py "${COMMON[@]}" --libs "${LIBS//glass,/}" --build-only --out "$OUT/$DATASET-throughput.json"
for threads in $THREADS; do
  echo "== search, $threads threads"
  numactl --interleave=all "$VENV/bin/python" bench/compete.py "${COMMON[@]}" --rounds 3 --search-threads "$threads" \
    --out "$OUT/$DATASET-throughput-t$threads.json"
done
