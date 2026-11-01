#!/usr/bin/env bash
# Sweep of Epzero's speed experiments beside the shipped index and Glass, on one ann-benchmarks
# dataset, interleaved in one process on a dedicated Gold 6226 (bench/compete.py's rules):
#   epzero-sq8-nav<dims>  walks the graph on 8-bit codes of that many principal directions
#   -r<n>                 re-ranks n candidates with full vectors
#   -fill                 free link slots filled with the closest pruned candidates
#   -p<div>               per-query effort, patience = ef / div
# Each index is built on all cores first, then searched from one pinned thread.
# Result: pace-results/gold6226/<dataset>-speed.json (untracked; copy it back with scp)
#
# After compete.sbatch has run for the dataset (it reuses that saved SQ8 index):
#   sbatch --export=ALL,DATASET=gist-960-euclidean scripts/pace/speed.sbatch
#   sbatch --export=ALL,DATASET=dbpedia-openai-1000k-angular scripts/pace/speed.sbatch
#   sbatch --export=ALL,DATASET=deep-image-96-angular --time=16:00:00 scripts/pace/speed.sbatch
#   sbatch --export=ALL,DATASET=glove-100-angular scripts/pace/speed.sbatch
# LIBS=a,b,c overrides the sweep.
#SBATCH --job-name=epzero-speed
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=10:00:00
#SBATCH --output=speed-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
: "${DATASET:?pass the dataset: sbatch --export=ALL,DATASET=gist-960-euclidean ...}"
VENV="$EPZERO_ROOT/.pace-bench"
DATA="data/ann/$DATASET.hdf5"
CACHE=data/indexes/compete-gold6226
[ -x "$VENV/bin/python" ] || { echo "set up first: bash scripts/pace/bench-env.sh && bash scripts/pace/compete-env.sh"; exit 1; }
[ -f "$DATA" ] || { echo "download first: scripts/download_ann_dataset.sh $DATASET"; exit 1; }
ls "$CACHE/$DATASET-epzero-sq8-"*.idx > /dev/null || { echo "run compete.sbatch for $DATASET first"; exit 1; }

case "$DATASET" in
  gist-*)    SWEEP=nav96-r100,nav128-r100,nav128-r200,nav192-r100,nav256-r50,fill,nav128-r100-fill,nav192-r100-fill,nav128-r100-fill-p3 ;;
  dbpedia-*) SWEEP=nav256-r50,nav256-r100,nav384-r50,nav384-r100,nav512-r50,nav512-r100,fill,nav384-r100-fill ;;
  deep-*)    SWEEP=nav48-r50,nav60-r50,nav60-r100,fill,nav60-r50-fill ;;
  glove-*)   SWEEP=fill,fill-p3,nav60-r50,nav60-r100,nav80-r50 ;;
  *)         SWEEP=nav60-r50,fill ;;
esac
DEFAULT=epzero-sq8,glass
for variant in ${SWEEP//,/ }; do DEFAULT+=",epzero-sq8-$variant"; done
LIBS="${LIBS:-$DEFAULT}"
echo "libraries: $LIBS"

# Glass compiles with -march=native: build it on this node's CPU (one job at a time, see compete.sbatch).
exec 9>"$VENV/.install.lock"
flock 9
if [ -d "$VENV/src/pyglass/python" ]; then
  "$VENV/bin/pip" install --quiet --no-build-isolation --no-deps --force-reinstall "$VENV/src/pyglass/python"
fi
"$VENV/bin/pip" install --quiet --no-deps --force-reinstall "$EPZERO_ROOT/engine"
flock -u 9

OUT=pace-results/gold6226
mkdir -p "$OUT"
COMMON=(--data "$DATA" --cache "$CACHE" --M 16 --efc 200 --threads "$(nproc)" --libs "$LIBS")
echo "== build the new indexes on $(nproc) cores"
# Without Glass, which is rebuilt for its search run anyway (see compete.py).
"$VENV/bin/python" bench/compete.py "${COMMON[@]}" --libs "${LIBS//glass,/}" --build-only --out "$OUT/$DATASET-speed.json"
echo "== search, one thread, pinned to socket 0"
numactl --cpunodebind=0 --membind=0 \
  "$VENV/bin/python" bench/compete.py "${COMMON[@]}" --rounds 3 --out "$OUT/$DATASET-speed.json"
