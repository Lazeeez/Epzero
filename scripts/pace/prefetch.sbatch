#!/usr/bin/env bash
# Does tune_prefetch's pick help at the beam widths the comparison measures? On the Gold 6226 it
# chose a shallow window for SQ8 (2 neighbours ahead), which was slower on a laptop. This searches
# the same saved SQ8 index tuned and with the default prefetching, beside Glass, interleaved in one
# process, pinned to one socket (as compete.sbatch does).
# Result: pace-results/gold6226/<dataset>-prefetch.json (untracked; copy it back with scp)
#
# After compete.sbatch has run for the dataset (it reuses the saved index):
#   sbatch --export=ALL,DATASET=glove-100-angular scripts/pace/prefetch.sbatch
#   sbatch --export=ALL,DATASET=deep-image-96-angular --time=06:00:00 scripts/pace/prefetch.sbatch
#SBATCH --job-name=epzero-prefetch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=03:00:00
#SBATCH --output=prefetch-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
: "${DATASET:?pass the dataset: sbatch --export=ALL,DATASET=glove-100-angular ...}"
VENV="$EPZERO_ROOT/.pace-bench"
DATA="data/ann/$DATASET.hdf5"
CACHE=data/indexes/compete-gold6226
[ -x "$VENV/bin/python" ] || { echo "set up first: bash scripts/pace/bench-env.sh && bash scripts/pace/compete-env.sh"; exit 1; }
[ -f "$DATA" ] || { echo "download first: scripts/download_ann_dataset.sh $DATASET"; exit 1; }
ls "$CACHE/$DATASET-epzero-sq8-"*.idx > /dev/null || { echo "run compete.sbatch for $DATASET first"; exit 1; }

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
numactl --cpunodebind=0 --membind=0 \
  "$VENV/bin/python" bench/compete.py --data "$DATA" --cache "$CACHE" --M 16 --efc 200 --threads "$(nproc)" \
  --libs epzero-sq8,epzero-sq8-default,glass --rounds 3 --out "$OUT/$DATASET-prefetch.json"
