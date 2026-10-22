#!/usr/bin/env bash
# Epzero's candidate improvements beside Glass, on one ann-benchmarks dataset, interleaved in one
# process on a dedicated Gold 6226 (bench/compete.py's rules):
#   epzero-sq8           as shipped (on this Cascade Lake: the AVX2 integer kernel)
#   epzero-sq8-avx512    the AVX-512 integer kernel, now without a masked tail
#   epzero-sq8-adaptive  per-query effort (patience = ef / 3)
# Result: pace-results/gold6226/<dataset>-variants.json (untracked; copy it back with scp)
#
# After compete.sbatch has run for the dataset (it reuses the saved SQ8 index):
#   sbatch --export=ALL,DATASET=glove-100-angular scripts/pace/variants.sbatch
#   sbatch --export=ALL,DATASET=deep-image-96-angular --time=10:00:00 scripts/pace/variants.sbatch
#   sbatch --export=ALL,DATASET=dbpedia-openai-1000k-angular scripts/pace/variants.sbatch
#   sbatch --export=ALL,DATASET=gist-960-euclidean scripts/pace/variants.sbatch
#SBATCH --job-name=epzero-variants
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=05:00:00
#SBATCH --output=variants-%j.log
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
LIBS=epzero-sq8,epzero-sq8-avx512,epzero-sq8-adaptive,glass
COMMON=(--data "$DATA" --cache "$CACHE" --M 16 --efc 200 --threads "$(nproc)" --libs "$LIBS")
echo "== search, one thread, pinned to socket 0"
numactl --cpunodebind=0 --membind=0 \
  "$VENV/bin/python" bench/compete.py "${COMMON[@]}" --rounds 3 --out "$OUT/$DATASET-variants.json"
