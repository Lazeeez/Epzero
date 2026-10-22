#!/usr/bin/env bash
# Does AVX-512 cost Epzero more than it gives on the Gold 6226? Heavy 512-bit instructions can lower
# a Cascade Lake core's clock for everything around them. This searches the same saved SQ8 index
# with the AVX-512 kernels (beside Glass) and then with the AVX2 ones (EPZERO_SIMD=avx2: the level
# is chosen once per process, hence two runs), on one exclusive node, pinned to one socket.
# Results: pace-results/gold6226/<dataset>-simd-avx512.json and -simd-avx2.json (copy back with scp)
#
# After compete.sbatch has run for the dataset (it reuses the saved index):
#   sbatch --export=ALL,DATASET=glove-100-angular scripts/pace/simd.sbatch
#   sbatch --export=ALL,DATASET=deep-image-96-angular --time=06:00:00 scripts/pace/simd.sbatch
#SBATCH --job-name=epzero-simd
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=03:00:00
#SBATCH --output=simd-%j.log
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
COMMON=(--data "$DATA" --cache "$CACHE" --M 16 --efc 200 --threads "$(nproc)" --rounds 3)
echo "== AVX-512 kernels, beside Glass"
numactl --cpunodebind=0 --membind=0 \
  "$VENV/bin/python" bench/compete.py "${COMMON[@]}" --libs epzero-sq8,glass --out "$OUT/$DATASET-simd-avx512.json"
echo "== AVX2 kernels"
EPZERO_SIMD=avx2 numactl --cpunodebind=0 --membind=0 \
  "$VENV/bin/python" bench/compete.py "${COMMON[@]}" --libs epzero-sq8 --out "$OUT/$DATASET-simd-avx2.json"
