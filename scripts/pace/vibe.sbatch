#!/usr/bin/env bash
# Runs VIBE (vector-index-bench) on one of its datasets with the methods built by vibe-images.sbatch,
# on a dedicated Gold 6226 node, as VIBE itself runs: one core per index, k = 100 by default, every
# method's own parameter grid. Then exports VIBE's summary tables.
#   sbatch --export=ALL,DATASET=agnews-mxbai-1024-euclidean scripts/pace/vibe.sbatch
#   sbatch --export=ALL,DATASET=imagenet-clip-512-normalized scripts/pace/vibe.sbatch
#   sbatch --export=ALL,DATASET=laion-clip-512-normalized scripts/pace/vibe.sbatch
# Options: COUNT=10 for k = 10; PARALLELISM=n indexes benchmarked at once (default 12 of the node's 24
# cores, which leaves memory bandwidth spare); METHODS="a b" to run fewer of the built methods.
#
# The dataset (2-5 GB) is downloaded to the node's own disk and VIBE runs there. Result:
# pace-results/vibe/<dataset>-<job>/ (untracked; copy it back with scp): VIBE's summary.parquet and
# <dataset>__detail.parquet, the run log and the node's description.
#SBATCH --job-name=epzero-vibe
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --mem=0
#SBATCH --tmp=60G
#SBATCH --time=16:00:00
#SBATCH --output=vibe-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/vibe-lib.sh
: "${DATASET:?pass the dataset: sbatch --export=ALL,DATASET=agnews-mxbai-1024-euclidean ...}"
COUNT="${COUNT:-100}"
PARALLELISM="${PARALLELISM:-12}"
vibe_runtime
vibe_local
VIBE_FAKEROOT="$(cat "$EPZERO_ROOT/.vibe/fakeroot" 2>/dev/null || true)"
export VIBE_FAKEROOT
for method in $VIBE_METHODS; do
  [ -f "$VIBE_DIR/images/$method.sif" ] || { echo "no image for $method: run scripts/pace/vibe-images.sbatch first"; exit 1; }
done
[ -f "$VIBE_DIR/plot.sif" ] || { echo "no plot.sif: run scripts/pace/vibe-images.sbatch first"; exit 1; }
python3 -c "from ctypes.util import find_library as f; assert f('numa'), 'libnuma missing'"

OUT="$EPZERO_ROOT/pace-results/vibe/$DATASET-$SLURM_JOB_ID"
mkdir -p "$OUT"
{
  date -u; hostname; lscpu; free -g
  singularity --version
  git -C "$VIBE_DIR" rev-parse HEAD
  singularity exec "$VIBE_DIR/images/epzero.sif" python3 -c "import epzero; print('epzero', epzero.__version__)"
  singularity exec "$VIBE_DIR/images/epzero.sif" cat /opt/epzero-commit 2>/dev/null || echo "epzero: the PyPI release"
  echo "methods: $VIBE_METHODS; count $COUNT; parallelism $PARALLELISM"
} > "$OUT/node.txt" 2>&1

# VIBE on the node's disk: its code and the chosen images (linked), the dataset, its results.
WORK="$VIBE_LOCAL/vibe"
mkdir -p "$WORK/images" "$WORK/data"
(cd "$VIBE_DIR" && tar --exclude=./images --exclude=./results --exclude=./data --exclude=./.git -cf - .) | (cd "$WORK" && tar -xf -)
for method in $VIBE_METHODS; do ln -s "$VIBE_DIR/images/$method.sif" "$WORK/images/$method.sif"; done
ln -sf "$VIBE_DIR/plot.sif" "$WORK/plot.sif"

echo "== download $DATASET"
curl -fsSL --retry 3 -o "$WORK/data/$DATASET.hdf5.part" \
  "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/$DATASET.hdf5"
mv "$WORK/data/$DATASET.hdf5.part" "$WORK/data/$DATASET.hdf5"
ls -la "$WORK/data"

cd "$WORK"
echo "== run"
python3 run.py --dataset "$DATASET" --count "$COUNT" --parallelism "$PARALLELISM" --image-dir images \
  2>&1 | tee "$OUT/run.log"
echo "== export"
./export_results.sh --parallelism 8 --output export --skip-pca-mahalanobis 2>&1 | tail -20 | tee "$OUT/export.log"
cp export/summary.parquet export/*__detail.parquet "$OUT/"
ls -la "$OUT"
echo "PASS: VIBE $DATASET: $OUT"
