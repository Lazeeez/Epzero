#!/usr/bin/env bash
# Round 9: why Epzero trails Glass at low recall on the Gold 6226 alone (ImageNet-CLIP 0.87x and Yandex
# 0.99x at recall 0.90 in round 8, where it led on the Threadripper and the laptop). At equal beam width
# Epzero's recall is higher but its queries per second 15-23% lower, only on this CPU, so the suspect is
# what Cascade Lake runs slowly. One epzero-auto index is built, then searched in one pinned process per
# kernel setting (the SIMD level is chosen once per process), each beside a freshly built Glass, so every
# setting is read as a multiple of Glass measured under the same conditions:
#   default    what Epzero picks on this CPU
#   avx2       EPZERO_SIMD=avx2: no AVX-512 anywhere (float kernels and re-rank included)
#   float      EPZERO_SQ8_KERNEL=float: the float SQ8 kernel, without the integer path and its outliers
#   sq8avx512  EPZERO_SQ8_KERNEL=avx512: the AVX-512 integer kernel at any dimension
#   sq8avx2    EPZERO_SQ8_KERNEL=avx2: the AVX2 integer kernel at any dimension
# Result: pace-results/gold6226/<dataset>-kernels-<setting>.json (copy back with scp)
#
#   sbatch --export=ALL,DATASET=yandex-200-cosine scripts/pace/kernels.sbatch   (or via round9.sh)
#SBATCH --job-name=epzero-kernels
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=08:00:00
#SBATCH --output=kernels-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
: "${DATASET:?pass the dataset: sbatch --export=ALL,DATASET=yandex-200-cosine ...}"
VENV="$EPZERO_ROOT/.pace-bench"
[ -x "$VENV/bin/python" ] || { echo "set up first: bash scripts/pace/bench-env.sh && bash scripts/pace/compete-env.sh"; exit 1; }
[ -n "${TMPDIR:-}" ] && [ -w "$TMPDIR" ] || { echo "needs the node's local disk (TMPDIR)"; exit 1; }

# The dataset and index on the node's own disk, deleted when the job ends.
CACHE="${TMPDIR:?}/epzero-${SLURM_JOB_ID:?}"
trap 'rm -rf "${TMPDIR:?}/epzero-${SLURM_JOB_ID:?}"' EXIT
mkdir -p "$CACHE"
DATA="$CACHE/$DATASET.hdf5"
curl -fsSL --retry 3 -o "$DATA.part" "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/$DATASET.hdf5"
mv "$DATA.part" "$DATA"
echo "dataset: $DATASET ($(du -h "$DATA" | cut -f1)); cache: $CACHE"

# Glass compiles with -march=native: build it on this node's CPU (one job at a time, see compete.sbatch).
exec 9>"$VENV/.install.lock"
flock 9
if [ -d "$VENV/src/pyglass/python" ]; then
  rm -rf "${VENV:?}/src/pyglass/python/build"
  echo "glass flags: [$(glass_cflags)]"; CFLAGS="$(glass_cflags)" CXXFLAGS="$(glass_cflags)" "$VENV/bin/pip" install --quiet --no-cache-dir --no-build-isolation --no-deps --force-reinstall "$VENV/src/pyglass/python"
fi
"$VENV/bin/pip" install --quiet --no-deps --force-reinstall "$EPZERO_ROOT/engine"
flock -u 9

OUT=pace-results/gold6226
mkdir -p "$OUT"
COMMON=(--data "$DATA" --cache "$CACHE" --M 16 --efc 200 --threads "$(nproc)")
echo "== build epzero-auto on $(nproc) cores"
"$VENV/bin/python" bench/compete.py "${COMMON[@]}" --libs epzero-auto --build-only --out "$OUT/$DATASET-kernels-build.json"
for setting in default avx2 float sq8avx512 sq8avx2; do
  case "$setting" in
    default)   env=() ;;
    avx2)      env=(EPZERO_SIMD=avx2) ;;
    float)     env=(EPZERO_SQ8_KERNEL=float) ;;
    sq8avx512) env=(EPZERO_SQ8_KERNEL=avx512) ;;
    sq8avx2)   env=(EPZERO_SQ8_KERNEL=avx2) ;;
  esac
  echo "== $setting ${env[*]:-}, one thread, pinned to socket 0"
  env "${env[@]}" numactl --cpunodebind=0 --membind=0 \
    "$VENV/bin/python" bench/compete.py "${COMMON[@]}" --libs epzero-auto,glass --rounds 3 --out "$OUT/$DATASET-kernels-$setting.json"
done
