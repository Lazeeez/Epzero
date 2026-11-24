#!/usr/bin/env bash
# Builds VIBE's container images for the methods in vibe-lib.sh, plus the image that exports results,
# on a dedicated Gold 6226 node. Run once after vibe-env.sh (and again after changing vibe/epzero):
#   sbatch scripts/pace/vibe-images.sbatch
#
# It first builds a one-line test image, so a cluster that doesn't allow building containers fails in
# minutes, not hours; where plain builds are refused it retries with --fakeroot and keeps that setting.
# Images go to .vibe/vibe/images in scratch (about 15-25 GB); the build itself runs on the node's disk.
#SBATCH --job-name=epzero-vibe-images
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --mem=0
#SBATCH --tmp=100G
#SBATCH --time=08:00:00
#SBATCH --output=vibe-images-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/vibe-lib.sh
[ -f "$VIBE_DIR/vibe/algorithms/epzero/image.def" ] || { echo "run scripts/pace/vibe-env.sh first"; exit 1; }
vibe_runtime
vibe_local
export APPTAINER_TMPDIR="$VIBE_LOCAL/tmp" APPTAINER_CACHEDIR="$VIBE_LOCAL/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "== can this node build images?"
printf 'Bootstrap: docker\nFrom: ubuntu:24.04\n\n%%post\n  true\n' > "$VIBE_LOCAL/probe.def"
if singularity build -F "$VIBE_LOCAL/probe.sif" "$VIBE_LOCAL/probe.def"; then
  echo "plain builds work"
elif VIBE_FAKEROOT=1 singularity build -F "$VIBE_LOCAL/probe.sif" "$VIBE_LOCAL/probe.def"; then
  echo "builds need --fakeroot: using it"
  export VIBE_FAKEROOT=1
else
  echo "This node can't build container images, plain or with --fakeroot. Ask PACE support how to build" >&2
  echo "Apptainer images from a definition file (or build them elsewhere and copy the .sif files in)." >&2
  exit 1
fi
printf '%s\n' "${VIBE_FAKEROOT:-}" > "$EPZERO_ROOT/.vibe/fakeroot"

cd "$VIBE_DIR"
mkdir -p images
for method in $VIBE_METHODS; do
  echo "== $method"
  ./install.sh --algorithm "$method" --build-dir "$VIBE_LOCAL" --image-dir images --force
done
echo "== plot (exports results)"
singularity build -F plot.sif plot.def

echo "== images"
ls -la images plot.sif
singularity exec images/epzero.sif python3 -c "import epzero; print('epzero', epzero.__version__)"
singularity exec images/epzero.sif cat /opt/epzero-commit 2>/dev/null || echo "epzero: the PyPI release"
echo "PASS: VIBE images built for: $VIBE_METHODS"
