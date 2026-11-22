#!/usr/bin/env bash
# Sets up VIBE (vector-index-bench) with Epzero's entry, for vibe-images.sbatch and vibe.sbatch.
# Run once on a login node, from the checkout in scratch (it has internet access):
#   bash scripts/pace/vibe-env.sh
#
# Checks out VIBE at the pinned commit into .vibe/vibe, copies Epzero's entry (vibe/epzero) into it,
# and reports whether this cluster has a container runtime. Rerun after changing vibe/epzero.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/vibe-lib.sh"

mkdir -p "$(dirname "$VIBE_DIR")"
if [ ! -d "$VIBE_DIR/.git" ]; then
  git clone --quiet https://github.com/vector-index-bench/vibe.git "$VIBE_DIR"
fi
git -C "$VIBE_DIR" fetch --quiet origin
git -C "$VIBE_DIR" checkout --quiet --force "$VIBE_COMMIT"
rm -rf "$VIBE_DIR/vibe/algorithms/epzero"
cp -r "$EPZERO_ROOT/vibe/epzero" "$VIBE_DIR/vibe/algorithms/epzero"
echo "VIBE $(git -C "$VIBE_DIR" rev-parse --short HEAD) in $VIBE_DIR, with Epzero's entry"

for method in $VIBE_METHODS; do
  [ -f "$VIBE_DIR/vibe/algorithms/$method/image.def" ] || { echo "VIBE has no method '$method'" >&2; exit 1; }
done
echo "methods: $VIBE_METHODS"

# On PACE ICE, Apptainer is installed on the compute nodes (/usr/bin/apptainer) but not on the login
# nodes, so a miss here is expected; vibe-images.sbatch checks again on its node.
vibe_runtime || echo "(expected on a login node: the images job checks on a compute node)"
python3 -c "from ctypes.util import find_library as f; print('libnuma:', f('numa') or 'MISSING (VIBE needs it)')"
command -v numactl >/dev/null && echo "numactl: yes" || echo "numactl: no (VIBE falls back to taskset)"
