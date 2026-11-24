#!/usr/bin/env bash
# Sets up VIBE (vector-index-bench) with Epzero's entry, for vibe-images.sbatch and vibe.sbatch.
# Run once on a login node, from the checkout in scratch (it has internet access):
#   bash scripts/pace/vibe-env.sh
#   EPZERO_REF=main bash scripts/pace/vibe-env.sh    (Epzero built from this checkout's ref, not PyPI)
#
# Checks out VIBE at the pinned commit into .vibe/vibe, copies Epzero's entry (vibe/epzero) into it,
# and reports whether this cluster has a container runtime. Rerun after changing vibe/epzero.
# With EPZERO_REF, the image builds Epzero from that git ref's engine folder (copied to .vibe/epzero-src)
# instead of installing the release from PyPI, and records the commit in the image (/opt/epzero-commit).
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

if [ -n "${EPZERO_REF:-}" ]; then
  commit="$(git -C "$EPZERO_ROOT" rev-parse --verify "$EPZERO_REF^{commit}")"
  source_dir="$EPZERO_ROOT/.vibe/epzero-src"
  rm -rf "$source_dir"
  mkdir -p "$source_dir"
  git -C "$EPZERO_ROOT" archive --format=tar "$commit" engine | tar -x -C "$source_dir"
  # The image build runs in a temporary folder, so the source is named by its absolute path.
  cat > "$VIBE_DIR/vibe/algorithms/epzero/image.def" <<EOF
Bootstrap: localimage
From: base.sif

%files
  $source_dir/engine /opt/epzero-engine

%post
  pip install --no-cache-dir /opt/epzero-engine
  echo "$commit" > /opt/epzero-commit
EOF
  echo "Epzero from $EPZERO_REF ($commit), not PyPI"
fi

for method in $VIBE_METHODS; do
  [ -f "$VIBE_DIR/vibe/algorithms/$method/image.def" ] || { echo "VIBE has no method '$method'" >&2; exit 1; }
done
echo "methods: $VIBE_METHODS"

# On PACE ICE, Apptainer is installed on the compute nodes (/usr/bin/apptainer) but not on the login
# nodes, so a miss here is expected; vibe-images.sbatch checks again on its node.
vibe_runtime || echo "(expected on a login node: the images job checks on a compute node)"
python3 -c "from ctypes.util import find_library as f; print('libnuma:', f('numa') or 'MISSING (VIBE needs it)')"
command -v numactl >/dev/null && echo "numactl: yes" || echo "numactl: no (VIBE falls back to taskset)"
