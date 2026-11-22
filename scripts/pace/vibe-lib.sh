# Shared settings for running VIBE (vector-index-bench) on PACE: sourced by vibe-env.sh,
# vibe-images.sbatch and vibe.sbatch. Not run on its own.
#
# VIBE benchmarks each method inside its own container image. These scripts build the images for
# the methods below once, on a Gold 6226 node (some compile with -march=native, and VIBE's published
# results come from the same CPU family, Cascade Lake), then run one dataset per job.

EPZERO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EPZERO_ROOT
# VIBE, pinned. vibe-env.sh checks it out here and adds Epzero's entry from vibe/epzero.
VIBE_COMMIT=d2c94d480d7577fccaf20fb74366e4194a9a6e01
VIBE_DIR="$EPZERO_ROOT/.vibe/vibe"
# Epzero, the plain-HNSW reference, and the methods it has not been measured against:
#   glass   graph on SQ8/SQ4 codes          rabitq  SymphonyQG, HNSW-RaBitQ, IVF-RaBitQ
#   svs     Vamana, LVQ, LeanVec (Intel)    ngt     ONNG and NGT-QG
#   lorann  LoRANN                          scann   ScaNN
VIBE_METHODS="${METHODS:-epzero hnswlib glass rabitq svs ngt lorann scann}"

# A node-local working directory for this job (fast disk; removed with the job).
vibe_local() {
  local root="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
  VIBE_LOCAL="$root/vibe-${SLURM_JOB_ID:-$$}"
  mkdir -p "$VIBE_LOCAL"
  export VIBE_LOCAL
}

# VIBE calls `singularity`. PACE may provide Apptainer instead, or only through a module. This puts a
# `singularity` on PATH that runs whichever exists; with VIBE_FAKEROOT=1 it adds --fakeroot to builds.
vibe_runtime() {
  local runtime
  runtime="$(command -v apptainer || command -v singularity || true)"
  if [ -z "$runtime" ] && command -v module >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    runtime="$(command -v apptainer || command -v singularity || true)"
  fi
  if [ -z "$runtime" ]; then
    echo "No Apptainer or Singularity on this node (also tried 'module load apptainer'). VIBE needs one." >&2
    return 1
  fi
  local shim="$EPZERO_ROOT/.vibe/bin"
  mkdir -p "$shim"
  cat > "$shim/singularity" <<EOF
#!/usr/bin/env bash
if [ "\${1:-}" = build ] && [ -n "\${VIBE_FAKEROOT:-}" ]; then shift; exec "$runtime" build --fakeroot "\$@"; fi
exec "$runtime" "\$@"
EOF
  chmod +x "$shim/singularity"
  export PATH="$shim:$PATH"
  echo "container runtime: $runtime ($("$runtime" --version))"
}
