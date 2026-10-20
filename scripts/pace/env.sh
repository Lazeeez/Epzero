# Toolchain for building and benchmarking Oryx on Georgia Tech's PACE clusters (ICE / Phoenix).
# Usage: source scripts/pace/env.sh
module purge
module load gcc/13.3.0 cmake/3.30.2 ninja/1.12.1 python/3.12.5

# Repository root, whichever directory this is sourced from.
ORYX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ORYX_ROOT
export ORYX_BUILD="$ORYX_ROOT/build-pace"
