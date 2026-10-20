# Toolchain for building and benchmarking Oryx on Georgia Tech's PACE clusters (ICE / Phoenix).
# Usage: source scripts/pace/env.sh
#
# Only the compiler comes from the module system. PACE builds CMake, Ninja and Python against a
# different GCC, so Lmod hides them once GCC 13 is loaded; CMake and Ninja come from pip instead,
# into a small virtualenv inside the checkout (made once, on first use).
module purge
module load gcc/13.3.0

# Repository root, whichever directory this is sourced from.
ORYX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ORYX_ROOT
export ORYX_BUILD="$ORYX_ROOT/build-pace"

ORYX_TOOLS="$ORYX_ROOT/.pace-tools"
if [ ! -x "$ORYX_TOOLS/bin/cmake" ] || [ ! -x "$ORYX_TOOLS/bin/ninja" ]; then
  /usr/bin/python3 -m venv "$ORYX_TOOLS"
  "$ORYX_TOOLS/bin/pip" install --quiet --upgrade pip
  "$ORYX_TOOLS/bin/pip" install --quiet "cmake>=3.28" ninja
fi
export PATH="$ORYX_TOOLS/bin:$PATH"
