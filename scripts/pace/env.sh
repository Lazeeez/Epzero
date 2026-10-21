# Toolchain for building and benchmarking Epzero on Georgia Tech's PACE clusters (ICE / Phoenix).
# Usage: source scripts/pace/env.sh
#
# Only the compiler comes from the module system. PACE builds CMake, Ninja and Python against a
# different GCC, so Lmod hides them once GCC 13 is loaded; CMake and Ninja come from pip instead,
# into a small virtualenv inside the checkout (made once, on first use).
module purge
module load gcc/13.3.0

# Repository root, whichever directory this is sourced from.
EPZERO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EPZERO_ROOT
export EPZERO_BUILD="$EPZERO_ROOT/build-pace"

EPZERO_TOOLS="$EPZERO_ROOT/.pace-tools"
if [ ! -x "$EPZERO_TOOLS/bin/cmake" ] || [ ! -x "$EPZERO_TOOLS/bin/ninja" ]; then
  /usr/bin/python3 -m venv "$EPZERO_TOOLS"
  "$EPZERO_TOOLS/bin/pip" install --quiet --upgrade pip
  "$EPZERO_TOOLS/bin/pip" install --quiet "cmake>=3.28" ninja
fi
export PATH="$EPZERO_TOOLS/bin:$PATH"

# The project was renamed from Oryx: give indexes saved under the old names their new names, so
# nothing is rebuilt (the engine still loads files written under the old name).
for old in "$EPZERO_ROOT"/data/indexes/*.oryx "$EPZERO_ROOT"/data/indexes/*/*-oryx-*; do
  [ -e "$old" ] || continue
  new="${old%.oryx}"
  [ "$new" != "$old" ] && new="$new.epzero"
  new="${new//-oryx-/-epzero-}"
  [ -e "$new" ] || mv "$old" "$new"
done
