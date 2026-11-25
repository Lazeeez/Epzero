#!/usr/bin/env bash
# 4-bit codes on a Gold 6226: the AVX2 and AVX-512 VNNI kernels must return the same searches
# (bench/sq4_check.py). The laptop the kernels were written on has no AVX-512, so this runs before any
# speed round measures 4-bit codes on these nodes. Ten minutes, one node:
#   sbatch scripts/pace/sq4-check.sbatch
#SBATCH --job-name=epzero-sq4-check
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=00:30:00
#SBATCH --output=sq4-check-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
VENV="$EPZERO_ROOT/.pace-bench"
[ -x "$VENV/bin/python" ] || { echo "set up first: bash scripts/pace/bench-env.sh"; exit 1; }
exec 9>"$VENV/.install.lock"
flock 9
"$VENV/bin/pip" install --quiet --no-deps --force-reinstall "$EPZERO_ROOT/engine"
flock -u 9
OUT="pace-results/gold6226"
mkdir -p "$OUT"
grep -o -m1 'avx512_vnni' /proc/cpuinfo || { echo "this node has no AVX-512 VNNI"; exit 1; }
for kernel in avx2 avx512; do
  EPZERO_SQ8_KERNEL=$kernel "$VENV/bin/python" bench/sq4_check.py "$OUT/sq4-check-$kernel.json"
done
"$VENV/bin/python" bench/sq4_check.py --compare "$OUT/sq4-check-avx2.json" "$OUT/sq4-check-avx512.json"
echo "PASS: the 4-bit kernels agree on this CPU"
