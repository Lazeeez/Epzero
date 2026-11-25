#!/usr/bin/env bash
# The direction-count trial on a sample (bench/nav_trial.py), on a dedicated Gold 6226: which of the
# candidate counts a 100,000-vector sample ranks fastest at the recall target, to compare with the full
# datasets' ranking measured in rounds 12b and 13 on the same nodes.
#   sbatch --export=ALL,DATASET=arxiv-nomic-768-normalized,K=100,CANDIDATES=160,192,256,VIBE=1 scripts/pace/nav-trial.sbatch
# VIBE=1 fetches DATASET from VIBE's HuggingFace repository onto the node's own disk; otherwise data/ann/.
# Result: pace-results/gold6226/<dataset>-nav-trial-k<K>.json
#SBATCH --job-name=epzero-nav-trial
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=02:00:00
#SBATCH --output=nav-trial-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
: "${DATASET:?pass DATASET}" "${K:?pass K}" "${CANDIDATES:?pass CANDIDATES}"
VENV="$EPZERO_ROOT/.pace-bench"
[ -x "$VENV/bin/python" ] || { echo "set up first: bash scripts/pace/bench-env.sh"; exit 1; }
DATA="data/ann/$DATASET.hdf5"
if [ -n "${VIBE:-}" ]; then
  LOCAL_DIR="${TMPDIR:?VIBE=1 needs the node-local TMPDIR}/epzero-${SLURM_JOB_ID:?}"
  mkdir -p "$LOCAL_DIR"
  trap 'rm -rf "${LOCAL_DIR:?}"' EXIT
  DATA="$LOCAL_DIR/$DATASET.hdf5"
  curl -fsSL --retry 3 -o "$DATA.part" "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/$DATASET.hdf5"
  mv "$DATA.part" "$DATA"
fi
[ -f "$DATA" ] || { echo "download first: scripts/download_ann_dataset.sh $DATASET"; exit 1; }
exec 9>"$VENV/.install.lock"
flock 9
"$VENV/bin/pip" install --quiet --no-deps --force-reinstall "$EPZERO_ROOT/engine"
flock -u 9
OUT="pace-results/gold6226"
mkdir -p "$OUT"
echo "commit $(git -C "$EPZERO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
numactl --cpunodebind=0 --membind=0 \
  "$VENV/bin/python" bench/nav_trial.py --data "$DATA" --k "$K" --candidates "$CANDIDATES" \
  --out "$OUT/$DATASET-nav-trial-k$K.json"
