#!/usr/bin/env bash
# One index copy per socket: does it recover the throughput lost when 24 threads share one copy?
# On the 2-socket Gold 6226 node, 24 threads over one copy (memory interleaved across sockets)
# reached 14.0x one thread, while 12 threads on one socket's local memory reached 10.0x: about half
# of the 24-thread run's reads crossed to the other socket. Here each socket runs its own process
# with its own copy in its own memory, 12 threads each, at the same time, and the node's throughput
# is the sum. Uses the SIFT1M index saved by sift1m.sbatch; ef 80 and 200 as in scaling.sbatch.
# Results: pace-results/gold6226/sockets-run<n>-socket<s>.json (untracked; copy them back with scp)
#
#   sbatch scripts/pace/sockets.sbatch
#SBATCH --job-name=oryx-sockets
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=gold6226
#SBATCH --partition=coc-cpu,ice-cpu   # the partitions with Gold 6226 nodes
#SBATCH --time=00:30:00
#SBATCH --output=sockets-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
INDEX=data/indexes/sift1m-M16-efc200.oryx
[ -x "$ORYX_BUILD/bench_recall" ] || { echo "build first: scripts/pace/build.sh"; exit 1; }
[ -f "$INDEX" ] || { echo "run scripts/pace/sift1m.sbatch first (it saves the index)"; exit 1; }
cmake --build "$ORYX_BUILD" --target bench_recall -j "$(nproc)"

OUT=pace-results/gold6226
mkdir -p "$OUT"
# --repeat 48: each process searches for several seconds, so the two runs overlap almost entirely
# (each loads its copy in about a third of a second before it starts).
SIFT=(--base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs --gt data/sift/sift_groundtruth.ivecs
      --queries 10000 --k 10 --ef 80,200 --load "$INDEX" --no-baseline --search-threads 12 --repeat 48)

for run in 1 2 3; do
  echo "== run $run: two processes, one per socket"
  for socket in 0 1; do
    numactl --cpunodebind="$socket" --membind="$socket" "$ORYX_BUILD/bench_recall" "${SIFT[@]}" \
      --json "$OUT/sockets-run$run-socket$socket.json" > "$OUT/sockets-run$run-socket$socket.log" &
  done
  wait
  /usr/bin/python3 - "$OUT" "$run" <<'EOF'
import json, sys
out, run = sys.argv[1], sys.argv[2]
sockets = [json.load(open(f"{out}/sockets-run{run}-socket{s}.json"))["sweep"] for s in (0, 1)]
for a, b in zip(*sockets):
    print(f"   ef {a['ef']:4}: {a['qps']:9.0f} + {b['qps']:9.0f} = {a['qps'] + b['qps']:9.0f} QPS  (recall {a['recall']:.4f})")
EOF
done
