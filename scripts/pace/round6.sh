#!/usr/bin/env bash
# One round on three CPUs with identical code, queued as ONE chain: the Gold 6226 (Cascade Lake, AVX-512),
# the EPYC 7513 (Zen 3, AVX2 only) and the Xeon 6972P (Granite Rapids). Jobs share one Python
# environment and Glass is compiled for the node that installs it, so jobs on different CPUs must never
# overlap. Indexes go to each node's local $TMPDIR, which frees scratch and is wiped when the job ends.
#
#   bash scripts/pace/round6.sh          (from the checkout, on a login node)
#
# Per dataset: Glass, Epzero's shipped default, and its best settings so far (prefetch now tuned over a
# record's every line, the faster build, the early re-rank stop). Then throughput on DBpedia, per CPU.
# PACE allows 30,720 CPU-minutes per job and these jobs take whole nodes, so the time limit scales with
# cores: 24 cores 10 h, 64 cores 7.5 h, 192 cores 2.5 h. The Xeon 6740P nodes carry GPUs, which hit the
# GPU-minute cap instead, so Granite Rapids runs on the 192-core node.
set -euo pipefail
last="${AFTER:-}"
queue() {  # script dataset hw libs sbatch-args...
  local script=$1 dataset=$2 hw=$3 libs=$4
  shift 4
  local after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" HW="$hw" TAG=round6 LIBS="$libs" LOCAL=1 \
    sbatch --parsable --export=ALL "${after[@]}" "$@" "scripts/pace/$script.sbatch")
  echo "$last  $script $dataset on $hw"
}

S=epzero-sq8
GIST="glass,$S,$S-nav192-r400-z3-fast-refine-after,$S-nav96-r400-z3-fast-refine-after"
DBP="glass,$S,$S-nav256-r300-z3-fast-refine-after,$S-nav384-r300-z3-fast-refine-after"
LOW="glass,$S,$S-fast-refine"

machine() {  # hw sbatch-args...
  local hw=$1
  shift
  queue speed glove-100-angular "$hw" "$LOW" "$@"
  queue speed gist-960-euclidean "$hw" "$GIST" "$@"
  queue speed dbpedia-openai-1000k-angular "$hw" "$DBP" "$@"
  queue speed deep-image-96-angular "$hw" "$LOW" "$@"
}
machine gold6226 --time=10:00:00
machine epyc7513 -p ice-cpu -C epyc7513 --time=7:30:00
machine xeon6972p -p ice-cpu -C cpu-gnr --time=2:30:00

queue throughput dbpedia-openai-1000k-angular gold6226 "" --time=10:00:00
queue throughput dbpedia-openai-1000k-angular epyc7513 "" -p ice-cpu -C epyc7513 --time=7:30:00
queue throughput dbpedia-openai-1000k-angular xeon6972p "" -p ice-cpu -C cpu-gnr --time=2:30:00
