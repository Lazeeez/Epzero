#!/usr/bin/env bash
# Round 7: does automatic configuration (epzero.build) match the hand-tuned settings, and hold its
# recall target on the real queries? Glass, epzero-auto and each dataset's best hand-tuned settings,
# on the Gold 6226 (Cascade Lake), the EPYC 7513 (Zen 3, 8 NUMA nodes: also tests the builds now that
# index memory is spread across nodes) and the Threadripper (Zen, AVX-512), queued as ONE chain:
# jobs share one Python environment and Glass is compiled per CPU, so CPUs must never overlap.
# Then throughput on the EPYC.
#
#   AFTER=<last queued job id> bash scripts/pace/round7.sh     (from the checkout, on a login node)
#
# The Threadripper sits in a GPU partition: it needs a GPU in the request and --mem=0, as the
# partition's default of 4 GB per core is more than the node has.
set -euo pipefail
last="${AFTER:-}"
queue() {  # dataset hw libs sbatch-args...
  local dataset=$1 hw=$2 libs=$3
  shift 3
  local after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" HW="$hw" TAG=round7 LIBS="$libs" LOCAL=1 \
    sbatch --parsable --export=ALL "${after[@]}" "$@" scripts/pace/speed.sbatch)
  echo "$last  speed $dataset on $hw"
}

S=epzero-sq8
machine() {  # hw sbatch-args...
  local hw=$1
  shift
  queue glove-100-angular "$hw" "glass,epzero-auto,$S-fast-refine" "$@"
  queue gist-960-euclidean "$hw" "glass,epzero-auto,$S-nav96-r400-z3-fast-refine-after" "$@"
  queue dbpedia-openai-1000k-angular "$hw" "glass,epzero-auto,$S-nav256-r300-z3-fast-refine-after" "$@"
  queue deep-image-96-angular "$hw" "glass,epzero-auto,$S-fast-refine" "$@"
}
machine gold6226 --time=10:00:00
machine epyc7513 -p ice-cpu -C epyc7513 --time=7:30:00
machine threadripper -p ice-gpu -C threadripper --gres=gpu:1 --mem=0 --time=6:00:00

# Throughput on many threads on the EPYC (moved from round 6, whose job never got a whole node): the
# first run with index memory spread across its eight NUMA nodes.
last=$(env DATASET=dbpedia-openai-1000k-angular HW=epyc7513 TAG=round7 LOCAL=1 \
  sbatch --parsable --export=ALL --dependency="afterany:$last" -p ice-cpu -C epyc7513 --time=7:30:00 \
  scripts/pace/throughput.sbatch)
echo "$last  throughput dbpedia-openai-1000k-angular on epyc7513"
