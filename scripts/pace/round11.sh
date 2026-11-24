#!/usr/bin/env bash
# Round 11: the re-rank depth of the walk on principal directions.
#  - On VIBE's AG News (k = 100) Epzero's recall stopped at 0.988 from ef 400 to 1000. epzero.build had chosen a
#    walk on 192 principal directions, re-ranking at most 300 candidates with full vectors (early stop at 3
#    standard deviations). On a 300K LongEval sample (448 directions) re-ranking at least ef candidates, with the
#    same early stop, lifted recall@100 from 0.9878 to 0.9908 at ef 1000 at equal speed, and changed nothing at
#    k = 10, where the early stop ends the re-rank first.
#  - Before that becomes the plan's rule: epzero-auto against the same index searched with a re-rank of
#    max(300, ef) (epzero-auto-deep) and Glass, on the four datasets where the plan walks on directions, at
#    k = 100 and k = 10, the whole width range, three timed rounds each.
# Queued as ONE chain after whatever is already queued (the jobs share one Python environment, and Glass is
# compiled per CPU):
#
#   AFTER=<last queued job id> bash scripts/pace/round11.sh     (from the checkout, on a login node)
set -euo pipefail
last="${AFTER:-}"
LIBS="epzero-auto,epzero-auto-deep,glass"
chain() {  # dataset k widths [VAR=value ...]
  local dataset=$1 k=$2 widths=$3
  shift 3
  local after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" K="$k" WIDTHS="$widths" TAG="round11-k$k" LIBS="$LIBS" LOCAL=1 "$@" \
    sbatch --parsable --export=ALL "${after[@]}" scripts/pace/speed.sbatch)
  echo "$last  speed $dataset k=$k on gold6226"
}

WIDE100=100,150,200,300,400,600,800,1000,1200,1600
WIDE10=10,20,40,80,120,200,300,400,600,800,1000,1200
for dataset in agnews-mxbai-1024-euclidean arxiv-nomic-768-normalized; do
  chain "$dataset" 100 "$WIDE100" VIBE=1
  chain "$dataset" 10 "$WIDE10" VIBE=1
done
for dataset in dbpedia-openai-1000k-angular gist-960-euclidean; do
  chain "$dataset" 100 "$WIDE100"
  chain "$dataset" 10 "$WIDE10"
done
