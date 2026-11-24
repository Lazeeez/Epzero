#!/usr/bin/env bash
# Round 12: how many principal directions to walk on at k = 100.
#  - Round 11 left one loss to Glass: arXiv (Nomic, 768-d) at k = 100, 0.74-0.91x across recall 0.90-0.99,
#    while at k = 10 Epzero is 1.06-1.15x. epzero.build walks on 160 of its 768 principal directions, chosen
#    for k = 10; at ef 100 and k = 100 the walk's recall is 0.735 against Glass's 0.935. Telling the 100th
#    neighbour from the 101st may need more directions than telling the 10th from the 11th, or no walk.
#  - Arms, each with M and efConstruction fixed to the run's: the automatic build (now re-ranking up to
#    max(300, ef)); no walk (8-bit codes on every dimension, built on the codes, refined: what epzero.build
#    does for image embeddings); walks on 256 and 384 directions, otherwise as the automatic build (re-rank
#    up to max(300, ef) with the early stop at 3, built on the codes, refined, free slots filled); Glass.
#  - arXiv and AG News (where the walk is 0.98x Glass at 0.95, k = 100), at k = 100 and k = 10, the whole
#    width range, three timed rounds each. If one setting wins at k = 100 and loses at k = 10, the plan's
#    choice of directions has to depend on k.
# Queued as ONE chain after whatever is already queued (the jobs share one Python environment, and Glass is
# compiled per CPU):
#
#   AFTER=<last queued job id> bash scripts/pace/round12.sh     (from the checkout, on a login node)
set -euo pipefail
last="${AFTER:-}"
# Glass first: speed.sbatch drops "glass," from its build-only step.
LIBS="glass,epzero-auto,epzero-sq8-fast-refine,epzero-sq8-nav256-r300-z3-fast-refine-after,epzero-sq8-nav384-r300-z3-fast-refine-after"
chain() {  # dataset k widths [VAR=value ...]
  local dataset=$1 k=$2 widths=$3
  shift 3
  local after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" K="$k" WIDTHS="$widths" TAG="round12-k$k" LIBS="$LIBS" LOCAL=1 "$@" \
    sbatch --parsable --export=ALL "${after[@]}" scripts/pace/speed.sbatch)
  echo "$last  speed $dataset k=$k on gold6226"
}

WIDE100=100,150,200,300,400,600,800,1000,1200,1600
WIDE10=10,20,40,80,120,200,300,400,600,800,1000,1200
for dataset in arxiv-nomic-768-normalized agnews-mxbai-1024-euclidean; do
  chain "$dataset" 100 "$WIDE100" VIBE=1
  chain "$dataset" 10 "$WIDE10" VIBE=1
done
