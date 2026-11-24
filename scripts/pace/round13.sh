#!/usr/bin/env bash
# Round 13: a direction rule for k = 100, on datasets it was not fitted on.
#  - Round 12b (four datasets): at k = 100 more principal directions than the planner's (80% of the variance, at
#    most dim / 4) were faster everywhere, by 2-75% against the automatic build; the best counts held 0.88-0.92
#    of the variance. At k = 10 the planner's count was best or close on DBpedia and GIST1M, and more directions
#    cost GIST1M 3-13%, so a change would apply to k = 100 only. The rule recorded before reading DBpedia and
#    GIST1M (88%, dim / 3) failed its own test there (DBpedia's best was 512, not 384; GIST1M at k = 10 lost).
#  - Candidate, from round 12b: at k = 100, enough directions for 90% of the variance, at most dim / 3, rounded
#    up to 32 (`epzero-auto-nav90`: epzero.build with only nav_dim fixed to that count). Fitted on round 12b, so
#    it is measured here on four VIBE text datasets no round has used.
#  - Prediction (recorded before any run): at k = 100, epzero-auto-nav90 is faster than epzero-auto at recall
#    0.95, 0.98 and 0.99 on at least three of the four datasets, and at no target more than 3% slower on any.
#    k = 10 is run and reported, not judged (the rule would not apply there). If the prediction fails, the
#    planner keeps its rule.
# Queued as ONE chain after whatever is already queued:
#
#   AFTER=<last queued job id> bash scripts/pace/round13.sh     (from the checkout, on a login node)
set -euo pipefail
last="${AFTER:-}"
LIBS="glass,epzero-auto,epzero-auto-nav90"   # Glass first: speed.sbatch drops "glass," from its build-only step
chain() {  # dataset k widths [VAR=value ...]
  local dataset=$1 k=$2 widths=$3
  shift 3
  local after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" K="$k" WIDTHS="$widths" TAG="round13-k$k" LIBS="$LIBS" LOCAL=1 VIBE=1 "$@" \
    sbatch --parsable --export=ALL "${after[@]}" scripts/pace/speed.sbatch)
  echo "$last  speed $dataset k=$k on gold6226"
}

WIDE100=100,150,200,300,400,600,800,1000,1200,1600
WIDE10=10,20,40,80,120,200,300,400,600,800,1000,1200
# MS MARCO (Qwen, 1024-d) is 36 GB, too large for three indexes on one node; Yahoo (384-d) tests the rule at a
# small dimension instead, where dim / 3 caps it at 128.
for dataset in ccnews-nomic-768-normalized gooaq-distilroberta-768-normalized yahoo-minilm-384-normalized \
               simplewiki-openai-3072-normalized; do
  chain "$dataset" 100 "$WIDE100"
  chain "$dataset" 10 "$WIDE10"
done
