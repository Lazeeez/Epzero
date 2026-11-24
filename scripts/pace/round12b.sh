#!/usr/bin/env bash
# Round 12b: the control round 12 lacked, and two more datasets.
#  - Round 12: walks on 256 and 384 principal directions beat the automatic build (160 on arXiv, 192 on AG
#    News) by 15-60% against Glass, at k = 100 and k = 10. Those arms go through compete.py's variant path,
#    which differs from epzero.build in more than the directions (the 8-bit ranges are trained on 200,000
#    rows, prefetch is tuned without k and ef). Here each dataset also gets the variant path at the
#    automatic build's own direction count, so a gain can be put on the directions.
#  - The planner keeps enough directions for 80% of the variance, and at most dim / 4 (kNavVariance and
#    kNavMinSaving in engine/src/auto.cpp), chosen when the re-rank stopped at 300. DBpedia (1536-d, 288
#    chosen) and GIST1M (960-d, 96 chosen) test whether more directions pay beyond arXiv and AG News.
#  - Every variant: re-rank up to max(300, ef) with the early stop at 3, built on the codes, refined, free
#    slots filled, as the automatic build walks. Three timed rounds, k = 100 and k = 10.
#  - Prediction, recorded on 25 November 2026 after the arXiv and AG News k = 100 results and before any
#    DBpedia or GIST result was read. Variance held (100,000-row samples; DBpedia's 37,462-row sample):
#    arXiv 160/192/256/384 directions 0.80/0.84/0.90/0.96, AG News 192/256/384/512 0.81/0.88/0.95/0.98,
#    DBpedia 288/384/512/768 0.83/0.88/0.92/0.97, GIST1M 96/160/240/320 0.83/0.89/0.93/0.95. The winners
#    so far (256 on both) hold 0.88-0.90. Rule to test: enough directions for 88% of the variance, at most
#    dim / 3 (instead of 80% and dim / 4), which picks 160 on GIST1M and 384 on DBpedia. Predicted: at k = 100
#    those are faster than the automatic build at recall 0.98 and 0.99 and within 3% of the best arm
#    tested there; at k = 10 the counts the rule picks are at most 5% slower than the automatic build's at
#    any target on all four. If either fails, the rule is not adopted as it stands.
# Queued as ONE chain after whatever is already queued:
#
#   AFTER=<last queued job id> bash scripts/pace/round12b.sh     (from the checkout, on a login node)
set -euo pipefail
last="${AFTER:-}"
walk() { local out="" n; for n in "$@"; do out+=",epzero-sq8-nav$n-r300-z3-fast-refine-after"; done; echo "$out"; }
chain() {  # dataset k widths libs [VAR=value ...]
  local dataset=$1 k=$2 widths=$3 libs=$4
  shift 4
  local after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" K="$k" WIDTHS="$widths" TAG="round12b-k$k" LIBS="$libs" LOCAL=1 "$@" \
    sbatch --parsable --export=ALL "${after[@]}" scripts/pace/speed.sbatch)
  echo "$last  speed $dataset k=$k on gold6226"
}

WIDE100=100,150,200,300,400,600,800,1000,1200,1600
WIDE10=10,20,40,80,120,200,300,400,600,800,1000,1200
# Glass first: speed.sbatch drops "glass," from its build-only step. The first walk listed is the control.
for k in 100 10; do
  widths=$([ "$k" = 100 ] && echo "$WIDE100" || echo "$WIDE10")
  chain arxiv-nomic-768-normalized "$k" "$widths" "glass,epzero-auto$(walk 160 192 256)" VIBE=1
  chain agnews-mxbai-1024-euclidean "$k" "$widths" "glass,epzero-auto$(walk 192 256 512)" VIBE=1
  chain dbpedia-openai-1000k-angular "$k" "$widths" "glass,epzero-auto$(walk 288 384 512 768)"
  chain gist-960-euclidean "$k" "$widths" "glass,epzero-auto$(walk 96 160 240 320)"
done
