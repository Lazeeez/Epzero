#!/usr/bin/env bash
# Round 15: 16-bit full vectors after the build fix, a narrower refine pass on walks over principal directions,
# and check_walk on COCO. Run from the nav-trial branch (it holds sq4-codes and both changes), after round 14.
# Recorded on 26 November 2026, before any job of this round and before round 14's speed results were read
# (only its trials, kernel check and GooAQ k = 100 were seen).
#  - 16-bit full vectors: round 14 ran a build that widened each 16-bit vector to floats one value at a time,
#    so its half-vector builds took 1.72x an 8-bit one on GooAQ. With F16C/NEON widening a laptop build took
#    1.01x (2.57x before). Speed and memory looked good on GooAQ (+3%, 37% less memory).
#  - Refine width: with a walk on directions the build refines at efConstruction (200), set by a GIST1M result
#    (width 100 saved 5% and cost 0.002-0.004 recall). On the AG News pilot (300,000, 192 directions) width 100
#    cut the refine pass from 12.0 to 5.5 s (27% of the build) with recall -0.0007 to +0.0027 at equal ef.
#    `epzero-auto-refine100` is the 8-bit build with refine_ef 100.
#  - check_walk: on COCO (text queries, image vectors) the rule's 128-direction walk never reached recall 0.95
#    (ef 4096, laptop); with check_walk the build kept no walk and reached it at ef 128.
#  - Predictions:
#    R1. epzero-auto-half's build time is at most 1.10x epzero-auto-8bit's on every dataset run; its memory is
#        28-50% lower; at equal ef its recall is within 0.002 (OpenAI datasets: at most 0.005 lower); at recall
#        0.95-0.99 it is at least 0.97x as fast at k = 100 (768+ dimensions) and 0.90-1.03x at k = 10.
#    R2. On the six datasets that walk on directions (AG News, arXiv, CC-News, DBpedia, GIST1M, Simple Wikipedia)
#        epzero-auto-refine100 builds 15-35% faster than epzero-auto-8bit and is 0.97-1.05x as fast at recall
#        0.95-0.99 at both k, on at least five of the six (GIST1M is the likeliest exception).
#    R3. On COCO epzero-auto-ood-checkwalk walks on no directions at both k, reaches recall 0.95 at both k,
#        and is at least 2x epzero-auto-ood's speed at recall 0.90, for at most 1.5x its build time.
#    Decisions: half_vectors becomes the planner's default (32-bit stays an option) only if R1 holds on every
#    dataset. The refine width becomes 100 on walks over directions only if R2 holds on all six, or on five with
#    the sixth at least 0.97x at every target. check_walk becomes the default whenever queries are given only
#    if R3 holds.
# Queued as ONE chain after whatever is already queued (the jobs share one Python environment):
#
#   AFTER=<last queued job id> bash scripts/pace/round15.sh     (from the checkout, on a login node)
set -euo pipefail
last="${AFTER:-}"
after() { [ -z "$last" ] && echo "" || echo "--dependency=afterany:$last"; }
chain() {  # libs dataset k widths [VAR=value ...]
  local libs=$1 dataset=$2 k=$3 widths=$4
  shift 4
  last=$(env DATASET="$dataset" K="$k" WIDTHS="$widths" TAG="round15-k$k" LIBS="$libs" LOCAL=1 "$@" \
    sbatch --parsable --export=ALL $(after) scripts/pace/speed.sbatch)
  echo "$last  speed $dataset k=$k on gold6226"
}

WIDE100=100,150,200,300,400,600,800,1000,1200,1600
WIDE10=10,20,40,80,120,200,300,400,600,800,1000,1200
WALKS="glass-sq4,epzero-auto-8bit,epzero-auto-half,epzero-auto-refine100"
PLAIN="glass-sq4,epzero-auto-8bit,epzero-auto-half"
for k in 100 10; do
  widths=$([ "$k" = 100 ] && echo "$WIDE100" || echo "$WIDE10")
  chain "$WALKS" agnews-mxbai-1024-euclidean "$k" "$widths" VIBE=1
  chain "$WALKS" arxiv-nomic-768-normalized "$k" "$widths" VIBE=1
  chain "$WALKS" ccnews-nomic-768-normalized "$k" "$widths" VIBE=1
  chain "$WALKS" simplewiki-openai-3072-normalized "$k" "$widths" VIBE=1
  chain "$WALKS" dbpedia-openai-1000k-angular "$k" "$widths"
  chain "$WALKS" gist-960-euclidean "$k" "$widths"
  chain "$PLAIN" gooaq-distilroberta-768-normalized "$k" "$widths" VIBE=1
  chain "$PLAIN" yahoo-minilm-384-normalized "$k" "$widths" VIBE=1
  chain "$PLAIN" glove-100-angular "$k" "$widths"
  chain "glass-sq4,epzero-auto-ood,epzero-auto-ood-checkwalk" coco-nomic-768-normalized "$k" "$widths" VIBE=1
done
