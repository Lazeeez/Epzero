#!/usr/bin/env bash
# Round 14: Glass at its best, and 16-bit full vectors, on all fourteen speed datasets at k = 10 and 100.
#  - Every Epzero-vs-Glass ratio so far ran Glass built on full vectors and walked on SQ8U codes (`glass`). Its
#    README recommends building on SQ8U and walking on SQ4U, and that is where its VIBE results came from: there
#    (k = 100) 4-bit walking was 1.4-1.6x its 8-bit walk on AG News and 1.2-1.3x on ImageNet-CLIP up to recall
#    0.98, and never reached 0.90 on LAION. `glass-sq4` (SQ8U build, SQ4U walk, FP16 refine) runs beside `glass`.
#  - `epzero-auto-half` is epzero.build with the full vectors kept as 16-bit floats (half_vectors), opt-in today.
#    Its last measurements were laptop runs at k = 10 (up to 17% slower on GIST1M, recall 0.003-0.004 lower on
#    the DBpedia sample). At k = 100 the re-rank reads up to max(300, ef) full vectors a query (1.2 MB on AG
#    News), where half the bytes may pay for the conversion.
#  - Predictions (recorded before any run):
#    P1. glass-sq4 is faster than glass at recall 0.95 on at least 9 of the 14 datasets at k = 10; its highest
#        recall on LAION is below 0.90 at both k; at k = 100 it is faster than epzero-auto at 0.95 on AG News and
#        ImageNet-CLIP (as in VIBE).
#    P2. epzero-auto-half's memory is 28-50% below epzero-auto's on every dataset.
#    P3. At equal ef, its recall is within 0.002 of epzero-auto's on every dataset except the two OpenAI ones
#        (DBpedia, Simple Wikipedia), where it is at most 0.005 lower.
#    P4. At recall 0.95-0.99 it is no more than 3% slower than epzero-auto at k = 100 on every dataset of 768
#        dimensions or more, and between 10% slower and 3% faster at k = 10.
#    Decision: the planner may turn half_vectors on (where its calibration shows recall within 0.002) only if
#    P3 and P4 hold on at least 12 of the 14 datasets; otherwise it stays opt-in. P1 decides nothing in the
#    engine; it says which Glass the comparisons quote from now on.
# Before the first job, on a login node: `bash scripts/pace/compete-env.sh` (pyglass pinned to VIBE's commit).
# Queued as ONE chain after whatever is already queued (the jobs share one Python environment), the datasets
# where Epzero lost first:
#
#   AFTER=<last queued job id> bash scripts/pace/round14.sh     (from the checkout, on a login node)
set -euo pipefail
last="${AFTER:-}"
LIBS="glass,glass-sq4,epzero-auto,epzero-auto-half"   # Glass first: speed.sbatch drops "glass," from its build-only step
chain() {  # dataset k widths [VAR=value ...]
  local dataset=$1 k=$2 widths=$3
  shift 3
  local after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" K="$k" WIDTHS="$widths" TAG="round14-k$k" LIBS="$LIBS" LOCAL=1 "$@" \
    sbatch --parsable --export=ALL "${after[@]}" scripts/pace/speed.sbatch)
  echo "$last  speed $dataset k=$k on gold6226"
}

WIDE100=100,150,200,300,400,600,800,1000,1200,1600
WIDE10=10,20,40,80,120,200,300,400,600,800,1000,1200
for dataset in gooaq-distilroberta-768-normalized arxiv-nomic-768-normalized agnews-mxbai-1024-euclidean \
               imagenet-clip-512-normalized ccnews-nomic-768-normalized yahoo-minilm-384-normalized \
               yandex-200-cosine laion-clip-512-normalized imagenet-align-640-normalized \
               simplewiki-openai-3072-normalized; do
  chain "$dataset" 100 "$WIDE100" VIBE=1
  chain "$dataset" 10 "$WIDE10" VIBE=1
done
for dataset in dbpedia-openai-1000k-angular gist-960-euclidean glove-100-angular; do
  chain "$dataset" 100 "$WIDE100"
  chain "$dataset" 10 "$WIDE10"
done
# DEEP10M: ten million vectors, four indexes built twice over (Glass is rebuilt for its search run).
chain deep-image-96-angular 100 "$WIDE100" SBATCH_TIMELIMIT=16:00:00
chain deep-image-96-angular 10 "$WIDE10" SBATCH_TIMELIMIT=16:00:00
