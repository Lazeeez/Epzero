#!/usr/bin/env bash
# Round 14: Glass at its best, 16-bit full vectors, 4-bit codes, and the direction-count trial, on all
# fourteen speed datasets at k = 10 and 100. Run from the sq4-codes branch. (Revised on 25 November 2026,
# before any job of this round was submitted: the first version, on main, had only P1-P4 and no 4-bit
# or trial jobs.)
#  - Every Epzero-vs-Glass ratio so far ran Glass built on full vectors and walked on SQ8U codes (`glass`). Its
#    README recommends building on SQ8U and walking on SQ4U, and that is where its VIBE results came from: there
#    (k = 100) 4-bit walking was 1.4-1.6x its 8-bit walk on AG News and 1.2-1.3x on ImageNet-CLIP up to recall
#    0.98, and never reached 0.90 on LAION. `glass-sq4` (SQ8U build, SQ4U walk, FP16 refine) runs beside `glass`.
#  - `epzero-auto-half` is epzero.build with the full vectors kept as 16-bit floats (half_vectors) and 8-bit codes,
#    opt-in today. Its last measurements were laptop runs at k = 10 (up to 17% slower on GIST1M, recall 0.003-0.004
#    lower on the DBpedia sample); a laptop pilot on 300,000 AG News vectors at k = 100 (25 November) found it 1-7%
#    faster at equal recall, in 45% less memory.
#  - 4-bit codes (this branch): `epzero-auto` now calibrates and times the 8-bit and the 4-bit walk on the same
#    graph and keeps 4 bits only where they are at least 3% faster at recall 0.95 (k = 10, the k it calibrates
#    at); `epzero-auto-8bit` and `epzero-auto-4bit` fix the choice. The AG News pilot (laptop, 300,000 vectors):
#    on the planner's 192-direction walk 4-bit was 3-6% slower at k = 100 and 18% slower at k = 10; on a walk over
#    every dimension it was 2-9% faster at equal beam width (about +5% at recall 0.99).
#  - The direction-count trial (bench/nav_trial.py, scripts/pace/nav-trial.sbatch): on a 100,000-vector sample,
#    the counts rounds 12b and 13 measured on the full datasets, each calibrated and timed; the trial's choice is
#    compared with the full-scale ranking. On 30,000 AG News vectors (laptop) it chose 256, the full-scale best.
#  - Predictions (recorded before any run):
#    P1. glass-sq4 is faster than glass at recall 0.95 on at least 9 of the 14 datasets at k = 10; its highest
#        recall on LAION is below 0.90 at both k; at k = 100 it is faster than epzero-auto-8bit at 0.95 on AG News
#        and ImageNet-CLIP (as in VIBE).
#    P2. epzero-auto-half's memory is 28-50% below epzero-auto-8bit's on every dataset.
#    P3. At equal ef, its recall is within 0.002 of epzero-auto-8bit's on every dataset except the two OpenAI ones
#        (DBpedia, Simple Wikipedia), where it is at most 0.005 lower.
#    P4. At recall 0.95-0.99 it is no more than 3% slower than epzero-auto-8bit at k = 100 on every dataset of 768
#        dimensions or more, and between 10% slower and 3% faster at k = 10.
#    P5. Where epzero.build walks on principal directions (AG News, arXiv, CC-News, DBpedia, GIST1M, Simple
#        Wikipedia), epzero-auto-4bit is slower than epzero-auto-8bit at recall 0.95 at both k.
#    P6. Where it walks on every dimension with at least 384 of them (GooAQ, Yahoo, ImageNet-CLIP, LAION,
#        ImageNet-ALIGN), epzero-auto-4bit is faster than epzero-auto-8bit at recall 0.98 at k = 100 on at least
#        three of the five.
#    P7. epzero-auto is at most 3% slower than the faster of epzero-auto-8bit and epzero-auto-4bit at recall 0.95
#        at k = 10 on at least 10 of the 12 datasets where it runs its trial (all but GloVe and DEEP, whose 96-100
#        codes are below its 128).
#    P8. On at least 12 of the 16 (dataset, k) trials, the sample's choice is the count that was fastest on the full
#        dataset at recall 0.95 (rounds 12b and 13, interpolated as in docs/performance.md), or within 5% of it.
#    Decisions: half_vectors may be chosen by the planner (where its calibration shows recall within 0.002) only if
#    P3 and P4 hold on at least 12 of the 14 datasets. The 4-bit trial is merged to main only if P7 holds and
#    epzero-auto-8bit is within 5% of round 12b/13's epzero-auto at recall 0.95 wherever both ran (no loss in the
#    8-bit path). The direction-count trial goes into epzero.build, to be measured on every dataset in a later
#    round, only if P8 holds. P1 decides nothing in the engine; it says which Glass the comparisons quote.
# Decision addendum (25 November 2026, after the jobs were submitted and before any round 14 result existed):
# Epzero is built for other developers, so a speed gain alone does not decide. The 4-bit merge and a planner
# choice of half_vectors also require, wherever both ran, a build time within 10% of epzero-auto-8bit's and no
# more memory than it; the 4-bit trial's choice is also recorded in the plan's reasons, as it depends on the CPU.
# Before the first job, on a login node: `bash scripts/pace/compete-env.sh` (pyglass pinned to VIBE's commit).
# Queued as ONE chain after whatever is already queued (the jobs share one Python environment): the kernel check,
# the trials, then the speed jobs, the datasets where Epzero lost first.
#
#   AFTER=<last queued job id> bash scripts/pace/round14.sh     (from the checkout, on a login node)
set -euo pipefail
last="${AFTER:-}"
after() { [ -z "$last" ] && echo "" || echo "--dependency=afterany:$last"; }
ALL="glass,glass-sq4,epzero-auto,epzero-auto-8bit,epzero-auto-4bit,epzero-auto-half"  # Glass first: speed.sbatch
SHORT="glass,glass-sq4,epzero-auto,epzero-auto-half"   # drops "glass," from its build-only step
chain() {  # libs dataset k widths [VAR=value ...]
  local libs=$1 dataset=$2 k=$3 widths=$4
  shift 4
  last=$(env DATASET="$dataset" K="$k" WIDTHS="$widths" TAG="round14-k$k" LIBS="$libs" LOCAL=1 "$@" \
    sbatch --parsable --export=ALL $(after) scripts/pace/speed.sbatch)
  echo "$last  speed $dataset k=$k on gold6226"
}
trial() {  # dataset k candidates [VAR=value ...]
  local dataset=$1 k=$2 candidates=$3
  shift 3
  last=$(env DATASET="$dataset" K="$k" CANDIDATES="$candidates" "$@" \
    sbatch --parsable --export=ALL $(after) scripts/pace/nav-trial.sbatch)
  echo "$last  nav trial $dataset k=$k ($candidates)"
}

last=$(sbatch --parsable $(after) scripts/pace/sq4-check.sbatch)
echo "$last  4-bit kernel check"

# The counts each dataset was measured at in full (round12b.sh, round13.sh).
for k in 100 10; do
  trial arxiv-nomic-768-normalized "$k" 160,192,256 VIBE=1
  trial agnews-mxbai-1024-euclidean "$k" 192,256,512 VIBE=1
  trial dbpedia-openai-1000k-angular "$k" 288,384,512,768
  trial gist-960-euclidean "$k" 96,160,240,320
  trial ccnews-nomic-768-normalized "$k" 192,256 VIBE=1
  trial gooaq-distilroberta-768-normalized "$k" 0,256 VIBE=1
  trial simplewiki-openai-3072-normalized "$k" 448,864 VIBE=1
  trial yahoo-minilm-384-normalized "$k" 0,128 VIBE=1
done

WIDE100=100,150,200,300,400,600,800,1000,1200,1600
WIDE10=10,20,40,80,120,200,300,400,600,800,1000,1200
for dataset in gooaq-distilroberta-768-normalized arxiv-nomic-768-normalized agnews-mxbai-1024-euclidean \
               imagenet-clip-512-normalized ccnews-nomic-768-normalized yahoo-minilm-384-normalized \
               yandex-200-cosine laion-clip-512-normalized imagenet-align-640-normalized \
               simplewiki-openai-3072-normalized; do
  chain "$ALL" "$dataset" 100 "$WIDE100" VIBE=1
  chain "$ALL" "$dataset" 10 "$WIDE10" VIBE=1
done
for dataset in dbpedia-openai-1000k-angular gist-960-euclidean; do
  chain "$ALL" "$dataset" 100 "$WIDE100"
  chain "$ALL" "$dataset" 10 "$WIDE10"
done
# GloVe and DEEP10M: 96-100 codes, below where the planner tries 4 bits, so no 4-bit arms. DEEP has ten million
# vectors, its indexes built twice over (Glass is rebuilt for its search run).
chain "$SHORT" glove-100-angular 100 "$WIDE100"
chain "$SHORT" glove-100-angular 10 "$WIDE10"
chain "$SHORT" deep-image-96-angular 100 "$WIDE100" SBATCH_TIMELIMIT=16:00:00
chain "$SHORT" deep-image-96-angular 10 "$WIDE10" SBATCH_TIMELIMIT=16:00:00
