#!/usr/bin/env bash
# Queues the next benchmark round as ONE chain, each job starting when the last ends. Jobs share one
# Python environment, and Glass is compiled for the CPU of the job that installs it, so jobs on
# different CPUs must never overlap. Each job also deletes its experiment indexes when it ends, which
# keeps scratch use to one sweep at a time.
#
#   bash scripts/pace/round5.sh          (from the checkout, on a login node)
#   AFTER=<job id> bash scripts/pace/round5.sh xeon     (resume: the Xeon and throughput jobs only)
#
# 1. Gold 6226: a grid of fixed prefetch settings on GloVe, and the early re-rank stop on GIST and DBpedia.
# 2. AMD EPYC 7513 and Xeon 6740P: Glass, shipped SQ8 and the best settings, on all four datasets. On the
#    Xeon, GloVe also runs with AVX2 kernels: does AVX-512 still lose on short vectors there?
# 3. Throughput on many threads, on DBpedia, on all three CPUs.
set -euo pipefail
last="${AFTER:-}"
FROM="${1:-all}"
queue() {  # script dataset hw tag libs sbatch-args...
  local script=$1 dataset=$2 hw=$3 tag=$4 libs=$5
  shift 5
  local after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" HW="$hw" TAG="$tag" LIBS="$libs" \
    sbatch --parsable --export=ALL "${after[@]}" "$@" "scripts/pace/$script.sbatch")
  echo "$last  $script $dataset on $hw ${tag:+($tag)}"
}

GIST=gist-960-euclidean
DBP=dbpedia-openai-1000k-angular
DEEP=deep-image-96-angular
GLOVE=glove-100-angular
S=epzero-sq8
GIST_A=$S-nav192-r100-fast-refine-after
GIST_B=$S-nav96-r100-fast-refine-after
DBP_A=$S-nav256-r100-fast-refine-after
DBP_B=$S-nav384-r50-fast-refine-after
LOW=$S-fast-refine

GRID=glass,$LOW
for setting in pfall.1 pfall.2 pf16.2 pf8.1 pf8.2 pf8.3 pf4.2 pf4.4 pf2.2; do GRID+=",$LOW-$setting"; done
if [ "$FROM" = all ]; then
  queue speed $GLOVE gold6226 round5 "$GRID"
  queue speed $GIST gold6226 round5 "glass,$GIST_A,$S-nav192-r400-z3-fast-refine-after,$GIST_B,$S-nav96-r400-z3-fast-refine-after"
  queue speed $DBP gold6226 round5 "glass,$DBP_A,$S-nav256-r300-z3-fast-refine-after,$DBP_B,$S-nav384-r300-z3-fast-refine-after"
fi

machine() {  # hw glove-extra sbatch-args...
  local hw=$1 extra=$2
  shift 2
  queue speed $GLOVE "$hw" round5 "glass,$S,$LOW$extra" "$@"
  queue speed $GIST "$hw" round5 "glass,$S,$GIST_A,$GIST_B" "$@"
  queue speed $DBP "$hw" round5 "glass,$S,$DBP_A,$DBP_B" "$@"
  queue speed $DEEP "$hw" round5 "glass,$S,$LOW" --time=16:00:00 "$@"
}
[ "$FROM" != all ] || machine epyc7513 "" -p ice-cpu -C epyc7513
machine xeon6740p ",$S-avx2,$LOW-avx2" -p ice-bw-cpu -C core96

queue throughput $DBP gold6226 "" ""
queue throughput $DBP epyc7513 "" "" -p ice-cpu -C epyc7513
queue throughput $DBP xeon6740p "" "" -p ice-bw-cpu -C core96
