#!/usr/bin/env bash
# Round 10, from round 9's results on the Gold 6226 (kernels.sbatch):
#  - Yandex (200-d): with no AVX-512 anywhere (EPZERO_SIMD=avx2) Epzero was 4-10% faster than its default
#    and 1.08x Glass at recall 0.90. Before that becomes the default on this CPU, the same five kernel
#    settings on three more of VIBE's datasets, of 512-1024 dimensions.
#  - ImageNet-CLIP (512-d): no kernel setting helped (0.86-0.88x Glass at recall 0.90). At equal beam width
#    Epzero finds more neighbours than Glass but each step costs 17-20% more, and epzero.build chose no
#    walk on principal directions here (nav_dim 0). The speed sweep tries that walk at 128, 192 and 256
#    directions with deeper re-ranks, beside the automatic index and Glass.
# Queued as ONE chain after whatever is already queued (the jobs share one Python environment, and Glass
# is compiled per CPU):
#
#   AFTER=<last queued job id> bash scripts/pace/round10.sh     (from the checkout, on a login node)
set -euo pipefail
last="${AFTER:-}"
chain() {  # what dataset script [VAR=value ...]   (variables go through the environment: sbatch's
  local what=$1 dataset=$2 script=$3          # --export splits on commas, which LIBS contains)
  shift 3
  local after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" "$@" sbatch --parsable --export=ALL "${after[@]}" "$script")
  echo "$last  $what $dataset on gold6226"
}

for dataset in agnews-mxbai-1024-euclidean arxiv-nomic-768-normalized laion-clip-512-normalized; do
  chain kernels "$dataset" scripts/pace/kernels.sbatch
done

NAV="epzero-auto,glass,epzero-sq8-fast-refine"
for variant in nav128-r50 nav192-r50 nav256-r50 nav256-r100 nav128-r100; do NAV+=",epzero-sq8-$variant-fast-refine"; done
chain speed imagenet-clip-512-normalized scripts/pace/speed.sbatch LOCAL=1 VIBE=1 TAG=round10 LIBS="$NAV"
