#!/usr/bin/env bash
# Round 9: the Gold 6226's low-recall losses (see kernels.sbatch), on the two datasets where round 8
# found them, queued as ONE chain after whatever is already queued (jobs share one Python environment,
# and Glass is compiled per CPU).
#
#   AFTER=<last queued job id> bash scripts/pace/round9.sh     (from the checkout, on a login node)
set -euo pipefail
last="${AFTER:-}"
for dataset in yandex-200-cosine imagenet-clip-512-normalized; do
  after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" sbatch --parsable --export=ALL "${after[@]}" scripts/pace/kernels.sbatch)
  echo "$last  kernels $dataset on gold6226"
done
