#!/usr/bin/env bash
# Round 8: modern embeddings and out-of-distribution queries, from VIBE (vector-index-bench). Glass
# against epzero-auto (calibrated on rows of the data) and, where queries come from elsewhere,
# epzero-auto-ood (calibrated on the dataset's sample queries), on the Threadripper and the Gold 6226,
# queued as ONE chain: jobs share one Python environment and Glass is compiled per CPU.
#
#   AFTER=<last queued job id> bash scripts/pace/round8.sh     (from the checkout, on a login node)
#
# Each job downloads its dataset (2-4 GB) to the node's own disk. The Threadripper needs a GPU in the
# request, --mem=0, and at most 4 hours (it is charged for all four GPUs); see round7.sh.
set -euo pipefail
last="${AFTER:-}"
queue() {  # dataset hw libs sbatch-args...
  local dataset=$1 hw=$2 libs=$3
  shift 3
  local after=()
  [ -z "$last" ] || after=(--dependency="afterany:$last")
  last=$(env DATASET="$dataset" HW="$hw" TAG=round8 LIBS="$libs" LOCAL=1 VIBE=1 \
    sbatch --parsable --export=ALL "${after[@]}" "$@" scripts/pace/speed.sbatch)
  echo "$last  speed $dataset on $hw"
}

# In distribution: text (arXiv abstracts, Nomic; news, mxbai) and images (ImageNet, CLIP).
IN="arxiv-nomic-768-normalized agnews-mxbai-1024-euclidean imagenet-clip-512-normalized"
# Out of distribution: text searching images (LAION and ImageNet with CLIP and ALIGN; Yandex).
OOD="laion-clip-512-normalized imagenet-align-640-normalized yandex-200-cosine"

machine() {  # hw sbatch-args...
  local hw=$1
  shift
  for dataset in $IN; do queue "$dataset" "$hw" "glass,epzero-auto" "$@"; done
  for dataset in $OOD; do queue "$dataset" "$hw" "glass,epzero-auto,epzero-auto-ood" "$@"; done
}
machine threadripper -p ice-gpu -C threadripper --gres=gpu:1 --mem=0 --time=4:00:00
machine gold6226 --time=10:00:00
