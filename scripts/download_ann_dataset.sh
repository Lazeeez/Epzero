#!/usr/bin/env bash
# Downloads ann-benchmarks datasets (HDF5: base vectors, queries, exact top-100 neighbours) into
# data/ann/. Run on a machine with internet access (a PACE login node).
#
#   scripts/download_ann_dataset.sh gist-960-euclidean glove-100-angular
#
# The competitive benchmark uses:
#   gist-960-euclidean            1M x 960,  3.6 GB   image descriptors (high dimension)
#   glove-100-angular             1.2M x 100, 0.5 GB  word embeddings (hard to search)
#   (dbpedia-openai-1000k-angular, 1M x 1536 OpenAI text embeddings, is not hosted: ann-benchmarks
#    generates it. Build it with scripts/make_dbpedia.py.)
#   deep-image-96-angular         10M x 96,  3.8 GB   image embeddings at 10M scale
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/ann"
mkdir -p "$DEST"
for name in "$@"; do
  if [[ $name == dbpedia-openai-* ]]; then
    echo "$name is generated, not downloaded: see scripts/make_dbpedia.py"
    continue
  fi
  file="$DEST/$name.hdf5"
  if [ -f "$file" ]; then
    echo "$name already present"
    continue
  fi
  curl -fL --retry 3 -o "$file.part" "https://ann-benchmarks.com/$name.hdf5"
  mv "$file.part" "$file"
  ls -lh "$file"
done
