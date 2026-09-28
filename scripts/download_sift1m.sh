#!/usr/bin/env bash
# Downloads SIFT1M (1M x 128-dim base, 10k queries, exact top-100 ground truth) from the
# INRIA TEXMEX corpus into data/sift/. About 160 MB compressed.
set -euo pipefail

DEST="$(dirname "$0")/../data"
mkdir -p "$DEST"
if [ -f "$DEST/sift/sift_base.fvecs" ]; then
  echo "SIFT1M already present in $DEST/sift"
  exit 0
fi
curl -fL --retry 3 -o "$DEST/sift.tar.gz" ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf "$DEST/sift.tar.gz" -C "$DEST"
rm "$DEST/sift.tar.gz"
ls -lh "$DEST/sift"
