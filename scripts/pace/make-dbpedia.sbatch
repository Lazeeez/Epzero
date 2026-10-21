#!/usr/bin/env bash
# Builds data/ann/dbpedia-openai-1000k-angular.hdf5 from the downloaded embeddings: exact top-100
# neighbours of 10,000 queries among 990,000 vectors (1536-d), on every core of one node.
# First, on the login node: .pace-bench/bin/python scripts/make_dbpedia.py download
#   sbatch scripts/pace/make-dbpedia.sbatch
#SBATCH --job-name=epzero-dbpedia
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=coc-cpu,ice-cpu
#SBATCH --time=01:00:00
#SBATCH --output=make-dbpedia-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/pace/env.sh
"$EPZERO_ROOT/.pace-bench/bin/python" scripts/make_dbpedia.py build
