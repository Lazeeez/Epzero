# Running Oryx on PACE

Scripts for building and benchmarking on Georgia Tech's PACE clusters. Benchmarks run on exclusive
CPU nodes, so nothing else shares the machine; GPU nodes only prepare data (exact ground truth,
embeddings). Results are written as JSON and copied back into `bench/results/`.

## Setup (once)

Keep the checkout and datasets in scratch; the home quota is small.

```bash
cd ~/scratch
git clone https://github.com/Lazeeez/Oryx.git
cd Oryx
```

Then, on the login node (compute nodes may have no internet access), fetch the build tools once:

```bash
source scripts/pace/env.sh   # makes .pace-tools/ with CMake and Ninja from pip
```

## Build and check a node

```bash
srun -N1 --exclusive -C gold6226 -t 0:30:00 bash scripts/pace/build.sh 2>&1 | tee build-pace.log
```

This builds with GCC 13, runs the full test suite, measures the distance kernels at every SIMD
level the CPU supports (AVX-512 on the Xeon nodes) and does a quick recall run.

## SIFT1M

```bash
scripts/download_sift1m.sh              # login node, once (~160 MB)
sbatch scripts/pace/sift1m.sbatch       # build on 24 cores, then 3 pinned single-thread search runs
squeue -u $USER                         # wait for it; the log is sift1m-<jobid>.log
```

Results land in `pace-results/gold6226/`, which git ignores, so a later `git pull` never collides
with them. Copy that folder back with `scp -r`; it's committed to `bench/results/pace/` from there.

## Oryx vs hnswlib vs FAISS

```bash
bash scripts/pace/bench-env.sh          # login node, once: Python env with FAISS and Oryx's module
sbatch scripts/pace/compare.sbatch      # builds all three on 24 cores, then pinned search rounds
```

hnswlib is compiled inside the job, on the benchmark node, because it builds with `-march=native`
and the login node has a newer CPU than the benchmark nodes.

## Thread scaling

```bash
sbatch scripts/pace/scaling.sbatch      # after sift1m.sbatch; reuses its saved index (~15 min)
```

1 to 12 threads on one socket, then 12 to 24 over both sockets with interleaved memory, at ef=80
and ef=200, three runs per point.

## Filtered search and churn

```bash
sbatch scripts/pace/filter.sbatch       # every filter strategy at 0.1%-90% selectivity (~45 min)
sbatch scripts/pace/churn.sbatch        # search p99 under writes; delete/re-insert cycles (~15 min)
```

Both reuse SIFT1M; `filter.sbatch` also reuses the index saved by `sift1m.sbatch`.

## Competitive benchmark: more datasets, more libraries

Oryx (full precision and SQ8) against hnswlib, FAISS (HNSW, and IVF-PQ fast-scan with exact
re-rank), USearch, VSAG and Glass, on the ann-benchmarks datasets ([`bench/compete.py`](../../bench/compete.py)).
On the login node, once:

```bash
bash scripts/pace/bench-env.sh            # if not done already
bash scripts/pace/compete-env.sh          # USearch, VSAG, h5py; Glass's source
scripts/download_ann_dataset.sh gist-960-euclidean glove-100-angular dbpedia-openai-1000k-angular deep-image-96-angular
```

Then one job per dataset (DEEP10M is ten times larger, so it gets more time):

```bash
sbatch --export=ALL,DATASET=gist-960-euclidean scripts/pace/compete.sbatch
sbatch --export=ALL,DATASET=glove-100-angular scripts/pace/compete.sbatch
sbatch --export=ALL,DATASET=dbpedia-openai-1000k-angular scripts/pace/compete.sbatch
sbatch --export=ALL,DATASET=deep-image-96-angular --time=10:00:00 scripts/pace/compete.sbatch
```

Results: `pace-results/gold6226/<dataset>-compete.json`.

## Notes

- Nodes differ (`sinfo -p ice-cpu -o "%D %c %m %f"` lists the types). Benchmarks run on the
  `gold6226` nodes (2 × Xeon Gold 6226, 24 cores, AVX-512, clock fixed at 2.7 GHz), of which there
  are plenty, so results are comparable; `-C gold6226` picks one. The 64-core AMD EPYC and the
  192-core Granite Rapids nodes are for thread-scaling runs.
- Benchmark output records hardware facts only (CPU model, cores, memory), never user or account
  names.
