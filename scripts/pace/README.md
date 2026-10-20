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

Results land in `bench/results/pace/gold6226/`; copy that folder back with `scp -r`.

## Notes

- Nodes differ (`sinfo -p ice-cpu -o "%D %c %m %f"` lists the types). Benchmarks run on the
  `gold6226` nodes (2 × Xeon Gold 6226, 24 cores, AVX-512, clock fixed at 2.7 GHz), of which there
  are plenty, so results are comparable; `-C gold6226` picks one. The 64-core AMD EPYC and the
  192-core Granite Rapids nodes are for thread-scaling runs.
- Benchmark output records hardware facts only (CPU model, cores, memory), never user or account
  names.
