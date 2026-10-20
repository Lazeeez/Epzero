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

## Build and check a node

```bash
srun -N1 --exclusive -t 0:30:00 bash scripts/pace/build.sh 2>&1 | tee build-pace.log
```

This builds with GCC 13, runs the full test suite, measures the distance kernels at every SIMD
level the CPU supports (AVX-512 on the Xeon nodes) and does a quick recall run.

## Notes

- Nodes differ (`sinfo -p ice-cpu -o "%D %c %m %f"` lists the types). All benchmark runs target
  one node type, chosen with `--constraint`, so results are comparable.
- Benchmark output records hardware facts only (CPU model, cores, memory), never user or account
  names.
