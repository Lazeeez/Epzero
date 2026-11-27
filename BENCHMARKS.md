# Reproducing the benchmarks

This guide lets anyone check Epzero's published numbers on their own machine. The steps are
ordered from cheapest to most thorough, and every command below was run on a clean Linux machine
with Python 3.12. You need nothing but Python for steps 1–3: the engine and every competing
library install from PyPI.

| Step | What it checks | Time | Download |
| --- | --- | --- | --- |
| [1](#1-regenerate-the-published-tables) | The published tables follow from the raw results | seconds | none |
| [2](#2-a-two-minute-head-to-head) | Epzero against hnswlib, FAISS, USearch, VSAG and Glass on a small dataset | ~2 min | 0.2 GB |
| [3](#3-a-full-size-dataset) | The same comparison on the 1M-vector datasets in the README | 10–60 min each | 0.5–3.8 GB |
| [4](#4-the-c-engine-from-source) | The C++ engine builds, passes its tests, and its own benchmarks | ~10 min | none |
| [5](#5-dedicated-nodes) | The published runs, on a Slurm cluster | hours | as step 3 |

## Setup

```bash
git clone https://github.com/Lazeeez/Epzero.git && cd Epzero
python3 -m venv .venv && source .venv/bin/activate
pip install -r bench/requirements.txt
```

This installs Epzero 0.6.0 from PyPI and the libraries it is compared with, at the versions these
steps were checked with. Glass is published only as source. To include it:

```bash
git clone --depth 1 https://github.com/zilliztech/pyglass.git /tmp/pyglass
pip install --no-build-isolation /tmp/pyglass/python
```

Every library is optional. `bench/compete.py` skips any library that fails to import and prints
the reason.

## 1. Regenerate the published tables

The [scoreboard](docs/scoreboard.md) is generated from the raw JSON of every dedicated-node run,
which is committed under [`bench/results/pace/`](bench/results/pace): one folder per CPU. To
regenerate it and check that nothing was edited by hand:

```bash
python bench/scoreboard.py --results bench/results/pace --out /tmp/scoreboard.md
diff /tmp/scoreboard.md docs/scoreboard.md && echo "scoreboard reproduces exactly"
```

## 2. A two-minute head-to-head

[Fashion-MNIST](https://ann-benchmarks.com) has 60,000 vectors of 784 dimensions and is the
smallest standard ann-benchmarks dataset:

```bash
scripts/download_ann_dataset.sh fashion-mnist-784-euclidean
python bench/compete.py --data data/ann/fashion-mnist-784-euclidean.hdf5 \
    --out results/fashion-mnist.json --cache data/indexes --queries 2000 --rounds 1
```

The script builds every library's index with the same graph degree (`M=16`) and build beam
(`efConstruction=200`). It searches single-threaded, interleaves the libraries' timing rounds in one
process, and prints each library's queries per second at a recall target. A checked run on a
12-core laptop (one round, 2,000 queries, recall ≥ 0.95) gave:

```
epzero       build    12.0 s   QPS at recall>=0.95:     17299 (ef 20)
epzero-sq8   build     7.7 s   QPS at recall>=0.95:     34316 (ef 20)
hnswlib      build     4.8 s   QPS at recall>=0.95:     12994 (ef 20)
faiss-hnsw   build     5.8 s   QPS at recall>=0.95:     15897 (ef 20)
faiss-ivfpq  build    11.5 s   QPS at recall>=0.95:     13058 (nprobe 8)
usearch      build     9.1 s   QPS at recall>=0.95:      6376 (ef 20)
```

Expect different absolute numbers on other hardware; compare the ratios. A laptop is shared
with everything else running on it, so its numbers are noisier than the dedicated-node results.
Use `--rounds 3` (the default) for medians over interleaved rounds.

## 3. A full-size dataset

These are the datasets in the README's results table:

```bash
scripts/download_ann_dataset.sh glove-100-angular       # 1.2M x 100,  0.5 GB
scripts/download_ann_dataset.sh gist-960-euclidean      # 1M x 960,    3.6 GB
scripts/download_ann_dataset.sh deep-image-96-angular   # 10M x 96,    3.8 GB
pip install pyarrow                                     # DBpedia (1M x 1536 OpenAI embeddings) is built, not hosted:
python scripts/make_dbpedia.py download                 #   ~6 GB from Hugging Face
python scripts/make_dbpedia.py build                    #   computes the exact neighbours

python bench/compete.py --data data/ann/glove-100-angular.hdf5 --out results/glove.json --cache data/indexes
```

- **Memory:** at least 16 GB of RAM for the 1M datasets and 64 GB for DEEP10M.
- **Rebuilds:** indexes are cached under `--cache`, so a second run measures search only. Use
  `--rebuild epzero` to force a rebuild.
- **Glass:** it builds with `-march=native`, so build it on the machine that runs the benchmark.
- **More options:** run `python bench/compete.py --help` for every option, including
  `--libs` to choose libraries, `--M` and `--efc`.

The SIFT1M comparison in the README (Epzero, hnswlib and FAISS) has its own script:

```bash
scripts/download_sift1m.sh
python bench/compare_libs.py --data data/sift --cache data/indexes --out results/sift1m.json
```

## 4. The C++ engine from source

You need CMake 3.20 or later, Ninja, and GCC 13 or later or Clang 17 or later:

```bash
cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure      # the engine's test suite
build/bench_distance                            # distance kernels: scalar vs SIMD
build/bench_recall --synthetic --n 100000       # recall / QPS / latency sweep, no download
```

To build the Python package from this source instead of PyPI, run `pip install ./engine`. The
other C++ benchmarks are `bench_filter` (filtered search), `bench_churn` (search under concurrent
writes, deletes and re-inserts), `bench_guard` (recall under drift) and `bench_adaptive`.
[CONTRIBUTING.md](CONTRIBUTING.md) explains how changes are measured, and
[docs/performance.md](docs/performance.md) records every optimization, including those that did
not help and were removed.

## 5. Dedicated nodes

The published results come from exclusive nodes on Georgia Tech's PACE cluster, which run nothing
else while a benchmark runs. The CPUs are Intel Xeon Gold 6226, AMD EPYC 7513 and AMD Threadripper
PRO 9985WX. The Slurm job scripts and their setup are in [`scripts/pace/`](scripts/pace/README.md).
They run on any Slurm cluster once the partition and account lines are changed. Each job writes
the same JSON that step 1 reads.
