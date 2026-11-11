# Epzero in VIBE

[VIBE](https://github.com/vector-index-bench/vibe) benchmarks vector search on modern embeddings,
including out-of-distribution queries (text searching images, for example). `epzero/` is Epzero's entry
in its format: the adapter (`module.py`), the parameter grid (`config.yml`) and the container
(`image.def`, pinned to a released version).

The adapter builds each index with `epzero.build`, so Epzero configures itself for each dataset, and
VIBE sweeps the beam width. On out-of-distribution datasets VIBE passes sample queries (`fit_ood`), and
Epzero calibrates its search on them rather than on rows of the data. Builds use one thread, as VIBE
requires.

## Running it

```bash
git clone https://github.com/vector-index-bench/vibe.git && cd vibe
cp -r ../Epzero/vibe/epzero vibe/algorithms/epzero
./install.sh --algorithm epzero
python3 run.py --dataset laion-clip-512-normalized --algorithm epzero
./export_results.sh && ./plot.sh --dataset laion-clip-512-normalized
```

To submit, open a pull request adding `vibe/algorithms/epzero/` and a row in VIBE's algorithm table
with the pinned version. Before that, `scripts/pace/vibe.sbatch` in this repository runs the same
datasets against Glass with `bench/compete.py`, which is quicker to iterate with.
