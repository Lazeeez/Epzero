# Epzero in ann-benchmarks

[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) is the standard comparison of
approximate nearest-neighbour libraries. `epzero/` is Epzero's entry in its format: a Dockerfile that
installs `epzero` from PyPI, the parameter grid (`config.yml`: M 8/16/32 at full precision and
with 8-bit codes, each swept over the search beam width), and the wrapper (`module.py`).

## Submitting

Needs Docker and a published `epzero` release (see `.github/workflows/wheels.yml`).

```bash
git clone https://github.com/erikbern/ann-benchmarks.git && cd ann-benchmarks
pip install -r requirements.txt
cp -r ../Epzero/ann-benchmarks/epzero ann_benchmarks/algorithms/epzero
python install.py --algorithm epzero                              # builds the Docker image
python run.py --dataset glove-25-angular --algorithm epzero --runs 1
python run.py --dataset sift-128-euclidean --algorithm epzero --runs 1
python plot.py --dataset glove-25-angular                       # check the curve looks right
```

Then open a pull request adding `ann_benchmarks/algorithms/epzero/`, as other libraries have
(the CI there runs a small dataset against every changed algorithm).

## Before submitting, check locally

`bench/compete.py` in this repository runs the same libraries on the same datasets with the
same fairness rules (one process, interleaved rounds, one search thread), and is faster to iterate
with than the Docker setup.
