"""Builds dbpedia-openai-1000k-angular.hdf5 the way ann-benchmarks does: ann-benchmarks generates
this dataset rather than hosting it. 1M DBpedia entities embedded with OpenAI's
text-embedding-ada-002 (1536-d), from Hugging Face (KShivendu/dbpedia-entities-openai-1M);
10,000 of them become queries, and the exact top 100 neighbours (cosine) are computed here.

    python scripts/make_dbpedia.py download    # login node (needs internet; ~6 GB)
    python scripts/make_dbpedia.py build       # compute node (exact neighbours: a few minutes on 24 cores)

Output: data/ann/dbpedia-openai-1000k-angular.hdf5 (train, test, neighbors, distances).
"""

import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ann")
RAW = os.path.join(ROOT, "dbpedia-openai-raw")
OUT = os.path.join(ROOT, "dbpedia-openai-1000k-angular.hdf5")
QUERIES = 10_000
EXPECTED = 1_000_000  # rows in the full dataset


REPO = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M"
API = "https://huggingface.co/api/datasets/KShivendu/dbpedia-entities-openai-1M/tree/main/data"


def download():
    # Plain curl, one file at a time: the Hugging Face client's download backend needs enough
    # memory per file that PACE's login nodes kill it. curl streams to disk, and -C - resumes a
    # partial file, so rerunning picks up where a previous run stopped.
    import json
    import subprocess
    import urllib.request

    with urllib.request.urlopen(API) as resp:
        files = [f for f in json.load(resp) if f["path"].endswith(".parquet")]
    os.makedirs(os.path.join(RAW, "data"), exist_ok=True)
    for i, f in enumerate(files, 1):
        dest = os.path.join(RAW, f["path"])
        if os.path.exists(dest) and os.path.getsize(dest) == f["size"]:
            continue
        print(f"[{i}/{len(files)}] {f['path']} ({f['size'] / 1e6:.0f} MB)", flush=True)
        subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-s", "-o", dest, f"{REPO}/resolve/main/{f['path']}"], check=True)
        if os.path.getsize(dest) != f["size"]:
            sys.exit(f"{dest}: size mismatch; rerun to resume")
    print(f"downloaded: {len(files)} parquet files, complete")


def build():
    import h5py
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(RAW, "**", "*.parquet"), recursive=True))
    if not files:
        sys.exit("no parquet files: run `python scripts/make_dbpedia.py download` first")
    parts = []
    for f in files:
        # Straight from Arrow's flat buffer to NumPy: a Python list per vector would need tens of GB.
        column = pq.read_table(f, columns=["openai"]).column("openai").combine_chunks()
        flat = column.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        parts.append(flat.reshape(len(column), -1))
        print(f"  read {os.path.basename(f)}: {parts[-1].shape}", flush=True)
    data = np.concatenate(parts)
    if len(data) < EXPECTED:
        sys.exit(f"only {len(data):,} of {EXPECTED:,} vectors in {len(files)} files: the download is incomplete; "
                 "rerun `python scripts/make_dbpedia.py download`")
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(data))
    test, train = data[order[:QUERIES]], np.ascontiguousarray(data[order[QUERIES:]])
    print(f"train {train.shape}, test {test.shape}")

    start = time.time()
    neighbors = np.empty((QUERIES, 100), dtype=np.int32)
    distances = np.empty((QUERIES, 100), dtype=np.float32)
    for lo in range(0, QUERIES, 500):
        sims = test[lo:lo + 500] @ train.T  # BLAS on every core
        top = np.argpartition(-sims, 100, axis=1)[:, :100]
        top_sims = np.take_along_axis(sims, top, axis=1)
        ranked = np.argsort(-top_sims, axis=1)
        neighbors[lo:lo + 500] = np.take_along_axis(top, ranked, axis=1)
        distances[lo:lo + 500] = 1.0 - np.take_along_axis(top_sims, ranked, axis=1)
        print(f"  exact neighbours: {lo + 500}/{QUERIES} queries, {time.time() - start:.0f} s", flush=True)

    with h5py.File(OUT, "w") as f:
        f.attrs["type"] = "dense"
        f.attrs["distance"] = "angular"
        f.attrs["dimension"] = train.shape[1]
        f.attrs["point_type"] = "float"
        f["train"], f["test"], f["neighbors"], f["distances"] = train, test, neighbors, distances
    print("wrote", OUT)


if __name__ == "__main__":
    {"download": download, "build": build}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
