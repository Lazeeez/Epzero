"""Epzero against the fastest graph-based ANN libraries, on the ann-benchmarks datasets.

Libraries (each optional: one that fails to import is skipped and the run says so):
  epzero, epzero-sq8   this repository (full precision; 8-bit codes + exact re-rank)
  hnswlib          the reference HNSW implementation
  faiss-hnsw       FAISS IndexHNSWFlat
  faiss-ivfpq      FAISS tuned for speed: IVF + 4-bit PQ fast-scan + exact re-rank (IVF{4 sqrt n},PQ{d/2}x4fs,RFlat)
  usearch          USearch HNSW
  vsag             VSAG HGraph with SQ8 codes + exact re-rank (Ant Group)
  glass            Glass HNSW with SQ8U codes + refine (Zilliz)

Fairness rules (as in compare_libs.py):
  * graph indexes get the same degree (M, i.e. 2M links on the base layer) and build beam;
  * search is single-threaded and batched, so timing covers each library's C++ loop (except VSAG,
    whose Python binding searches one query per call: its numbers include ~1-2 us of Python call
    overhead per query, noted in the output);
  * one process, timing rounds interleaved with a rotating library order, median QPS reported;
  * angular (cosine) datasets are normalized once and searched with L2 by everyone, which ranks
    neighbours identically.

    python bench/compete.py --data data/ann/gist-960-euclidean.hdf5 --out results/gist1m.json --cache data/indexes
"""

import argparse
import json
import math
import os
import statistics
import time

import numpy as np


def load_dataset(path, max_queries):
    """An ann-benchmarks HDF5 file: train (base), test (queries), neighbors (exact top 100)."""
    import h5py

    with h5py.File(path, "r") as f:
        metric = f.attrs.get("distance", "euclidean")
        metric = metric.decode() if isinstance(metric, bytes) else str(metric)
        base = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:max_queries], dtype=np.float32)
        truth = np.asarray(f["neighbors"][:max_queries])
    if metric == "angular":
        base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
        queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    return base, queries, truth, metric


def recall_at_k(found, truth, k):
    hits = sum(len(set(np.asarray(f[:k]).tolist()) & set(t[:k].tolist())) for f, t in zip(found, truth))
    return hits / (len(found) * k)


class Library:
    """build(base) / save(path) / load(path, base) / search(queries, k, param) -> labels; params: the sweep."""

    params = [10, 20, 40, 80, 120, 200, 400, 800]
    param_name = "ef"
    saves = True
    note = ""

    def __init__(self, dim, args):
        self.dim, self.args = dim, args


class Epzero(Library):
    name = "epzero"
    quantization = "none"

    def __init__(self, dim, args):
        super().__init__(dim, args)
        import epzero

        self.mod = epzero

    def build(self, base):
        self.index = self.mod.Index(self.dim, "l2", len(base), self.args.M, self.args.efc, quantization=self.quantization)
        if self.quantization != "none":
            self.index.train(base[: min(len(base), 200_000)])
        self.index.add(base, threads=self.args.threads)
        self._tune(base)

    def _tune(self, base):
        # The best prefetch setting for this machine, measured on a sample of the data (as Glass's
        # optimize step does); not saved with the index, so it runs after loading too.
        if hasattr(self.index, "tune_prefetch"):
            sample = np.random.default_rng(0).choice(len(base), min(len(base), 1000), replace=False)
            print(f"{self.name:12s} prefetch {self.index.tune_prefetch(base[sample])}")

    def save(self, path):
        self.index.save(path)

    def load(self, path, base):
        self.index = self.mod.Index.load(path)
        self._tune(base)

    def search(self, q, k, ef):
        return self.index.search(q, k=k, ef=ef, threads=1)[0]


class EpzeroSq8(Epzero):
    name = "epzero-sq8"
    quantization = "sq8"


class EpzeroSq8Default(EpzeroSq8):
    """The same index with the default prefetching (every neighbour's first line up front) instead
    of tune_prefetch's pick: tells whether the tuner's choice helps at the measured beam widths.
    Not in the default comparison (scripts/pace/prefetch.sbatch runs it)."""

    name = "epzero-sq8-default"
    cache_name = "epzero-sq8"

    def _tune(self, base):
        print(f"{self.name:12s} prefetch {self.index.prefetch} (default, not tuned)")


class Hnswlib(Library):
    name = "hnswlib"

    def __init__(self, dim, args):
        super().__init__(dim, args)
        import hnswlib

        self.mod = hnswlib

    def build(self, base):
        self.index = self.mod.Index(space="l2", dim=self.dim)
        self.index.init_index(max_elements=len(base), M=self.args.M, ef_construction=self.args.efc, random_seed=42)
        self.index.set_num_threads(self.args.threads)
        self.index.add_items(base, np.arange(len(base)))

    def save(self, path):
        self.index.save_index(path)

    def load(self, path, base):
        self.index = self.mod.Index(space="l2", dim=self.dim)
        self.index.load_index(path)

    def search(self, q, k, ef):
        self.index.set_ef(max(ef, k))
        return self.index.knn_query(q, k=k, num_threads=1)[0]


class FaissHnsw(Library):
    name = "faiss-hnsw"

    def __init__(self, dim, args):
        super().__init__(dim, args)
        import faiss

        self.mod = faiss

    def build(self, base):
        self.mod.omp_set_num_threads(self.args.threads)
        self.index = self.mod.IndexHNSWFlat(self.dim, self.args.M)
        self.index.hnsw.efConstruction = self.args.efc
        self.index.add(base)

    def save(self, path):
        self.mod.write_index(self.index, path)

    def load(self, path, base):
        self.index = self.mod.read_index(path)

    def search(self, q, k, ef):
        self.mod.omp_set_num_threads(1)
        self.index.hnsw.efSearch = max(ef, k)
        return self.index.search(q, k)[1]


class FaissIvfPq(FaissHnsw):
    name = "faiss-ivfpq"
    params = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    param_name = "nprobe"

    def build(self, base):
        self.mod.omp_set_num_threads(self.args.threads)
        nlist = 1 << round(math.log2(4 * math.sqrt(len(base))))
        m = self.dim // 2 if self.dim % 2 == 0 else self.dim
        self.index = self.mod.index_factory(self.dim, f"IVF{nlist},PQ{m}x4fs,RFlat")
        sample = base[np.random.default_rng(0).choice(len(base), min(len(base), 50 * nlist), replace=False)]
        self.index.train(sample)
        self.index.add(base)

    def search(self, q, k, nprobe):
        self.mod.omp_set_num_threads(1)
        ivf = self.mod.extract_index_ivf(self.index)
        ivf.nprobe = nprobe
        self.index.k_factor = 4  # re-rank 4k PQ candidates exactly
        return self.index.search(q, k)[1]


class USearch(Library):
    name = "usearch"

    def __init__(self, dim, args):
        super().__init__(dim, args)
        from usearch.index import Index

        self.cls = Index

    def build(self, base):
        self.index = self.cls(ndim=self.dim, metric="l2sq", dtype="f32", connectivity=self.args.M, expansion_add=self.args.efc)
        self.index.add(np.arange(len(base)), base, threads=self.args.threads, log=False)

    def save(self, path):
        self.index.save(path)

    def load(self, path, base):
        self.index = self.cls(ndim=self.dim, metric="l2sq", dtype="f32", connectivity=self.args.M, expansion_add=self.args.efc)
        self.index.load(path)

    def search(self, q, k, ef):
        self.index.expansion_search = max(ef, k)
        return self.index.search(q, k, threads=1).keys


class Vsag(Library):
    name = "vsag"
    note = "searches one query per Python call (~1-2 us overhead each)"

    def __init__(self, dim, args):
        super().__init__(dim, args)
        import pyvsag

        self.mod = pyvsag

    def _params(self):
        return json.dumps({
            "dtype": "float32", "metric_type": "l2", "dim": self.dim,
            "index_param": {"base_quantization_type": "sq8", "use_reorder": True, "precise_quantization_type": "fp32",
                            "max_degree": 2 * self.args.M, "ef_construction": self.args.efc,
                            "build_thread_count": self.args.threads},
        })

    def build(self, base):
        self.index = self.mod.Index("hgraph", self._params())
        self.index.build(vectors=base.reshape(-1), ids=np.arange(len(base), dtype=np.int64), num_elements=len(base), dim=self.dim)

    def save(self, path):
        self.index.save(path)

    def load(self, path, base):
        self.index = self.mod.Index("hgraph", self._params())
        self.index.load(path)

    def search(self, q, k, ef):
        params = json.dumps({"hgraph": {"ef_search": max(ef, k)}})
        out = np.full((len(q), k), -1, dtype=np.int64)
        for i, row in enumerate(q):
            result = self.index.knn_search(vector=row, k=k, parameters=params)
            ids = result[0] if isinstance(result, tuple) else result
            out[i, : len(ids)] = ids
        return out


class Glass(Library):
    name = "glass"
    # Its saved graph loses recall when loaded back (measured: never reaches 0.95), so it is rebuilt
    # for the search run; the build time reported is the all-core one from the build run.
    saves = False

    def __init__(self, dim, args):
        super().__init__(dim, args)
        import glass

        self.mod = glass

    def build(self, base):
        self.mod.set_num_threads(self.args.threads)
        index = self.mod.Index("HNSW", "L2", R=2 * self.args.M, L=self.args.efc)
        self.graph = index.build(base)
        self._searcher(base)

    def _searcher(self, base):
        # SQ8U codes for the walk, refined with the full vectors (as VSAG and epzero-sq8 do).
        self.searcher = self.mod.Searcher(self.graph, base, "L2", "SQ8U", "FP32")
        self.searcher.optimize(1)



    def search(self, q, k, ef):
        self.searcher.set_ef(max(ef, k))
        ids = self.searcher.batch_search(q, k, num_threads=1)
        ids = ids[0] if isinstance(ids, tuple) else ids  # (ids, distances) in newer versions
        return np.asarray(ids).reshape(len(q), k)


COMPARED = (Epzero, EpzeroSq8, Hnswlib, FaissHnsw, FaissIvfPq, USearch, Vsag, Glass)
class EpzeroSq8Adaptive(EpzeroSq8):
    """SQ8 with per-query effort: a search stops once its best k have held for ef / 3 expansions
    (patience), so ef only caps the hard queries. Not in the default comparison."""

    name = "epzero-sq8-adaptive"
    cache_name = "epzero-sq8"

    def search(self, q, k, ef):
        return self.index.search(q, k=k, ef=ef, threads=1, patience=max(1, ef // 3))[0]


class EpzeroSq8Avx512(EpzeroSq8):
    """SQ8 forced onto the AVX-512 integer kernel, which Epzero skips on the first AVX-512 Xeons
    (EPZERO_SQ8_KERNEL is read when an index is made, so this copy alone gets it). Not in the
    default comparison."""

    name = "epzero-sq8-avx512"
    cache_name = "epzero-sq8"

    def load(self, path, base):
        os.environ["EPZERO_SQ8_KERNEL"] = "avx512"
        try:
            super().load(path, base)
        finally:
            del os.environ["EPZERO_SQ8_KERNEL"]


LIBRARIES = {cls.name: cls for cls in (*COMPARED, EpzeroSq8Default, EpzeroSq8Adaptive, EpzeroSq8Avx512)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="an ann-benchmarks .hdf5 file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", help="directory for built indexes")
    parser.add_argument("--libs", default=",".join(cls.name for cls in COMPARED))
    parser.add_argument("--rebuild", default="", help="comma-separated libraries to rebuild even if cached")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--efc", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-only", action="store_true", help="build and cache the indexes, then stop")
    args = parser.parse_args()
    dataset = os.path.splitext(os.path.basename(args.data))[0]
    base, queries, truth, metric = load_dataset(args.data, args.queries)
    n, dim = base.shape
    print(f"{dataset}: {n} x {dim} ({metric}), {len(queries)} queries, M={args.M}, efc={args.efc}, threads={args.threads}")

    libs, builds, skipped = [], {}, {}
    rebuild = set(filter(None, args.rebuild.split(",")))
    for name in args.libs.split(","):
        try:
            lib = LIBRARIES[name](dim, args)
        except Exception as error:  # not installed here
            skipped[name] = f"{type(error).__name__}: {error}"
            print(f"{name:12s} skipped: {skipped[name]}")
            continue
        cache_name = getattr(lib, "cache_name", name)  # variants share their library's saved index
        path = os.path.join(args.cache, f"{dataset}-{cache_name}-M{args.M}-efc{args.efc}.idx") if args.cache else None
        cached = path is not None and os.path.exists(path + ".json") and name not in rebuild
        if cached and lib.saves and os.path.exists(path):
            lib.load(path, base)
            with open(path + ".json") as f:
                builds[name] = json.load(f)["build_seconds"]
            print(f"{name:12s} loaded (built in {builds[name]:.1f} s)")
        else:
            start = time.perf_counter()
            lib.build(base)
            builds[name] = time.perf_counter() - start
            print(f"{name:12s} built in {builds[name]:.1f} s")
            if cached:  # rebuilt only to search: report the build run's time
                with open(path + ".json") as f:
                    builds[name] = json.load(f)["build_seconds"]
            elif path:
                os.makedirs(args.cache, exist_ok=True)
                if lib.saves:
                    lib.save(path)
                with open(path + ".json", "w") as f:
                    json.dump({"build_seconds": builds[name], "threads": args.threads}, f)
        libs.append(lib)
    if args.build_only:
        return

    recall = {(lib.name, p): recall_at_k(lib.search(queries, args.k, p), truth, args.k) for lib in libs for p in lib.params}
    samples = {key: [] for key in recall}
    for r in range(args.rounds):
        order = libs[r % len(libs):] + libs[: r % len(libs)]
        for i in range(max(len(lib.params) for lib in libs)):
            for lib in order:
                if i >= len(lib.params):
                    continue
                p = lib.params[i]
                start = time.perf_counter()
                lib.search(queries, args.k, p)
                samples[(lib.name, p)].append(len(queries) / (time.perf_counter() - start))
        print(f"round {r + 1}/{args.rounds} done")

    results = {}
    for lib in libs:
        sweep = [{lib.param_name: p, "recall": recall[(lib.name, p)], "qps": statistics.median(samples[(lib.name, p)]),
                  "qps_samples": samples[(lib.name, p)]} for p in lib.params]
        results[lib.name] = {"build_seconds": builds[lib.name], "param": lib.param_name, "sweep": sweep}
        if lib.note:
            results[lib.name]["note"] = lib.note
        best = max((s for s in sweep if s["recall"] >= 0.95), key=lambda s: s["qps"], default=None)
        print(f"{lib.name:12s} build {builds[lib.name]:7.1f} s   QPS at recall>=0.95: "
              f"{best['qps']:9.0f} ({lib.param_name} {best[lib.param_name]})" if best else f"{lib.name:12s} never reaches 0.95")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"dataset": dataset, "n": n, "dim": dim, "metric": metric, "queries": len(queries), "k": args.k,
                   "M": args.M, "ef_construction": args.efc, "build_threads": args.threads, "rounds": args.rounds,
                   "skipped": skipped, "libraries": results}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
