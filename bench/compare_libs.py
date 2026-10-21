"""Head-to-head benchmark of Epzero, hnswlib and FAISS (IndexHNSWFlat) on SIFT1M.

Fairness rules:
  * identical graph parameters (M, efConstruction) and build thread count;
  * queries run single-threaded as one batch per ef, so timing covers each library's C++
    search loop, not Python call overhead;
  * all indexes live in one process and timing rounds are interleaved with a rotating
    library order, so CPU frequency drift and scheduler noise hit everyone equally;
  * the reported QPS is the median over rounds.

Built indexes are cached (--cache) so reruns only measure search; --rebuild forces a build.

    python bench/compare_libs.py --data data/sift --cache data/indexes --out results/sift1m.json
"""

import argparse
import json
import os
import statistics
import time

import numpy as np


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    rows = raw.reshape(-1, raw[0] + 1)[:, 1:]
    return np.ascontiguousarray(rows.view(np.float32) if dtype == np.float32 else rows)


def recall_at_k(found, truth, k):
    hits = sum(len(set(f[:k].tolist()) & set(t[:k].tolist())) for f, t in zip(found, truth))
    return hits / (len(found) * k)


class Epzero:
    name = "epzero"

    def __init__(self, dim, args):
        import epzero as epzero
        self.mod, self.dim, self.args = epzero, dim, args

    def build(self, base):
        self.index = self.mod.Index(self.dim, "l2", len(base), self.args.M, self.args.efc)
        self.index.add(base, threads=self.args.threads)

    def save(self, path): self.index.save(path)
    def load(self, path): self.index = self.mod.Index.load(path)
    def search(self, q, k, ef): return self.index.search(q, k=k, ef=ef, threads=1)[0]


class Hnswlib:
    name = "hnswlib"

    def __init__(self, dim, args):
        import hnswlib
        self.mod, self.dim, self.args = hnswlib, dim, args

    def build(self, base):
        self.index = self.mod.Index(space="l2", dim=self.dim)
        self.index.init_index(max_elements=len(base), M=self.args.M, ef_construction=self.args.efc, random_seed=42)
        self.index.set_num_threads(self.args.threads)
        self.index.add_items(base, np.arange(len(base)))

    def save(self, path): self.index.save_index(path)

    def load(self, path):
        self.index = self.mod.Index(space="l2", dim=self.dim)
        self.index.load_index(path)

    def search(self, q, k, ef):
        self.index.set_ef(ef)
        return self.index.knn_query(q, k=k, num_threads=1)[0]


class Faiss:
    name = "faiss"

    def __init__(self, dim, args):
        import faiss
        self.mod, self.dim, self.args = faiss, dim, args

    def build(self, base):
        self.mod.omp_set_num_threads(self.args.threads)
        self.index = self.mod.IndexHNSWFlat(self.dim, self.args.M)
        self.index.hnsw.efConstruction = self.args.efc
        self.index.add(base)

    def save(self, path): self.mod.write_index(self.index, path)
    def load(self, path): self.index = self.mod.read_index(path)

    def search(self, q, k, ef):
        self.mod.omp_set_num_threads(1)
        self.index.hnsw.efSearch = ef
        return self.index.search(q, k)[1]


LIBRARIES = {cls.name: cls for cls in (Epzero, Hnswlib, Faiss)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="directory with sift_base/query/groundtruth files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", help="directory for built indexes")
    parser.add_argument("--rebuild", default="", help="comma-separated libraries to rebuild even if cached")
    parser.add_argument("--libs", default="epzero,hnswlib,faiss")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--efc", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--ef", default="10,20,40,80,120,200,400")
    args = parser.parse_args()
    efs = [int(x) for x in args.ef.split(",")]
    rebuild = set(filter(None, args.rebuild.split(",")))

    base = read_vecs(os.path.join(args.data, "sift_base.fvecs"), np.float32)
    queries = read_vecs(os.path.join(args.data, "sift_query.fvecs"), np.float32)
    truth = read_vecs(os.path.join(args.data, "sift_groundtruth.ivecs"), np.int32)
    n, dim = base.shape
    print(f"SIFT1M: {n} x {dim}, {len(queries)} queries, M={args.M}, efc={args.efc}, build threads={args.threads}")

    libs, builds = [], {}
    for name in args.libs.split(","):
        lib = LIBRARIES[name](dim, args)
        path = os.path.join(args.cache, f"{name}-M{args.M}-efc{args.efc}.idx") if args.cache else None
        meta = path + ".json" if path else None
        if path and os.path.exists(path) and name not in rebuild:
            lib.load(path)
            with open(meta) as f:
                builds[name] = json.load(f)["build_seconds"]
            print(f"{name:8s} loaded from cache (built in {builds[name]:.1f}s)")
        else:
            start = time.perf_counter()
            lib.build(base)
            builds[name] = time.perf_counter() - start
            print(f"{name:8s} built in {builds[name]:.1f}s")
            if path:
                os.makedirs(args.cache, exist_ok=True)
                lib.save(path)
                with open(meta, "w") as f:
                    json.dump({"build_seconds": builds[name], "threads": args.threads}, f)
        libs.append(lib)

    recall = {(lib.name, ef): recall_at_k(lib.search(queries, args.k, ef), truth, args.k) for lib in libs for ef in efs}
    samples = {key: [] for key in recall}
    for r in range(args.rounds):
        order = libs[r % len(libs):] + libs[:r % len(libs)]
        for ef in efs:
            for lib in order:
                start = time.perf_counter()
                lib.search(queries, args.k, ef)
                samples[(lib.name, ef)].append(len(queries) / (time.perf_counter() - start))
        print(f"round {r + 1}/{args.rounds} done")

    print(f"\n{'ef':>5} " + "".join(f"{lib.name + ' recall':>16}{lib.name + ' QPS':>14}" for lib in libs))
    for ef in efs:
        cells = "".join(f"{recall[(l.name, ef)]:16.4f}{statistics.median(samples[(l.name, ef)]):14.0f}" for l in libs)
        print(f"{ef:5d} {cells}")

    results = {
        lib.name: {
            "build_seconds": builds[lib.name],
            "sweep": [{"ef": ef, "recall": recall[(lib.name, ef)], "qps": statistics.median(samples[(lib.name, ef)]),
                       "qps_samples": samples[(lib.name, ef)]} for ef in efs],
        }
        for lib in libs
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"dataset": "sift1m", "n": n, "dim": dim, "queries": len(queries), "k": args.k, "M": args.M,
                   "ef_construction": args.efc, "build_threads": args.threads, "rounds": args.rounds,
                   "libraries": results}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
