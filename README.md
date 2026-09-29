# Oryx

A vector database built from scratch in C++20. It implements HNSW
(Hierarchical Navigable Small World) approximate nearest-neighbor search, with
hand-written SIMD kernels, lock-free concurrent search and memory-system
tuning. On SIFT1M it serves **1.46× the queries per second of hnswlib and 1.40×
FAISS at the same recall**.

The same engine is being built for three targets: native (benchmarks and
servers), WebAssembly (a live, in-browser 3D demo) and a sharded, replicated
cluster.

**[Live demo →](https://lazeeez.github.io/Oryx/)** The engine compiled to
WebAssembly (SIMD128 + threads), running entirely in your browser:
- **[Landing page](https://lazeeez.github.io/Oryx/).** The engine indexes
  the photo collection behind the hero and answers live queries, then a
  scroll-driven explainer builds a small HNSW index (the real engine, in
  WebAssembly) and replays a search layer by layer. Click to search it yourself.
- **[Playground](https://lazeeez.github.io/Oryx/try).** 29,825 Open Images
  photos embedded with CLIP, indexed live on every core, and shown as a 3D
  stack of HNSW layers. Search by **description, uploaded photo or webcam**
  (CLIP text and image encoders run in the browser), step through the search
  hop by hop in a debugger, or race it against an exact scan. Every search is
  checked against the exact answer, e.g. *75 µs, 12/12 same results, 45×
  faster*.
  **Blend** a photo with words ("in the snow") and slide between them live;
  **tune and rebuild** the whole index with new `M` / `efConstruction` and
  compare measured recall and speed; orbit, zoom and tap through the 3D
  graph with a navigator HUD following each search.
- **[Play](https://lazeeez.github.io/Oryx/play).** *Out-navigate Oryx*: hop
  from photo to photo along the real base-layer links to reach a target,
  then see the route Oryx's search took through its express layers.
- **[Benchmarks](https://lazeeez.github.io/Oryx/benchmarks).** Interactive
  SIFT1M charts against hnswlib and FAISS, the optimization log, SQ8 results,
  and a lab that benchmarks the engine on your own device.

> **Status:** milestones 1–7 of 9 are done (engine, benchmarks, WebAssembly,
> dataset pipeline, 3D explorer, text search, quantization and durability).
> See the [roadmap](#roadmap).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/sift1m-recall-qps-dark.svg">
  <img alt="Recall versus QPS on SIFT1M: Oryx is above hnswlib and FAISS at every recall level" src="docs/img/sift1m-recall-qps-light.svg">
</picture>

## Results

**SIFT1M** (1M × 128-dim, 10k queries, official ground truth), 1 search thread,
median of 3 runs. Machine: Intel i5-13420H (AVX2), WSL2 Ubuntu 24.04, GCC 13,
`M=16`, `efConstruction=200`.

| ef  | recall@10 | QPS    | p50 latency | p99 latency | distances/query | vs. brute force |
|-----|-----------|--------|-------------|-------------|-----------------|-----------------|
| 10  | 0.7064    | 47,367 | 20 µs       | 55 µs       | 391             | 1,215×          |
| 40  | 0.9265    | 18,672 | 51 µs       | 113 µs      | 884             | 479×            |
| 80  | 0.9744    | 10,477 | 94 µs       | 208 µs      | 1,484           | 269×            |
| 200 | 0.9955    | 4,718  | 210 µs      | 413 µs      | 3,086           | 121×            |
| 400 | 0.9986    | 2,529  | 396 µs      | 697 µs      | 5,396           | 65×             |

- **Build:** 1M vectors in 42.9 s on 12 threads, vs 55.4 s for hnswlib and 68.4 s
  for FAISS. The index takes 764 MiB. It loads from a snapshot in 0.3 s.
- **Library comparison:** all three libraries were benchmarked in one process
  with interleaved rounds. Methodology and raw data are in
  [docs/performance.md](docs/performance.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/optimizations-dark.svg">
  <img alt="Search throughput after each optimization: 7,706 to 10,069 QPS, +31%" src="docs/img/optimizations-light.svg">
</picture>

Profiling showed search is memory-latency bound, so the optimizations target
the memory system. The [performance log](docs/performance.md) covers each
change, including the ones that were measured, didn't help and were removed
(graph reordering, early abandoning, LTO/PGO).

**Distance kernels.** Nanoseconds per call, L1-resident data:

| dim | L2 scalar | L2 AVX2 | speedup | IP scalar | IP AVX2 | speedup |
|-----|-----------|---------|---------|-----------|---------|---------|
| 128 | 36.7      | 5.2     | 7.0×    | 35.7      | 4.5     | 8.0×    |
| 384 | 140.6     | 17.5    | 8.1×    | 135.8     | 12.5    | 10.9×   |
| 960 | 412.7     | 48.7    | 8.5×    | 402.8     | 38.6    | 10.4×   |

The speedup is higher than the 8-lane width because the scalar loop is limited
by a serial dependency chain, while the SIMD kernels use two independent FMA
accumulators.

## Engine design

- **Memory layout.** Layer 0 is one contiguous block, aligned to and backed by
  2 MiB transparent huge pages. Each node's vector sits next to its neighbor
  list. Upper layers (about 1/M of the nodes) are stored separately, per node.
- **Search.** Greedy descent through the upper layers, then a beam search of
  width `ef` on layer 0. For each expanded node, unvisited neighbors are
  filtered first and all their vectors prefetched, so up to 32 cache misses
  overlap instead of running one after another.
- **Neighbor selection.** Uses the diversity heuristic from the HNSW paper
  (Algorithm 4), so links point in different directions and the graph stays
  navigable across clusters.
- **Visited set.** An 8-bit epoch-tagged array from a pool. It's 1 MiB for 1M
  nodes, which fits in L2, and it resets without clearing and allocates
  nothing per query.
- **SIMD dispatch.** Each ISA (AVX2, AVX-512, NEON) is compiled in its own
  translation unit and picked at runtime by CPU detection. `ORYX_SIMD=scalar`
  forces a level, e.g. for A/B benchmarks.
- **Concurrency.** Search is lock-free: neighbor-list slots are atomics written
  with release stores and read with acquire loads, so a reader that sees a node
  id also sees that node's vector. Writers are serialized per node, and a
  thread never holds two node locks at once, so lock-order deadlocks are
  impossible. ThreadSanitizer checks this with 8 writers and a concurrent
  reader.
- **Tracing.** Searches can optionally record every hop (layer, from, to,
  distance, accepted). This feeds the 3D visualizer and costs nothing when
  disabled.
- **8-bit scalar quantization (SQ8).** Nodes are stored as uint8 codes with
  per-dimension scales, scored against the float query by asymmetric distance
  (a single uint8×float dot product), then re-ranked exactly. On SIFT1M this
  gives **+48% QPS at identical recall**, or 58% less memory without the float
  vectors. See the [performance log](docs/performance.md#8-bit-scalar-quantization-sq8).
- **Deletes.** Tombstones that keep routing searches but are never returned or
  linked to again; `compacted()` rebuilds the index without them.
- **Durability.** `Database` adds a write-ahead log: CRC32C-framed records,
  fsync per write or group commit, and atomic checkpoints (the snapshot is
  written, fsynced and renamed before `CHECKPOINT` switches to it). Recovery
  replays newer records and truncates a torn tail. Tests cut the last record
  at every byte offset, corrupt records in the middle of the log, and
  `kill -9` a writer process at random moments; every acknowledged write
  survives.
- **Persistence.** A versioned binary snapshot format that doesn't depend on
  the in-memory layout. Every link is validated on load, so a corrupt file
  can't cause out-of-bounds reads.

## Python

```bash
pip install ./engine
```

```python
import numpy as np, oryx

index = oryx.Index(dim=128, metric="l2", max_elements=1_000_000, M=16, ef_construction=200)
index.add(vectors, threads=12)                  # labels default to 0..n-1
labels, distances = index.search(queries, k=10, ef=80, threads=1)
trace = index.trace(queries[0], k=10, ef=80)    # every hop, for visualization
index.save("sift.oryx"); index = oryx.Index.load("sift.oryx")

# 8-bit quantization: train on a sample, then add as usual
sq8 = oryx.Index(dim=128, quantization="sq8")
sq8.train(vectors[:100_000]); sq8.add(vectors, threads=12)

# Durable collection with a write-ahead log and crash recovery
db = oryx.Database.open("./vectors.db", dim=128, sync="always")
db.upsert(vectors[:1000], labels=ids)           # one log write + fsync per batch
db.remove(42)
print(db.recovery())                            # what the last open() replayed or truncated
```

Batch calls release the GIL and run on C++ threads.

## WebAssembly

The same C++ core compiles to WebAssembly with a SIMD128 kernel as one more
dispatch level. Two modules are shipped:
- a **threaded** build, which runs builds and exact search on every core via
  Web Workers sharing memory;
- a **single-threaded** fallback, for pages that aren't cross-origin isolated.

GitHub Pages can't send the isolation headers, so a service worker adds them.
The whole C++ test suite also runs as WebAssembly under Node in CI.

```bash
source emsdk/emsdk_env.sh
scripts/build_wasm.sh                 # -> web/public/wasm/{mt,st}
cd web && npm install && npm run dev  # http://localhost:5173/Oryx/
npx playwright test                   # full benchmark in headless Chromium
```

The page runs the engine in a Web Worker, so the UI never blocks while every
core builds the graph.

## Demo dataset

[`pipeline/build_dataset.py`](pipeline/build_dataset.py) builds the
explorer's data. It takes the Open Images V7 validation photos that are still
online (29,825 of 35,479), embeds them with CLIP ViT-B/32 on the GPU (92 s on
an RTX 4050), lays them out in 2-D with UMAP, and exports compact binaries:
- **int8 vectors** with per-dimension scales: 15 MB instead of 61 MB, with
  recall@10 of 0.994 against float vectors;
- a 16-bit layout;
- metadata with each photo's author, license and source link.

Photos are CC BY 2.0 and hotlinked from Flickr with attribution, and labels
are CC BY 4.0. (Unsplash Lite was evaluated and rejected: its terms forbid
publishing any portion of the data.)

```bash
pip install torch transformers umap-learn pillow
python pipeline/build_dataset.py --work data/openimages --out web/public/data/openimages
```

## Build and test

Linux or WSL2, CMake ≥ 3.20, GCC ≥ 13 or Clang ≥ 17.

```bash
cmake -S engine -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure

build/bench_distance
build/bench_recall --synthetic --n 100000

scripts/download_sift1m.sh
build/bench_recall --base data/sift/sift_base.fvecs --query data/sift/sift_query.fvecs \
                   --gt data/sift/sift_groundtruth.ivecs --queries 10000
python bench/compare_libs.py --data data/sift --cache data/indexes --out bench/results/sift1m-compare.json
```

Sanitizers: `-DORYX_SANITIZE=thread` or `-DORYX_SANITIZE=address,undefined`.
CI runs GCC and Clang on x86-64, GCC on AArch64 (NEON), both sanitizer
configurations, the Python tests, and the test suite as WebAssembly under Node.

## Roadmap

1. ✅ Engine core: HNSW, SIMD kernels, concurrency, snapshots, recall harness, CI
2. ✅ Head-to-head benchmarks against hnswlib and FAISS; profiling-driven optimization; Python bindings
3. ✅ WebAssembly build (SIMD128 + threads) and an in-browser benchmark lab on GitHub Pages
4. ✅ Offline dataset pipeline: CLIP image embeddings, UMAP layout, int8 export
5. ✅ 3D explorer: layered graph, animated search replay, exact-scan comparison
6. ✅ In-browser text-to-image search (CLIP text encoder on ONNX Runtime Web, self-hosted)
7. ✅ Deletes, write-ahead log, crash recovery; 8-bit scalar quantization
8. Cluster: Go router, sharded C++ nodes, replication, chaos testing (Oracle Cloud Ampere A1)
9. Filtered search, observability, write-ups
