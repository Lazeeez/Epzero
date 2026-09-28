# Oryx

A vector database built from scratch in C++20. It implements HNSW
(Hierarchical Navigable Small World) approximate nearest-neighbor search, with
hand-written SIMD kernels and a thread-safe index that allows concurrent
inserts and searches.

The same engine is being built for three targets: native (benchmarks and
servers), WebAssembly (a live, in-browser 3D demo) and a sharded, replicated
cluster.

> **Status:** milestone 1 of 9 is done (engine core). See the [roadmap](#roadmap).

## Results

**SIFT1M** (1M × 128-dim, 10k queries, official ground truth). Queries run on
1 thread; the build uses 12 threads.
Machine: Intel i5-13420H (AVX2, no AVX-512), WSL2 Ubuntu 24.04, GCC 13, `M=16`, `efConstruction=200`.

| ef  | recall@10 | QPS    | p50 latency | p99 latency | distances/query | vs. brute force |
|-----|-----------|--------|-------------|-------------|-----------------|-----------------|
| 10  | 0.7065    | 40,854 | 23 µs       | 49 µs       | 391             | 980×            |
| 40  | 0.9266    | 14,637 | 67 µs       | 115 µs      | 884             | 351×            |
| 80  | 0.9743    | 7,944  | 123 µs      | 251 µs      | 1,484           | 190×            |
| 200 | 0.9954    | 3,796  | 266 µs      | 482 µs      | 3,086           | 91×             |
| 400 | 0.9987    | 1,868  | 529 µs      | 1.04 ms     | 5,396           | 45×             |

Build: 1M vectors in 48.9 s (20.5k inserts/s), 764 MiB.

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

- **Memory layout.** Layer 0 is one contiguous 64-byte-aligned block. Each
  node's vector sits next to its neighbor list, so expanding a node touches
  adjacent cache lines. Upper layers (about 1/M of the nodes) are stored
  separately, per node.
- **Search.** Greedy descent through the upper layers, then a beam search of
  width `ef` on layer 0. The next neighbor's vector is prefetched while the
  current one is scored.
- **Neighbor selection.** Uses the diversity heuristic from the HNSW paper
  (Algorithm 4), so links point in different directions and the graph stays
  navigable across clusters.
- **Visited set.** An epoch-tagged array from a pool, so each query costs
  nothing to reset and allocates nothing.
- **SIMD dispatch.** Each ISA (AVX2, AVX-512, NEON) is compiled in its own
  translation unit and picked at runtime by CPU detection. `ORYX_SIMD=scalar`
  forces a level, e.g. for A/B benchmarks.
- **Concurrency.** Each node has its own lock covering only its neighbor lists,
  and a thread never holds two node locks at once, so lock-order deadlocks are
  impossible. The entry point and top level are packed into a single atomic
  word, so readers see a consistent pair without taking a lock. Checked by
  ThreadSanitizer with 8 writers and 1 concurrent reader.
- **Tracing.** Searches can optionally record every hop (layer, from, to,
  distance, accepted). This feeds the 3D visualizer and costs nothing when
  disabled.
- **Persistence.** A versioned binary snapshot format that doesn't depend on
  the in-memory layout. Every link is validated on load, so a corrupt file
  can't cause out-of-bounds reads.

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
```

Sanitizers: `-DORYX_SANITIZE=thread` or `-DORYX_SANITIZE=address,undefined`.
CI runs GCC and Clang on x86-64, GCC on AArch64 (NEON), and both sanitizer
configurations.

## Roadmap

1. ✅ Engine core: HNSW, SIMD kernels, concurrency, snapshots, recall harness, CI
2. Head-to-head benchmarks against hnswlib and FAISS; perf profiling
3. WebAssembly build (SIMD128) plus a minimal in-browser search page
4. Offline dataset pipeline: CLIP image embeddings, UMAP layout, int8 export
5. 3D visualizer: layered graph, search replay, recall/QPS explorer
6. In-browser text-to-image search (ONNX Runtime Web)
7. Deletes, write-ahead log, crash recovery; int8 scalar quantization
8. Cluster: Go router, sharded C++ nodes, replication, chaos testing (Oracle Cloud Ampere A1)
9. Filtered search, observability, write-ups
