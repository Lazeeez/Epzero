# Performance engineering log

How Oryx's search got 31% faster, and which ideas didn't work. Each change was
kept only if an interleaved A/B test on SIFT1M showed a gain.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/optimizations-dark.svg">
  <img alt="Search throughput after each optimization: baseline 7,706 QPS; lock-free reads 7,879; batched prefetch 9,052; huge pages 10,189; 8-bit visited tags 10,069 (+31% overall)" src="img/optimizations-light.svg">
</picture>

## Methodology

- **Workload:** SIFT1M (1M × 128-dim base vectors, 10k queries, official exact
  ground truth). Graph built with M=16, efConstruction=200. Queries run on 1
  thread with k=10.
- **Same graph for every search change.** The index is built once, saved and
  reloaded by every candidate binary, so search changes are measured on an
  identical graph and recall is identical by construction (verified: 0.9744 at
  every step).
- **Interleaved A/B.** Candidate binaries run in alternating order for 5–7
  rounds, and the median is reported. The machine is a laptop CPU under WSL2
  (Intel i5-13420H, which has both P-cores and E-cores). Hyper-V hides which
  core a thread runs on, so pinning isn't possible. Interleaving makes frequency
  drift and scheduler noise affect every candidate equally. Differences under
  ~3% are treated as noise.
- **Library comparison.** Oryx, hnswlib and FAISS (`IndexHNSWFlat`) are loaded
  into one process and queried in interleaved rounds, each through its own C++
  batch loop, so Python call overhead isn't part of the timing.
  See [`bench/compare_libs.py`](../bench/compare_libs.py).

## Where the time goes

`perf record` on the search phase (hardware counters aren't available under
WSL2, so this is timer-based sampling):

| Hotspot | Share of search samples | Interpretation |
|---|---|---|
| `l2_avx2` | ~57% | Mostly stalls: each 128-dim vector is 512 bytes (8 cache lines) at a random address |
| `search_layer` | ~30% | Heap operations, visited-set checks, loop control |
| `pthread_mutex_lock/unlock` | ~12% | Readers locking each node's neighbor list |

At about 63 ns per distance evaluation, search is **bound by memory latency**,
not by computation. That points to changes that improve memory access, not
instruction-level changes.

## Kept

| Change | Result | Why it works |
|---|---|---|
| **Batched prefetch.** Filter a node's unvisited neighbors first, prefetch all their vectors, then score them | **+15%** | Up to 32 cache misses are in flight at once instead of one per distance call. Prefetching only the first line is enough: the hardware stream prefetcher follows once the kernel starts reading (64, 128, 256 and 512 bytes all measured the same; no prefetch at all was 14% slower) |
| **2 MiB transparent huge pages** for the layer-0 block (`madvise(MADV_HUGEPAGE)`, 2 MiB aligned) | **+13%**, p99 −16% | 764 MB spans ~190k 4 KiB pages, far more than the TLB covers, so almost every node visit was also a page walk. With huge pages it's ~380 pages. Verified via `AnonHugePages` in `/proc/<pid>/smaps_rollup` (672 MB huge-page backed) |
| **Lock-free reads.** Neighbor-list slots are `std::atomic_ref` with release stores and acquire loads | +2% (noise) | No throughput gain: uncontended mutexes are cheap, and the profiler overstated them. Kept because a search now never waits behind an insert, and the memory-ordering argument is simple (see [`hnsw.h`](../engine/include/oryx/hnsw.h)). Checked with ThreadSanitizer under concurrent inserts and searches |
| **8-bit visited tags** (1 MiB for 1M nodes vs 2 MiB) | −1% to +3% (noise) | Fits the P-core's 1.25 MiB L2 and halves per-search scratch memory. The extra full clear every 255 queries costs ~0.1 µs per query |

## Tried and removed

| Idea | Result | Why it didn't help |
|---|---|---|
| **BFS graph reordering** (renumber nodes so graph neighbors are adjacent in memory) | 0% | Node records are 704 bytes, so neighbors never share cache lines; huge pages had already solved the TLB problem; and a query's ~1,500 nodes don't fit in L3 in any order |
| **Exact early abandoning** (stop an L2 computation once the partial sum exceeds the current k-th best) | −10% (checks every 32 dims), 0% (every 64) | Lossless, but the horizontal sum at each checkpoint breaks the FMA pipeline, and the prefetcher has already pulled the lines the early exit would skip. Probably worthwhile only with a variance-concentrating rotation on high-dimensional data (ADSampling-style); revisit with 512–1536-dim embeddings |
| **LTO and PGO** | ±2% (noise) | The code is memory-bound; better branch layout doesn't shorten DRAM latency |
| **Chunked snapshot I/O** (8 MiB `fread` + `memcpy` instead of `istream`) | 0.31 s → 0.31 s to load 1M vectors | Loading was dominated by allocating and zeroing 764 MB and rebuilding the label map, not by parsing. Kept for its explicit error handling, without claiming a speedup |

## Result against other libraries

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/sift1m-recall-qps-dark.svg">
  <img alt="Recall versus QPS on SIFT1M: Oryx is above hnswlib and FAISS at every recall level" src="img/sift1m-recall-qps-light.svg">
</picture>

At the same recall, Oryx serves **1.46× hnswlib's QPS and 1.40× FAISS's**, and
builds the graph faster (42.9 s vs 55.4 s and 68.4 s on 12 threads). Raw data:
[`bench/results/sift1m-compare.json`](../bench/results/sift1m-compare.json).

To be clear about where the gain comes from: all three libraries implement the
same algorithm, and graph quality is essentially identical (recall matches at
each `ef`). The difference is memory-system engineering, mainly batched
prefetching and huge pages, which hnswlib and FAISS's HNSW don't use by default.

## Next levers

The remaining cost is bytes moved per distance evaluation, so the next wins are:

1. **int8 scalar quantization** (milestone 7): 4× fewer bytes per vector, with
   full-precision re-ranking of the top candidates.
2. **Graph-parameter tuning** (M, efConstruction) for the best
   recall-vs-distance-computations trade-off.

## Reproduce

```bash
scripts/download_sift1m.sh
pip install ./engine hnswlib faiss-cpu numpy
python bench/compare_libs.py --data data/sift --cache data/indexes --out bench/results/sift1m-compare.json
python bench/plot_results.py --compare bench/results/sift1m-compare.json \
    --optimizations bench/results/optimizations.json --out docs/img
```
