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

## 8-bit scalar quantization (SQ8)

After the memory-system work, the remaining cost was bytes moved per distance
evaluation. SQ8 stores each node as uint8 codes, with a per-dimension affine
map `x ≈ min + code × scale` learned by `train()`. A 128-d layer-0 record
shrinks from 704 to 320 bytes, so traversal touches about half the cache
lines.

- **Asymmetric distance.** The query stays in float. Its dot product with a
  dequantized node is `C + Σ wᵢ·codeᵢ`, where `wᵢ = qᵢ·scaleᵢ` and `C`
  are computed once per query. The inner loop is therefore a single
  uint8×float dot product (AVX2: `vpmovzxbd` → `vcvtdq2ps` → FMA; also
  AVX-512, NEON and WASM SIMD128). For L2, the dequantized node's squared
  norm is stored in the record.
- **Exact re-ranking.** When full vectors are kept, the beam's candidates are
  re-scored with float distances, so recall matches the float index. Without
  them, the index stores about 4× less vector data and loses some recall.
- **Construction** uses exact distances whenever full vectors exist, so the
  graph itself doesn't degrade.

SIFT1M, same machine and methodology (median of 5 interleaved runs):

| Mode | Recall@10 at ef=80 | QPS at ef=80 | p99 | QPS at ef=200 | Snapshot |
|---|---|---|---|---|---|
| float | 0.9744 | 10,394 | 191 µs | 4,667 | 658 MB |
| **SQ8 + re-rank** | **0.9744** | **15,365 (+48%)** | **118 µs** | **6,829 (+46%)** | 791 MB |
| SQ8, codes only | 0.9666 | 17,015 (+64%) | 109 µs | 7,649 | **279 MB** |

SIFT's components are small integers, so 8-bit codes are nearly lossless on
it, which flatters SQ8. The honest check is real float embeddings: 28,825 CLIP
ViT-B/32 image vectors (512-d, cosine) with 1,000 held-out queries:

| Mode | Recall@10 at ef=64 | QPS at ef=64 | Best recall | Index |
|---|---|---|---|---|
| float | 0.9979 | 9,229 | 0.9997 | 64 MiB |
| SQ8 + re-rank | 0.9981 | 11,570 (+25%) | 0.9999 | 78 MiB |
| SQ8, codes only | 0.9879 | 13,364 (+45%) | 0.9890 | 22 MiB |

With re-ranking, SQ8 is strictly better (same or higher recall, faster).
Without it, codes-only trades a recall ceiling of about 0.989 for roughly 3×
less memory.

## Exact ground truth: cache blocking

The benchmark lab (and `bench_recall`) needs the exact 10 nearest neighbors of every query.
The first version scanned the whole dataset once per query. That is bound by memory
bandwidth: 1,000 queries over 100k × 128-d vectors (51 MB) stream about 51 GB from DRAM.
`compute_ground_truth` now takes 32 queries at a time and walks the data in 128 KiB blocks
that stay in L2, so each block is fetched once per tile instead of once per query. Results
are bit-identical (same distances, same scan order per query).

| 100k × 128-d, 1,000 queries | Per-query scan | Blocked |
|---|---|---|
| Native, 12 threads | 898 ms | 346 ms (2.6×) |
| Native, 1 thread | 3,000 ms | 1,180 ms (2.5×) |
| WebAssembly, 12 threads (Node) | 825 ms | 480 ms (1.7×) |
| In the browser lab | 1.0–1.3 s | 0.53–0.61 s |

A WebAssembly-specific trap: the first blocked version was **slower** in the browser (2.7 s),
though 2.6× faster natively. V8 runs WebAssembly first in a baseline tier and only switches a
function to optimized code on a later call; a long loop that is already running never switches.
With the whole scan inside one call per thread, it ran unoptimized from start to finish
(`node --no-liftoff` made it fast, which confirmed the cause). The fix: the inner block scan is its
own function, called ~12,500 times, plus a ~6 ms single-threaded warm-up in WebAssembly builds, so
the optimizing compiler finishes while the CPU is idle rather than while every core is busy.

## Web demo: time to first results

Measured on the Try page (local server, 12-thread laptop):

| | Before | After |
|---|---|---|
| Index ready | 3.5–3.8 s (built in the browser on every visit) | 0.1–0.2 s (prebuilt graph) |
| First results, first visit | 15–16 s | 1.9 s |
| First results, repeat visit | 15 s | 1.2 s |

- **Prebuilt graph.** `HnswIndex::export_graph` writes only the links (16-bit ids below 65,536
  nodes): 1.3 MB for the 29,825-photo index, next to the 15 MB of int8 vectors the page already
  downloads. `import_graph` rebuilds the index from the two, validating every link. Phones,
  which took 20–40 s to build the graph, gain the most. `web/scripts/build-graph.sh` regenerates it.
- **Nothing waits on anything it doesn't need.** Only the tiny manifest precedes the big
  downloads; vectors, graph, photo metadata and the map layout then arrive in parallel.
- **Cache reads skip the service worker.** On repeat visits the data is read straight from the
  Cache API. Through the service worker's fetch handler, its single thread relayed every byte and
  the dataset queued behind the model downloads it was also relaying (5 MB of metadata took over 5 s).
- **Models wait their turn.** The CLIP encoders (67 and 89 MB) start once the index is ready,
  instead of competing with it, and warm up in the background on idle (skipped on Data Saver or
  2G/3G). They are cached once by the service worker, not a second time by transformers.js.
- **No wasted first-visit download.** The very first visit reloads once to become cross-origin
  isolated (needed for WebAssembly threads); downloads now start after that reload.

## Next levers

1. **Graph-parameter tuning** (M, efConstruction) for the best
   recall-vs-distance-computations trade-off.
2. **Product quantization** for datasets where memory, not speed, is the limit.

## Reproduce

```bash
scripts/download_sift1m.sh
pip install ./engine hnswlib faiss-cpu numpy
python bench/compare_libs.py --data data/sift --cache data/indexes --out bench/results/sift1m-compare.json
python bench/plot_results.py --compare bench/results/sift1m-compare.json \
    --optimizations bench/results/optimizations.json --out docs/img
```
