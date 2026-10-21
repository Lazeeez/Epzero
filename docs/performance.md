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
- **Server runs.** The library comparison and the scaling numbers come from a dedicated node on
  Georgia Tech's PACE cluster: 2 × Xeon Gold 6226, AVX-512, clock fixed at 2.7 GHz, no other jobs
  on the node (Slurm `--exclusive`), search pinned with `numactl` to one socket and its memory.
  Runs agree within 0.5%, against a few percent on the laptop. The job scripts are in
  [`scripts/pace/`](../scripts/pace/).

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
| **Lock-free reads.** Neighbor-list slots are written with release stores and read with acquire loads (compiler atomic builtins) | +2% (noise) | No throughput gain: uncontended mutexes are cheap, and the profiler overstated them. Kept because a search now never waits behind an insert, and the memory-ordering argument is simple (see [`hnsw.h`](../engine/include/oryx/hnsw.h)). Checked with ThreadSanitizer under concurrent inserts and searches |
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

The reference run is on a dedicated server node: 2 × Intel Xeon Gold 6226 (Cascade Lake, 24
cores, AVX-512, clock fixed at 2.7 GHz), Slurm `--exclusive`, search pinned to one socket and its
local memory. Repeated runs agree within 0.5%. At the same recall (0.95), Oryx serves **1.13×
hnswlib's QPS and 1.17× FAISS's**, and the lead holds at every recall level (1.12–1.14× and
1.16–1.24× between recall 0.80 and 0.995). It also builds the graph fastest: 26.1 s, against 27.8 s
and 37.8 s, on 24 threads. Raw data:
[`bench/results/pace/gold6226/sift1m-compare.json`](../bench/results/pace/gold6226/sift1m-compare.json).

All three libraries implement the same algorithm, and graph quality is essentially identical
(recall matches at each `ef`). The difference is memory-system engineering: batched prefetching,
the node layout and the visited set.

### Why the laptop showed a bigger lead (huge pages)

The first comparison, on the development laptop, measured **1.46× hnswlib and 1.40× FAISS**.
The server's smaller lead is explained by transparent huge pages:

| Setup (SIFT1M, equal recall, one search thread) | vs hnswlib | vs FAISS |
|---|---|---|
| Laptop: huge pages opt-in (`madvise`); only Oryx asks for them | 1.44–1.55× | 1.39–1.55× |
| Laptop: no library on huge pages (`ORYX_NO_HUGEPAGES=1`) | 1.18–1.25× | 1.21–1.35× |
| Server: huge pages for every program (`always`) | 1.12–1.14× | 1.16–1.24× |

A random walk over a 765 MB graph misses the TLB on almost every node with 4 KiB pages. Oryx
requests 2 MiB pages for its graph (`MADV_HUGEPAGE`), which is worth about 16% on its own: turning
the request off cut its QPS at recall 0.97 from 11,556 to 9,999, and the kernel reported no huge
pages in use. hnswlib and FAISS don't ask, so they get huge pages only where the kernel hands them
to every program. Asking for them is still a real advantage on kernels where they are opt-in; but
the like-for-like comparison is on equal pages, and there Oryx's design is worth about 1.13–1.25×
depending on the CPU. Laptop data:
[`bench/results/sift1m-compare.json`](../bench/results/sift1m-compare.json).

### Against the field: four more datasets

SIFT1M is one dataset with 128 dimensions. [`bench/compete.py`](../bench/compete.py) runs seven
libraries on four more ann-benchmarks datasets on the same server node. The rules match the SIFT1M
comparison: the same graph degree (M = 16, i.e. 32 links on the base layer) and build beam (200),
every index built on all 24 cores and then searched from one process on one thread pinned to one
socket, the median of three interleaved rounds. Each library's QPS is interpolated (log-linear)
to the same recall@10.

| Library | GIST1M, 960-D (recall 0.95) | GloVe-100 (recall 0.90) | DBpedia OpenAI, 1,536-D (recall 0.95) | DEEP10M, 10M × 96-D (recall 0.95) |
|---|---|---|---|---|
| Oryx | 351 | 1,311 | 1,419 | 3,824 |
| Oryx SQ8 | 481 | 1,373 | 2,170 | 4,101 |
| hnswlib | 356 | 1,133 | 1,438 | 3,710 |
| FAISS HNSW | 336 | 1,160 | 1,617 | 3,407 |
| FAISS IVF-PQ | never reaches | never reaches | 1,620 | never reaches |
| USearch | 260 | 734 | 993 | 2,052 |
| VSAG | 414 | 1,050 | 1,798 | 3,113 |
| Glass | **709** | **2,867** | **3,604** | **6,344** |

- **Glass is the fastest on every dataset**, 1.5–2.1× Oryx SQ8. Like Oryx SQ8 and VSAG it walks
  the graph on 8-bit codes and re-ranks with the full vectors, and it measures the best prefetch
  distance for the machine before searching (its `optimize` step). It is the one to learn from:
  tuned prefetch depth and a batch search loop are the next levers worth trying here.
- **Oryx SQ8 is second on all four.**
- **Uncompressed, Oryx is level with hnswlib** on the high-dimensional sets (GIST 351 vs 356,
  DBpedia 1,419 vs 1,438) and ahead on the low-dimensional ones (DEEP10M +3%, GloVe +16%). With
  960 or 1,536 dimensions, each distance is a long, bandwidth-bound scan, and the memory work
  around it (prefetching, the visited set) is a small share of the time. On DBpedia, FAISS HNSW
  (+14%) and VSAG are ahead of uncompressed Oryx.
- **GloVe is compared at recall 0.90**: at M = 16 no library but VSAG reaches 0.95 by `ef` = 800
  (the others top out at 0.942–0.946). Tuned FAISS IVF-PQ reaches 0.95 only on DBpedia.
- **Build is not Oryx's strength here.** On SIFT1M it built fastest; on these sets hnswlib builds
  5–30% faster (GIST 167 s vs 213 s, DBpedia 220 s vs 288 s, DEEP10M 315 s vs 335 s).

| Build, 24 threads | GIST1M | GloVe-100 | DBpedia OpenAI | DEEP10M |
|---|---|---|---|---|
| Oryx | 213 s | 42 s | 288 s | 335 s |
| hnswlib | 167 s | 45 s | 220 s | 315 s |
| FAISS HNSW | 219 s | 45 s | 335 s | 384 s |
| USearch | 178 s | 51 s | 220 s | 422 s |
| VSAG | 159 s | 73 s | 251 s | 468 s |
| Glass | 223 s | 38 s | 299 s | 462 s |

Caveats: VSAG is searched one query per Python call (about 1–2 µs of overhead each, small next to
its 250 µs–3 ms queries). Glass's saved graph lost recall when loaded back, so it is rebuilt for
the search run; its build time is from the build run. The DBpedia dataset is generated the way
ann-benchmarks does it ([`scripts/make_dbpedia.py`](../scripts/make_dbpedia.py)): 990,000 base
vectors, 10,000 queries, exact neighbours. Raw data: `bench/results/pace/gold6226/*-compete.json`.

### Thread scaling

Search throughput on the same server node, SIFT1M, ef = 80 (recall 0.974), median of three runs
([`bench/results/pace/gold6226/`](../bench/results/pace/gold6226/)):

| Threads | Placement | QPS | Speedup | Efficiency | p99 |
|---|---|---|---|---|---|
| 1 | one socket | 5,987 | 1.0× | 100% | 218 µs |
| 4 | one socket | 22,793 | 3.8× | 95% | 229 µs |
| 12 | one socket | 59,636 | 10.0× | 83% | 266 µs |
| 12 | both sockets, interleaved memory | 48,051 | 8.0× | 67% | 334 µs |
| 24 | both sockets, interleaved memory | 83,746 | 14.0× | 58% | 387 µs |

Lock-free reads scale nearly linearly within a socket; the loss there comes from sharing the L3
cache and memory bandwidth. Spreading the same 12 threads over both sockets costs 19%, because
about half of the memory reads then cross to the other socket. The index is small enough (765 MB)
to keep one copy per socket and route each socket's queries to its own copy; if each socket
matched its 12-thread result, the node would serve about 119k QPS instead of 84k. That is an
estimate, not yet measured.

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

## RaBitQ: correct, but not faster in memory

RaBitQ (Gao and Long, SIGMOD 2024) stores each vector as one bit per (randomly rotated)
dimension plus three floats, and estimates distances with a provable error bound
(`Quantization::RaBitQ`, `src/rabitq.h`). Two ways to use it in the graph walk were tried.

1. **Navigate on the estimates, re-rank at the end** (as SQ8 does). On SIFT1M the estimates
   are too noisy at 128 dimensions to steer the walk: recall@10 fell to 0.63 at `ef` 64, against
   0.956 for full precision.
2. **Use the estimate as a filter** (kept): a neighbour whose estimate, less its error bound,
   cannot beat the beam's worst is dropped without reading its full vector; any other is scored
   exactly. The beam then holds exact distances, and recall matches full precision at every beam
   width.

Laptop, one search thread, `ef` 64:

| Data | Recall@10 (float / RaBitQ) | QPS (float / SQ8 / RaBitQ) | Full-vector reads skipped |
|---|---|---|---|
| SIFT1M, 128-d | 0.9566 / 0.9555 | 11,944 / 19,045 / 7,179 | 27% |
| CLIP, 512-d (28,825 photos) | 0.9959 / 0.9960 | 8,875 / 10,855 / 8,978 | 64% |

The bound skips more reads as dimensions grow (it shrinks as 1/sqrt(D)), but skipping them
doesn't pay here: the full vectors live in a separate array, so every read that remains is an
extra cache miss, and computing the estimates isn't free. Loosening the bound (eps0 0.5 instead of
the paper's 1.9) bought 10-20% at a small recall cost, still behind SQ8. RaBitQ belongs where a
full vector is expensive to fetch: on SSD, or in collections far larger than memory. It stays as an
experimental, tested mode for that; in memory, SQ8 is the one to use.

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
2. **Vectors on SSD** (a DiskANN-style layout), where RaBitQ's skipped reads would pay off.
3. **What Glass does better** ([against the field](#against-the-field-four-more-datasets)):
   measure the prefetch depth per machine instead of fixing it, and search a batch of queries in
   one loop so their memory stalls overlap.

## Reproduce

```bash
scripts/download_sift1m.sh
pip install ./engine hnswlib faiss-cpu numpy
python bench/compare_libs.py --data data/sift --cache data/indexes --out bench/results/sift1m-compare.json
python bench/plot_results.py --compare bench/results/sift1m-compare.json \
    --optimizations bench/results/optimizations.json --out docs/img
```
