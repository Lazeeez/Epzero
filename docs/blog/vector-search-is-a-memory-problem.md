# Vector search is a memory problem

*What I learned making an HNSW index 31% faster, and which "obvious" optimizations did nothing.*

I built [Oryx](https://lazeeez.github.io/Oryx/), a vector database, from scratch in C++20. Its
core is HNSW: a layered graph you walk greedily toward a query, scoring each neighbour by its
distance. On paper the hot loop is arithmetic. A 128-dimensional L2 distance is 128 subtractions,
multiplications and additions, which AVX-512 does in a handful of instructions.

So the first profile was a surprise.

## Where the time actually goes

On SIFT1M (one million 128-dimensional vectors), with a search beam that reaches recall 0.97:

| Hotspot | Share of search time |
|---|---|
| The L2 distance kernel | ~57% |
| Beam bookkeeping (heaps, visited set, loop control) | ~30% |
| Locks on neighbour lists | ~12% |

The distance kernel dominates, which looks like a case for better SIMD. But divide the time by
the number of distances computed and it comes to about **63 nanoseconds per distance**. A
128-dimensional distance takes a few nanoseconds of arithmetic. The rest is waiting.

Each vector is 512 bytes, eight cache lines, at an address the search couldn't predict until it
read the previous node's neighbour list. A query touches about 1,500 nodes scattered across a
765 MB graph. Almost every one is a cache miss, and many are also a TLB miss. The kernel's time
is mostly time spent stalled on DRAM.

Once you see search as **a chain of dependent memory accesses with a little arithmetic between
them**, the useful optimizations change.

## What worked

### 1. Overlap the misses: batched prefetching (+15%)

The textbook loop scores each neighbour as it reads the neighbour list: read an id, compute a
distance (miss, stall), read the next id, compute (miss, stall). The misses are serialized.

Instead, Oryx first filters a node's neighbours against the visited set, issues a prefetch for
every surviving vector, and only then computes the distances. Up to 32 misses are now in flight
at once instead of one at a time.

A detail I didn't expect: prefetching only the *first* cache line of each vector is enough.
Prefetching 64, 128, 256 or all 512 bytes measured the same, because the hardware stream
prefetcher picks up the rest once the kernel starts reading. Prefetching nothing was 14% slower.

### 2. Make each miss cheaper: 2 MiB huge pages (+13% throughput, −16% p99)

With 4 KiB pages, the 765 MB graph spans about 190,000 pages, far more than the TLB covers. So
almost every node visit also paid for a page-table walk. Asking the kernel for 2 MiB pages
(`madvise(MADV_HUGEPAGE)` on a 2 MiB-aligned block) cuts that to about 380 pages. I checked
`AnonHugePages` in `/proc/<pid>/smaps_rollup` to make sure they were actually granted.

### 3. Move fewer bytes: 8-bit quantization (+48%)

If a distance costs a cache miss per line, fewer lines means fewer misses. Oryx stores each
vector as 8-bit codes with a per-dimension scale, which shrinks a node's record from 704 bytes to
320. The query stays in float, so the inner loop is a single uint8-times-float dot product. The
final candidates are re-ranked with the full vectors, so recall doesn't drop.

On SIFT1M that's **+48% throughput at identical recall**. On real 512-dimensional CLIP
embeddings, where 8 bits are less flattering, it's still +25%.

## What didn't work, and why that's the interesting part

Every change was kept only if an interleaved A/B test (same graph, alternating binaries, median
of several rounds) showed a gain.

| Idea | Result | Why |
|---|---|---|
| Reorder nodes so graph neighbours are adjacent in memory | 0% | A node's record is 704 bytes, so neighbours never share a cache line anyway. Huge pages had already fixed the TLB, and a query's ~1,500 nodes don't fit in L3 in any order. |
| Stop a distance early once it exceeds the current k-th best | −10% | It's lossless, but each checkpoint's horizontal sum breaks the FMA pipeline, and the prefetcher had already fetched the lines the early exit would skip. |
| Link-time and profile-guided optimization | ±2% | Better branch layout doesn't make DRAM faster. |
| Lock-free neighbour lists | +2% (noise) | The profiler blamed locks for 12%, but uncontended mutexes are cheap. Kept anyway, because a search now never waits behind an insert. |

All four are reasonable ideas for compute-bound code. Here they had nothing to act on.

The same lesson came back later with **RaBitQ**, a 2024 quantization scheme that stores one bit
per dimension and estimates distances with a provable error bound. Used as a filter, it lets the
search skip reading 27% of full vectors on SIFT1M and 64% on CLIP embeddings, with recall
unchanged. It still wasn't faster. The vectors it can't skip live in a separate array, so each
remaining read is an extra cache miss, and computing the estimates isn't free. Skipping work
only pays when the work you skip is what you're waiting on.

## A benchmarking trap

My first comparison, on a laptop, said Oryx served **1.46× hnswlib's** queries per second at the
same recall. On a dedicated server (2 × Xeon Gold 6226, clock fixed, no other jobs, search pinned
to one socket) it was **1.13×**.

The difference was huge pages. On the laptop they're opt-in: Oryx asked for them, and the other
libraries didn't. On the server the kernel gives them to every program. Turning Oryx's request
off on the laptop brought the lead down to 1.18–1.25×.

Both numbers are real, and asking for huge pages is a fair advantage on kernels where they're
opt-in. But the like-for-like comparison is on equal pages. That's the number I lead with now,
with the explanation next to it. For a memory-bound workload, a page-size setting the benchmark
doesn't mention can matter more than the algorithm.

## Threads, sockets and the memory wall

Because reads take no locks, throughput scales almost linearly within a socket: 10× on 12 cores.
The loss comes from sharing L3 and memory bandwidth. Spread the same 12 threads over two sockets
and throughput drops 19%, because about half the reads now cross to the other socket's memory.
The index is small enough to keep one copy per socket; by my estimate (not yet measured) that
would take the node from 84k to about 119k queries per second.

## Against the fastest library, and when it stops being a memory problem

Then I measured Oryx against seven libraries on four more ann-benchmarks datasets, from 96 to 1,536
dimensions and up to 10 million vectors. Its 8-bit mode came second on all four. First was Glass,
from Zilliz, by 1.5–2.1×.

I assumed Glass had better memory tricks, so I read its search loop. Its prefetching, once tuned,
ends up close to what Oryx already did. Two other things weren't:

- **It re-scores 15 candidates with full vectors; Oryx re-scored the whole beam.** At a beam of 400
  on 960-dimensional GIST, that was 400 extra reads of a 3.8 KB vector per query, and re-scoring
  only 15–30 gave exactly the same recall.
- **Its distances are integer arithmetic.** With 8-bit codes, a 960-dimensional node is 960 bytes:
  15 cache lines instead of the 60 of the float vector. Once the bytes shrink that much, the
  arithmetic is visible again, and Oryx's kernel spent about ten instructions per 32 dimensions
  turning bytes into floats before multiplying. Rounding the query's weights to 8-bit integers once
  per query lets one VNNI instruction (`vpdpbusd`) multiply and add 32 or 64 dimensions at a time.
  Recall moved by at most 0.001.

Together, on my laptop: **1.95× faster on GIST** (960 dimensions), +20% on GloVe (100) and +26% on
SIFT (128). The gain grows with the dimension, which is exactly where Glass's lead was largest.

So the thesis needs a footnote: vector search is a memory problem *until you shrink the bytes*.
Quantization moves the bottleneck back toward the core, and then instruction counts matter again.

## Takeaways

1. **Measure what the cycles are waiting on, not just where they're spent.** "57% in the distance
   kernel" pointed at SIMD. "63 ns per distance" pointed at memory.
2. **Overlap misses, make them cheaper, or avoid them.** Prefetching, huge pages and quantization
   each attack one of these, and each worked. Everything else was noise.
3. **Count bytes per query, then instructions per byte.** For graph search, bytes predict
   performance better than FLOPs do, until quantization makes them small.
4. **Benchmark like-for-like, and say how.** Page size, NUMA placement and frequency scaling can
   each move results by more than the engineering you're trying to measure.

The code, the benchmark scripts and every raw result are in the
[Oryx repository](https://github.com/Lazeeez/Oryx). The full log of changes, including the ones
I removed, is [`docs/performance.md`](../performance.md). You can search 30,000 photos with the
same engine, compiled to WebAssembly, [in your browser](https://lazeeez.github.io/Oryx/try).
