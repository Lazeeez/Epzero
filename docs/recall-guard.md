# Recall you can see, and keep

Approximate nearest-neighbor search is fast because it doesn't compare the query with every
vector: an HNSW search walks a graph and stops early. Its accuracy, *recall*, is whatever that walk
happens to find, and it depends on the data, on the queries, on deletes, and on filters. In
production, nobody measures it. Teams pick a beam width (`ef`) once, from a benchmark, and then
hope.

This is a problem people face now. Retrieval-augmented generation (RAG) puts a vector index in front
of every LLM answer, so a retrieval miss becomes a wrong or unsupported answer, and nothing reports
it. Indexes are updated continuously, filters (per tenant, per permission, per category) shrink the
searchable set, and the queries change as products and embedding models change. Each of these
moves recall, silently.

Oryx makes recall observable and controllable, with three parts:

| | What it does | API |
|---|---|---|
| **Recall monitor** | Re-answers a small sample of real queries exactly, in the background, and reports live recall@k with a confidence interval, per traffic segment | `RecallMonitor`, `RecallGuard::report()` |
| **Recall guard** | Holds a recall target by moving the beam width: wider as soon as measured recall falls short, narrower only on confident surplus | `RecallGuard`, `oryxdb.RecallGuard` |
| **Miss explainer** | Answers "why wasn't this document returned?": filtered out, not near enough (the search was right), or missed by the walk, and then which beam width finds it | `HnswIndex::explain()`, `Index.explain()` |

## Using it

```python
import oryxdb as oryx

index = oryx.Index(dim=768, max_elements=1_000_000)
index.add(vectors, threads=16)

guard = oryx.RecallGuard(index, target=0.95, sample_rate=0.01)
labels, distances = guard.search(queries, k=10)          # serves at the current beam width
guard.report()
# [{'segment': 'unfiltered', 'ef': 72, 'changes': 3, 'samples': 1000,
#   'recall': 0.957, 'lower': 0.944, 'upper': 0.968}]

index.explain(query, k=10, label=4217, ef=64)
# {'cause': 'search-missed', 'exact_rank': 6, 'scored': False, 'ef_needed': 128, ...}
```

The C++ API is the same (`include/oryx/recall.h`, `HnswIndex::explain`). Filtered searches are
monitored and steered separately, in bands of filter selectivity, because a 1% filter and an
unfiltered query need very different beam widths.

## How it works

**A reference to compare against.** `HnswIndex::exact_search()` scores every live vector (only the
allowed ones, with a filter): the truth, at the cost of a full scan. On SIFT1M that is 29 ms per
query on one core, against 0.1 ms for a search at `ef` 80, so at 1% sampling exact checks would
cost about 3x the CPU of the searches themselves (on their own thread, so search latency is
unaffected, but it is CPU). The monitor can instead compare against a far wider search: at `ef`
1024 a check takes 0.96 ms, 31x cheaper than a scan, and that reference's own recall is 0.9995, so
the estimate is off by at most 0.0005. At 1% sampling that is about 10% of the search CPU
(`reference_ef` in `RecallMonitor::Options` and `oryx.RecallGuard`).

**Sampling that never slows search.** The monitor takes every *n*-th query (systematic sampling: a
lock-free counter, no random number generator on the search path), copies it, and queues it for
its own thread. The queue is bounded; when it's full, samples are dropped rather than waited for.

**An honest interval.** Recall over a window of sampled queries is reported with a Wilson score
interval (one observation per query). Unlike a normal-approximation interval, it stays meaningful
near 1, where recall lives: when every sample is perfect, it still says how sure that is.

**A control rule that reacts to recall, not to uncertainty.** The guard widens the beam (×1.5) as
soon as the *measured* recall falls below the target, and narrows it (×7/8) only when the
interval's *lower* end clears the target by a margin; otherwise it holds, and after any change it
measures afresh. An earlier version widened whenever the lower bound fell short, which reacted to
small sample sizes rather than to recall and over-provisioned; this asymmetric rule fixed it.

**Explaining a miss.** `explain()` finds the label's exact rank among the vectors the search could
return. Rank beyond *k* means the search was right and the document simply isn't among the
nearest (an embedding or chunking question, not a search one). Rank within *k* means the walk
missed it: a traced search tells whether it was ever scored, and doubling the beam width shows how
much effort finds it.

## Evidence

`engine/bench/bench_guard.cpp` streams queries through four kinds of drift on SIFT1M (1M × 128-d)
and runs three policies on identical streams, measuring true recall against exact ground truth.

1. **Steady:** the queries the index was tuned on.
2. **Harder queries:** queries shifted off the data distribution, like a new kind of user.
3. **Churn:** 20% of the vectors deleted and re-inserted.
4. **Filtered:** every query limited to a random 5% of the vectors.

Results on a dedicated server (2 × Xeon Gold 6226, clock fixed at 2.7 GHz;
[`scripts/pace/guard.sbatch`](../scripts/pace/guard.sbatch)), six passes of 2,500 queries per
phase, 2% sampling. At a target of 0.95:

| Policy | Batches meeting the target | After adapting (last pass per phase) | Mean cost (distances/query) |
|---|---|---|---|
| Fixed `ef` 64, tuned on the steady phase (usual practice) | 12 / 24 | 2 / 4 | 1,210 |
| Fixed `ef` 96, sized for the worst phase | 24 / 24 | 4 / 4 | 1,659 |
| **Recall guard**, exact checks | **24 / 24** | **4 / 4** | **1,562** (−6%) |
| **Recall guard**, wide-beam checks (`reference_ef` 1024) | 20 / 24 | **4 / 4** | **1,410** (−15%) |

- **The usual practice fails silently:** a beam width tuned once misses the target on harder
  queries (0.948) and under the filter (0.9496), in every batch, and nothing reports it.
- **The guard holds the target everywhere once it has adapted,** at 6–15% less cost than sizing
  for the worst case. How much it saves depends on whether it narrows again: the steady phase's
  recall at `ef` 64 (0.956) sits so close to the target that sampling noise made the exact-check
  run widen early and, narrowing only on strong evidence, stay wide.
- **The monitor tells the truth:** across all four runs its estimates were off by 0.004–0.011 on
  average, and the true recall was inside its 95% interval in 90 of 91 checked batches.
- **Where no drift pushes recall below the target, it costs nothing extra:** at 0.90 every phase
  stays above the target at `ef` 48 and the guard matches the fixed setting (−0.5%); at 0.99 it
  matches it within 4% (it briefly widened under the filter).

The development laptop showed the same pattern (fixed-tuned 2 / 4 phases, guard 4 / 4 at 15% less
than worst-case sizing). Raw data:
[`bench/results/pace/gold6226/sift1m-guard-*.json`](../bench/results/pace/gold6226/) and
[`bench/results/sift1m-guard-laptop.json`](../bench/results/sift1m-guard-laptop.json).

The website shows the same loop live: the playground's *Recall guard* card measures every search
against an exact scan, plots it against the target, and moves the beam width when recall falls
short. Shift-clicking a star in the 3D view (tapping, on a phone) explains why that photo is or
isn't among the results.

## What didn't work

The first idea was per-query adaptivity: predict from a cheap probe search how hard each query is,
and give each query its own beam width. On SIFT1M the probe's features do correlate with the effort
a query needs (Spearman 0.43 for local intrinsic dimensionality), and needs vary widely (median
`ef` 32 for a perfect top 10, 400 at the 99th percentile). But turning that into savings failed.

- **Bucketing queries by difficulty and meeting the target per bucket** cost 3–60% *more* than one
  fixed beam width, because a fixed width lets easy queries over-deliver for hard ones.
- **Optimal allocation across buckets** (spending effort where it buys the most recall) saved
  2–20% before counting the probe; counting it, results ranged from 5% cheaper to 49% more
  expensive. Part of the saving was only the coarseness of the fixed-width ladder.

So the guard controls one beam width per traffic segment, against measured recall, rather than
predicting per query. The data in this section came from a scripted study on 10,000 SIFT1M
queries; the scripts are summarized here rather than kept, since the approach was dropped.

## Limits and next steps

- **Checks cost CPU.** An exact check is a full scan, so its cost grows with *n*; the wide-beam
  reference keeps it near 10% of search CPU at 1% sampling on 1M vectors, and the sampling rate
  trades CPU for how quickly drift is detected.
- **Adaptation lags by about `min_samples / sample_rate` queries.** That's 2,000 at 5%; at
  production traffic it is seconds.
- **Narrowing is deliberately conservative.** On the server it kept `ef` 96 through the steady
  phase when 64 would have sufficed: it narrows only on strong evidence, trading cost for safety.
  When the target sits within sampling noise of what a beam width delivers, a smarter narrowing
  rule (e.g. one that remembers the last width that held) would save more.
- **More data:** GloVe-100 and higher-dimensional text-embedding sets, whose recall is harder to
  hold, are next.
