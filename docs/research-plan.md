# Research plan: what actually makes in-memory graph search fast

Draft for advisor review, 2 November 2026. Evidence base:
[the literature review](../reports/Graph%20vector%20search%20literature%20review.md) and
[the performance log](performance.md).

## Summary

Epzero began as an HNSW engine and now beats Glass on four datasets on one server. The literature
review found that nearly every technique behind that is already published, so **a "new faster method"
paper would not be safe.** Measurements and explanations are a better basis.

I propose an **experimental study** of what drives in-memory graph search, aimed at PVLDB's Experiment,
Analysis & Benchmark category. Two results anchor it. Both were measured this week and neither was
found in the literature:

1. **A prefetch setting that does nothing on one CPU changes speed 1.6-1.7× on another,** on identical
   search code. No paper we found varies prefetching on identical code across CPUs.
2. **Building a graph on compressed vectors changes its quality, and the direction depends on the
   dataset, even when the graph is walked at full precision.** LeanVec reports no noticeable difference,
   and explicitly says more study is needed.

**Decisions needed from you** are listed at the end.

## Working thesis

> The speed of graph search depends on interactions between compression, graph construction and the
> memory system, and these interactions differ across CPUs and datasets. Results tuned on one
> machine, or attributed to one component, can therefore mislead.

## Evidence so far

**Prefetching (the same grid on both machines; same engine, same kind of GloVe-100 graph).**
QPS with 8 neighbours prefetched ahead:

| Lines prefetched per node | Laptop (Raptor Lake, WSL2), `ef` 200 | Gold 6226 (Cascade Lake), recall 0.90 |
|---|---|---|
| 1 | 6,863 | 1,960 |
| 2 (the whole 104-byte record) | 6,900 (+0.5%) | 3,304 (+69%) |
| 3 | 5,872 (-15%) | 3,182 (-4% against 2) |

The laptop is indifferent between 1 and 2 lines, across five rounds with narrow ranges. The server gains
1.6-1.7× from recall 0.80 to 0.93. This is a sensitivity contrast, not a reversal. The server side is one
graph and one run so far. WSL2 is a possible artifact, and the laptop must be repeated on native Linux.

**Build space (the walk fixed, the graph varied).** Recall at equal `ef`, laptop, mean over seeds.
Seeds change node levels, not insertion order. Re-rank 200.

| Dataset, walk | Built on full vectors | Built on codes | Difference |
|---|---|---|---|
| GIST1M, 128 directions, walk on codes, `ef` 100 | 0.8088 | 0.8465 | **+0.038** |
| GIST1M, walk at full precision, `ef` 100 | 0.8256 | 0.8478 | **+0.022** |
| GIST1M, walk at full precision, `ef` 400 | 0.9530 | 0.9664 | **+0.013** |
| DBpedia 37k sample, 256 directions, walk on codes, `ef` 20 | 0.9196 | 0.9156 | -0.004 |
| DBpedia 37k sample, walk at full precision, `ef` 20 | 0.9522 | 0.9397 | **-0.013** |
| DBpedia 37k sample, walk at full precision, `ef` 80 | 0.9945 | 0.9927 | -0.002 |

Standard deviations across seeds are 0.0005-0.0035 (3-5 builds per cell). The code-built GIST graph wins
at full precision too, so **it is a better graph, not just a better fit for the walk.** That rules out the
explanation I first logged, that the graph suits the space it was built in. It does not establish why.
Code-built graphs also score up to 9% more nodes at the same `ef`, so a recall gain is not fully a speed gain.

**Server comparison with Glass** (one setting: M=16, efc=200; best variant per dataset, chosen on the test
queries, so these are development numbers only):

| Dataset | Recall range | Against Glass | Build, Epzero vs Glass |
|---|---|---|---|
| GIST1M | 0.90-0.95 | 3.08-3.58× | 90-127 s vs 223 s |
| DBpedia (ada-002) | 0.85-0.99 | 1.16-2.07× | 148-353 s vs 299 s |
| DEEP10M | 0.85-0.99 | 1.36-1.56× | 850 s vs 462 s |
| GloVe-100 | 0.80-0.93 | 0.99-1.22× | 127 s vs 38 s |

## What we can and cannot claim

| Claim | Status | Safe form |
|---|---|---|
| Line-count sensitivity differs across CPUs | Apparently new | "We are not aware of prior work measuring a large contrast in the effect of prefetch line count between CPUs on identical search code." |
| Build space changes graph quality, with dataset-dependent sign | Apparently new; conflicts with LeanVec's null | Only after replication on full datasets, and framed as extending LeanVec |
| Same-degree second pass on a finished HNSW | New measurement, known idea | "To our knowledge, no published evaluation..." only after reading the SIGMOD 2026 hubness paper |
| Per-query online re-rank stop | Mechanism apparently new | Only after reading DDC, and never as a statistical guarantee |
| Integer projection of a PCA basis | Narrow | Credit LoRANN |
| Faster than Glass | Measurement | Never as a novelty, and always with build cost |
| New method, first multi-pass HNSW, first factorial | **Unsafe** | Do not claim |

## Experiments

Each has a question, a hypothesis and an outcome that would defeat it.

**E1. Prefetch across CPUs (anchor).** Grid of lines {1, 2, 3, whole record} and depths {all, 8, 4, 2}, on
Gold 6226, EPYC 7513, Xeon 6740P and a laptop on native Linux, with three or more record sizes.
*Hypothesis:* the laptop's L2 adjacent-line prefetcher fetches the second line of a 128-byte pair, so
software prefetch adds nothing there. *Test without root:* offset records so they straddle a 128-byte pair
boundary. If the laptop then becomes sensitive, the mechanism is supported.
*Defeated by:* a sensitivity that disappears on native Linux, or on repeat runs.

**E2. Build space (anchor).** A factorial on one engine: build space {float, 8-bit, PCA d, random projection
d} crossed with walk space {full, codes d}, d from 64 to 512, on at least 8 datasets, 5+ seeds and shuffled
insertion orders. Four hypotheses, distinguishable by this design:
- *H1 denoising:* discarding low-variance directions removes noise that misleads pruning. Predicts PCA helps
  and random projection doesn't.
- *H2 quantization:* predicts 8-bit alone helps. Server data suggests little (+5% on GIST1M).
- *H3 hubness:* predicts a change in the in-degree distribution.
- *H4 spectrum:* the sign is predictable from data, for example from the variance kept or intrinsic dimension.
*Defeated by:* the GIST effect shrinking on modern data, leaving a legacy-dataset curiosity. A clean null
still extends LeanVec to HNSW with effect sizes, so the experiment holds its value.

**E3. Second pass.** Compare the pass directly against a build of equal cost at higher `efConstruction`,
which the literature warns can absorb such gains. Measure candidate quality by insertion time to test
whether early nodes carry the deficit.

**E4. Implementation against graph.** Export one graph into Epzero, Glass and hnswlib and search each. This
separates graph from code, which every current Glass ratio mixes. The review found no such 2025-26 study.
*Risk:* Glass's internals may limit importing a graph.

**E5. Re-rank stop.** Against a retuned fixed depth at equal recall, on the server. Report it as a heuristic.
Address the selection bias in the scored candidates, or say plainly that `z` has no coverage meaning.

**E6. The 31% tail.** A controlled test varying only load offset and masking, to resolve line split against
masked load. A small result; it may become a footnote.

## Evaluation protocol

Current execution is sound (exclusive nodes, single pinned thread, interleaved rounds). The design is not
yet publishable:

| Gap | Fix |
|---|---|
| **Variants were chosen on the test queries** | Split queries into development and final sets, freeze the engine, run final queries once |
| One setting, M=16 and efc=200 | One shared grid across datasets, keeping this setting as a labelled control |
| Three legacy datasets of four, no out-of-distribution queries | Add modern embeddings and OOD query sets. PCA and SQ8 are fitted to the corpus, so this may hurt us: report it |
| One build, one size | Multiple seeds and orders; a size series at fixed recall, including DEEP100M |
| Three rounds, medians | Five runs and paired per-query tests with multiple-comparison correction |
| Small comparator set | Add SymphonyQG, ScaNN, Vamana, LoRANN or justify omissions. Say "six projects", not "eight libraries" |
| Metrics | Add tail recall, 1/Ratio, `k` = 100, memory and multi-thread throughput |
| Environment unrecorded | Pin every library, record commit, flags and huge-page state in each result. Confirm Glass's version and quantization defaults, since `compete.py` passes arguments the current bindings don't show |

## Reading still required before claiming novelty

Hubness paper (SIGMOD 2026), DDC, DARTH+, Kriegel et al. (KAIS 2017), PiPNN, SkipComputing, original RaBitQ,
and LoRANN's code. Some need access we don't have; the library or your account would help.

## Paper outline

*Working title: "What Makes In-Memory Graph Search Fast? A Factorial Study Across CPUs and Datasets".*

1. Introduction and the single-machine problem
2. Background: HNSW, compression, memory
3. Methodology: harness, shared graphs, protocol
4. Memory system: prefetch sensitivity across CPUs (E1, E6)
5. Build space: when compression improves a graph, and why (E2)
6. Construction: what a second pass buys (E3)
7. Implementation against graph (E4)
8. Adaptive search at equal recall (E5, patience)
9. Negative results: two-stage scoring, pruning rules at degree 32, batching, FP16
10. Guidelines for practitioners and benchmark authors
11. Related work and threats to validity

## Timeline and venue

Target: **PVLDB, Experiment, Analysis & Benchmark.** It takes monthly submissions, is single-blind (so the
public epzero.dev name is fine), and needs a reproducibility package at first submission. All SIGMOD 2027
research rounds have closed, and SIGMOD is double-anonymous.

| When | Work |
|---|---|
| November | Advisor review, required reading, freeze protocol, pin software |
| Late Nov - December | E1 and E2 on all CPUs; go or no-go on each anchor |
| January | E3, E4, E5, new datasets |
| February | Final run on held-out queries; draft |
| March 1, 2027 | PVLDB Volume 20's last deadline (abstract February 25). A stretch date |
| Spring 2027 | Volume 21, if dates follow the pattern; not yet published |

I'd plan for spring and treat March 1 as a bonus. A rushed factorial would undercut the paper's whole point.

## Risks

- **An anchor fails to replicate.** E1 survives even as a one-CPU sensitivity result, and a null on E2 still
  extends LeanVec. If both fail, the fallback is a workshop paper or an experience report.
- **Someone publishes first.** VIBE names cross-hardware testing as needed, so this gap may close.
- **Multiple machines bring noise.** Turbo, NUMA and huge-page state need logging, and PACE's shared
  software environment means jobs on different CPUs mustn't overlap.
- **Quota.** 300 GB of scratch limits the size series; DEEP1B is out of reach without other storage.

## Decisions for the advisor

1. Is an experimental study the right form, as opposed to a method paper or a course report?
2. Is PVLDB's EA&B category the right target, or should a workshop come first?
3. Authorship, and how PACE should be acknowledged. Are there rules on publishing work done on PACE?
4. Which of E1-E6 are worth your time, and which should be cut?
5. Can you help get access to the unread papers, and are there datasets or hardware you'd recommend?
