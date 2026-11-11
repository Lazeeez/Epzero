# Changelog

## Unreleased

**Staying accurate after the build.**

- `index.recalibrate(queries)` (C++: `epzero::recalibrate`): re-measures the beam width for the
  recall target on real queries and saves it in the profile, without rebuilding.
- `index.health()` (C++: `check_health`): live and deleted vectors, growth since calibration, and how
  much of the data the principal directions still capture, with advice in plain words: compact,
  recalibrate, or rebuild.
- `RecallGuard(index)` takes its target and starting beam width from the profile.
- `index.get_vector(label)` returns a stored vector.

## 0.4.0

**Automatic configuration.** `epzero.build(vectors, metric=...)` (C++: `epzero::build_auto` in
`epzero/auto.h`) configures an index from its data and the machine it is built on:

- From the data: 8-bit codes, whether to walk the graph on principal directions (at 256 dimensions or
  more, when few directions hold most of the variance), the build passes, the re-rank, and the beam
  width (`ef`) that reaches a recall target, calibrated on rows held out of the graph or on queries
  you pass (`queries=`).
- From the machine: the prefetch setting, timed on this CPU, and every core.
- Measured on an Intel Xeon Gold 6226 and an AMD Threadripper PRO 9985WX: as fast as hand-tuned
  settings (within 1-5%), recall 0.949-0.957 on real queries for a 0.95 target, and 1.3-4.1× Glass's
  queries per second on GIST1M, OpenAI DBpedia and DEEP10M.
- Every setting can be fixed instead (`epzero.build(vectors, nav_dim=0, M=32)`, or `epzero.plan()`,
  edit, build). `index.plan` shows each setting and why it was chosen.
- The choices are saved inside the index (`IndexProfile`, file format v4), and `search()` uses them
  when called without settings. A loaded index re-applies its prefetch setting only on the same kind
  of CPU (`epzero.machine()` names it).

**Faster search and builds.**

- `nav_dim`: walk the graph on 8-bit codes of principal directions, re-ranking with the full vectors.
  With the other changes here, 2.8-4.2× Glass's queries per second on GIST1M and 1.6-2.7× on OpenAI
  DBpedia embeddings at equal recall, on Intel and AMD servers.
- `refine_links()`: a second build pass in which every node searches the finished graph and
  reconnects; `fill_free_links()` gives free link slots to two-hop neighbours.
- `fast_build`: build on the 8-bit codes.
- `rerank_z`: stop the re-rank early once the remaining candidates can't reach the top k.
- Queries are projected onto principal directions on the integer kernel; the AVX2 and AVX-VNNI integer
  kernels finish with vector loads instead of a scalar tail.
- The prefetch tuner can fetch a record's every cache line; on a Xeon Gold 6226 that was worth up to
  1.7× on some datasets.
- PCA runs as blocked matrix products (choosing directions 5× faster on GIST1M), training runs on
  every core, and index memory is zeroed on every core, which spreads it across NUMA nodes.

**Fixes.**

- `Database`: a write that could not be applied (the database full, or not yet trained) was written to
  the log first, after which every open failed replaying it. Writes are now checked before they are
  logged, and replacing a vector in a full database frees its slot before compacting.
- `RecallGuard` searched, and measured its reference, with the default re-rank depth; it now uses the
  index's own settings and re-ranks the reference's whole beam. `explain()` and `trace()` likewise.
- The LangChain store copied every stored vector on each add (quadratic in the store's size).
- `pip install epzero` from source failed on machines with gRPC installed: it tried to build the
  cluster node, whose protocol files aren't in the package.

**Compatibility.** Indexes without a profile are still written in the format older versions read
(v2, or v3 with principal directions); automatically configured ones are v4, which 0.3 cannot load.
`search()`'s `k`, `ef`, `rerank` and `rerank_z` now default to the index's profile, which for an index
built by hand is the old defaults (10, 64, 2 × k, off).

Earlier releases are described by their git tags (`v0.2.0`, `v0.3.0`).
