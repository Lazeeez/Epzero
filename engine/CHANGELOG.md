# Changelog

## Unreleased

**Staying accurate after the build.**

- `index.recalibrate(queries)` (C++: `epzero::recalibrate`): re-measures the beam width for the
  recall target on real queries and saves it in the profile, without rebuilding.
- `index.health()` (C++: `check_health`): live and deleted vectors, growth since calibration, and how
  much of the data the principal directions still capture, with advice in plain words: compact,
  recalibrate, or rebuild.
- `RecallGuard(index)` takes its target and starting beam width from the profile.
- `index.check_queries(queries)` (C++: `check_queries`) says whether queries look like the data: text
  queries against image vectors sit 2.2-2.3× farther from their nearest neighbours than rows of the data,
  against 0.99-1.09× for queries like the data, and recall calibrated on the data then falls to 0.74-0.80
  for a 0.95 target. Its advice: `recalibrate(queries)`, which restored 0.945-0.958 on VIBE's
  text-to-image sets.
- `index.requantize()` (C++: `HnswIndex::requantize`, then `refine_links`): for vectors added since the
  build that lie outside the ranges the 8-bit codes were fitted on. It refits the ranges on the live
  vectors, re-encodes them and reconnects the graph, in about half a rebuild's time; after a 4 sd drift
  it brought the beam width for recall 0.95 from 200 back to a rebuild's 12. `health()` reports
  `code_error` and advises it above 0.01.
- `index.get_vector(label)` returns a stored vector.

**Information retrieval.** `epzero.ir`: a `Collection` of documents with string ids, retrieved by vector,
by keyword (BM25) or both (reciprocal rank fusion), growing and shrinking after the build; TREC run files
in and out; relevance judgements in TREC or BEIR form; nDCG@k, MAP, recall, precision and MRR, matching
trec_eval (tested against it); and a BEIR dataset loader.

**Stemming.** `TextIndex` reduces English words to their Porter stems, so "connections" matches
"connected" (`stemmer="none"` keeps words as written). It is the reference implementation's algorithm,
the one Lucene and Anserini use, and agrees with NLTK's on all 39,011 words of SciFact's and NFCorpus's
vocabularies. nDCG@10 for BM25 went from 0.660 to 0.688 on SciFact and from 0.305 to 0.321 on NFCorpus
(BEIR's Anserini baselines: 0.665 and 0.325), and for hybrid search with all-MiniLM-L6-v2 from 0.682 to
0.713 and from 0.344 to 0.354. Stemmed text indexes are saved as text format 2; files saved before load
unstemmed, as they were written.

**Fixes.** `hybrid_search` searched with a fixed beam width of 64 and the default re-rank, ignoring the
index's calibrated profile; it now uses the profile unless `ef` is given. Calibration could settle on a
beam width below k, which only k-sized searches had measured.

**File format 5.** Indexes with a profile or 16-bit vectors are written as version 5; 0.4's version 4
files are still read (a file written by 0.4.0 is part of the test suite). See `docs/file-format.md`.

**Indexes grow.** `add()` no longer fails when an index is full: it grows the capacity (at least
doubling), copying the nodes and keeping the graph; `index.reserve(n)` and `index.capacity` do it ahead.
The LangChain, LlamaIndex and Haystack stores, which rebuilt their whole graph each time they filled, now
grow in place: loading 40,000 documents in batches of 100 went from 6.0 s to 3.4 s. Searches on other
threads wait for a growth, which moves the index's memory; with a RecallGuard attached, growing must be
done beforehand with `reserve()`.

**Metadata in the engine.** `epzero.Metadata` (C++: `MetadataIndex`) keeps str, int, float and bool fields
per label, with an inverted index per field and ordered numeric values, and `select(where)` returns the
labels that pass ({field: value}, {field: [values]}, or `$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`,
`$lte`, `$exists`) for a filtered search. The LangChain store now runs dict filters there instead of
checking every document in Python. Saved in its own file format, fuzzed in CI.

**Memory.** `half_vectors=True` keeps the full vectors as 16-bit floats: about half an index's memory,
at some speed (up to 17% on GIST1M) and recall (0.003-0.004 on OpenAI embeddings), so never chosen
automatically. Full-precision search is 4-9% faster, as vector addresses are resolved once per query.

**LlamaIndex.** `epzero.llamaindex.EpzeroVectorStore` (`pip install "epzero[llamaindex]"`): nodes with
their text and metadata, filters in the engine, `delete_ref_doc`, and persistence.

**Haystack.** `epzero.haystack.EpzeroDocumentStore` and `EpzeroEmbeddingRetriever`
(`pip install "epzero[haystack]"`): Haystack filters in the engine where the answer is provably the same,
and Haystack's document-store test suite passes.

**Windows.** The engine builds with MSVC and its tests run on Windows in CI; wheels for Windows (x64).

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
