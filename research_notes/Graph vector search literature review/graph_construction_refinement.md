# Graph construction, multi-pass refinement and pruning for proximity-graph ANN search

Review date: 2026-11-02. Literature searched through October 2026. Library source was read at the default-branch HEAD of shallow clones made on this date, so paths can drift.

## How each source was read

### Takeaway
No paper was read cover to cover. Every "T" entry below had its full text extracted, and only the passages quoted here were read. Treat a statement that a paper "does not mention X" as a keyword search result, not as a reading result.

### Cited Findings
Reading labels used throughout:
- **T** = full text obtained; named passages read and quoted from the extracted text.
- **T-partial** = much of the text read, not all.
- **W** = read only through a WebFetch summary of the page, not checked against the text.
- **A** = abstract or README only.
- **S** = search summary only.
- **C** = source code read directly.

| Source | Label |
|---|---|
| HNSW, Malkov & Yashunin, [arXiv 1603.09320](https://arxiv.org/abs/1603.09320) | T |
| DiskANN/Vamana, Subramanya et al., NeurIPS 2019, [proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) | T (sections 2.3-2.4 read directly) |
| Revisiting the Index Construction (FastNSG, FastHNSW), Yang, Xie et al., PVLDB 18(1), [arXiv 2410.01231](https://arxiv.org/abs/2410.01231) | T |
| The Impacts of Data, Ordering, and Intrinsic Dimensionality, Elliott & Clark, [arXiv 2405.17813](https://arxiv.org/abs/2405.17813) | T-partial |
| Down with the Hierarchy, Munyampirwa, Lakshman, Coleman, ICML 2025, [arXiv 2412.01940](https://arxiv.org/abs/2412.01940) | T |
| Graph-Based Vector Search: An Experimental Evaluation, Azizi, Echihabi, Palpanas, SIGMOD 2025, [arXiv 2502.05575](https://arxiv.org/abs/2502.05575) | T |
| Toward Efficient and Scalable Design, Azizi et al., [arXiv 2509.05750](https://arxiv.org/abs/2509.05750) | T |
| tau-MG/tau-MNG, Peng et al., [PDF](https://www.comp.hkbu.edu.hk/~edisonchan/publication/tau-MNG.pdf) | T |
| alpha-CG/alpha-CNG, Li, Yan, Lu, [arXiv 2510.05975](https://arxiv.org/abs/2510.05975) | W for parameters; T for the rule and headline gains |
| FreshDiskANN, [arXiv 2105.09613](https://arxiv.org/abs/2105.09613) | T |
| ParlayANN, [arXiv 2305.04359](https://arxiv.org/abs/2305.04359) | T (parameter table only); C |
| Through the Lens of Hubness, Xu, Dai, Li, Wang, Yue, SIGMOD 2026, [ACM](https://dl.acm.org/doi/10.1145/3802120) | S and README; paper itself blocked with HTTP 403; C for its repository |
| DGM cross-model migration, Gu et al., [arXiv 2609.23036](https://arxiv.org/abs/2609.23036) | T |
| LGP, LLM-guided pruning, Wu, Xu, Silwal, [arXiv 2609.36359](https://arxiv.org/abs/2609.36359) | T |
| EnhanceGraph, [arXiv 2506.13144](https://arxiv.org/abs/2506.13144) | T for the design; S for the 99.8 to 99.9 figure |
| Slipstream, Yang & Zhao, [arXiv 2606.02992](https://arxiv.org/abs/2606.02992) | T |
| Enhancing HNSW for Real-Time Updates, Xiao et al., [arXiv 2407.07871](https://arxiv.org/abs/2407.07871) | T for two-hop passage; W for most numbers |
| When to Repair a Graph ANN Index, Mandarapu & Kunkunuru, [arXiv 2607.00728](https://arxiv.org/abs/2607.00728) | T |
| LR-RN, Sengupta et al., [CEUR](https://ceur-ws.org/Vol-4250/paper776.pdf) | T |
| SPatch, [arXiv 2512.18060](https://arxiv.org/abs/2512.18060) | T |
| RNN-Descent, Ono & Matsui, [arXiv 2310.20419](https://arxiv.org/abs/2310.20419) | T |
| CAGRA, [arXiv 2308.15136](https://arxiv.org/abs/2308.15136) | T (definitions only) |
| ONNG, Iwasaki & Miyazaki, [arXiv 1810.07355](https://arxiv.org/abs/1810.07355) | A |
| WEAVESS survey, Wang et al., [arXiv 2101.12631](https://arxiv.org/abs/2101.12631) | T |
| NSG, Fu et al., [arXiv 1707.00143](https://arxiv.org/abs/1707.00143) | T, little used |
| Lucene PR 416, [GitHub](https://github.com/apache/lucene/pull/416) | W |
| NN-Descent, Dong et al., WWW 2011 | Not obtained; the PDF did not download. Described through Azizi et al. only |
| HCNNG, Munoz et al., Pattern Recognition 2019 | Not obtained; described through Azizi et al. and ParlayANN code only |

A correction to carry forward: the WebFetch summary of Slipstream claimed it was a multi-pass re-insertion method that addresses early-inserted hubs. The extracted text does not support this. [Slipstream's abstract](https://arxiv.org/abs/2606.02992) describes reusing candidates from the previous insertion in a stream, and a keyword search finds no such re-insertion or early-node discussion. The summary was fabricated by the fetch tool and has been discarded. The same risk applies to every W entry above.

### Inferences
- Numbers labelled W should be rechecked against the text before they go into a paper.

### Gaps
- The hubness paper is the most important unread source. It needs institutional ACM access.

## Which methods make more than one pass or re-insert after construction, and what gain is measured?

### Takeaway
Multi-pass construction is long established, but it is mostly applied to Vamana, NSG-family and kNN-descent graphs, and the gain is often stated without a number. Vamana's two passes are verified from the primary NeurIPS 2019 text, and the passages read give no measured gain for the second pass alone.

### Cited Findings
- **Vamana/DiskANN (T, verified from primary source).** "Our overall algorithm makes two passes over the dataset, the first pass with [alpha] = 1, and the second with a user-defined [alpha] >= 1. We observed that a second pass results in better graphs, and that running both passes with the user-defined [alpha] makes the indexing algorithm slower as the first pass computes a graph with higher average degree which takes longer." Also: "Vamana makes two passes over the dataset, whereas both HNSW and NSG make only one pass, motivated by our observation that the second pass improves the graph quality." — [DiskANN](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf). The arXiv HTML route failed, but this conference PDF is the primary source, so the claim does not remain unverified.
- **Vamana alpha values.** The paper's experiments use L = 125, R = 70, C = 3000 and a value of 2 on the 1M comparison; R = 128 on one billion-point build; and R = 64 with 40 shards for the merged build — [DiskANN](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf). The alpha glyph is dropped by text extraction, so "= 2" being alpha is read from context.
- **Vamana starting graph.** "We have observed that starting with a random graph results in better quality graphs than beginning with the empty graph." HNSW starts from an empty graph, Vamana from a random one, and NSG from an approximate kNN graph — [DiskANN](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf).
- **Later alpha values.** FreshDiskANN uses R = 64, L = 75, alpha = 1.2 and reports recall stable for all tested indices except alpha = 1 — [FreshDiskANN](https://arxiv.org/abs/2105.09613). ParlayANN's table lists 1.2 for BIGANN, MSSPACEV and SSNPP and 1.0 for TEXT2IMAGE (glyphs dropped; read from context) — [ParlayANN](https://arxiv.org/abs/2305.04359).
- **Second pass is often off in practice.** ParlayANN's `-two_pass` flag defaults to 0, and setting it to 1 gives `num_passes = 2`; earlier passes use alpha 1.0 and the last pass uses the user's alpha ([neighborsTime.C](https://github.com/cmuparlay/ParlayANN/blob/main/algorithms/bench/neighborsTime.C), [vamana/index.h](https://github.com/cmuparlay/ParlayANN/blob/main/algorithms/vamana/index.h)) (C). Its example commands in `vamana/neighbors.sh` pass `-two_pass 0`. A keyword search of the ParlayANN paper finds no discussion of passes.
- **FastNSG (T).** The "self-iterative" loop alternates pruning into an intermediate graph and beam search on that graph. "More iterations improve the search performance but encounter marginal effects after the first two iterations", so the iteration count defaults to 2 — [Revisiting](https://arxiv.org/abs/2410.01231). FastNSG's speedups over NSG are 1.2x, 2.1x, 5.6x, 3.2x, 2.3x and 1.2x on Sift1M, Gist1M, Msong, Crawl, Glove and Deep1M at k = 10.
- **Gist1M QPS at recall@10 of 0.85/0.90/0.95/0.99** — [Revisiting, Table 1](https://arxiv.org/abs/2410.01231):

| Method | Build (s) | 0.85 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|
| NSG | 573 | 971 | 712 | 417 | 133 |
| LSH-APG | 433 | 453 | 305 | 172 | 53 |
| DiskANN | 214 | 657 | 469 | 266 | 83 |
| ParlayANN | 409 | 669 | 472 | 274 | 89 |
| RNN-Descent | 121 | 739 | 539 | 345 | 139 |
| FastNSG | 119 | 964 | 694 | 402 | 148 |

- **tau-MNG (T).** Algorithm 4 takes a parameter tau, a neighborhood size and a beam size. Line 1 constructs a proximity graph, "e.g., NSG or HNSW", with an empty edge set for the result. Then, for each node, it finds approximate neighbors and inserts edges by the tau rule — [tau-MG](https://www.comp.hkbu.edu.hk/~edisonchan/publication/tau-MNG.pdf). At 0.95 (recall@100, metric glyph dropped), tau-MNG is 1.1-1.6, 1.2-2.1, 1.1-1.6, 1.1-1.5, 1.2-2.0 and 1.2-4.1 times faster than baselines on DEEP, SIFT, MSONG, GIST, CRAWL and GLOVE. This compares a differently pruned graph with baselines, not a second pass with its own base.
- **NSG (via survey).** The pipeline is initialization, search, then refinement with connect — [Revisiting](https://arxiv.org/abs/2410.01231); connectivity comes from a traversal step — [WEAVESS](https://arxiv.org/abs/2101.12631).
- **alpha-CNG.** Builds a kNN graph with K = 200, beam-searches it per point, prunes adaptively, adds backward edges with lazy pruning, then checks connectivity. Adaptive pruning reruns with alpha raised from 0.9 by 0.05 up to 1.6 whenever fewer than M/2 edges survive (W) — [alpha-CNG](https://arxiv.org/abs/2510.05975).
- **NN-Descent.** Refines a pre-existing graph over a set number of iterations, sourcing candidates from neighbors and neighbors of neighbors, with empirical cost about n^1.14 (secondary source only) — [Azizi et al.](https://arxiv.org/abs/2502.05575).
- **RNN-Descent (T).** Iterates neighbor updates under an RNG inequality and adds reverse edges so that a stalled update can restart. On GIST1M, construction ran at about twice NSG's speed with equivalent search performance — [RNN-Descent](https://arxiv.org/abs/2310.20419).
- **HCNNG.** Repeats a random hierarchical clustering subdivision several times (secondary source only) — [Azizi et al.](https://arxiv.org/abs/2509.05750).
- **CAGRA (T).** Optimizes a kNN graph after construction with rank-based reordering and reverse-edge addition, measuring strongly connected components and the average 2-hop node count — [CAGRA](https://arxiv.org/abs/2308.15136).
- **DGM-Search (T).** Revisits every vertex of an old-model graph, seeds beam search with current neighbors, and refines under new-model distances. It is 2.27-5.58x faster than the fastest reconstruction baseline and improves averaged Recall@10 by 0.35 percentage points and QPS by 22.3% against that baseline, with bottom-layer degree 64 and construction ef 400 — [DGM](https://arxiv.org/abs/2609.23036). The goal is embedding migration, not improving a graph on unchanged vectors.
- **FreshDiskANN.** Without careful updates, recall falls across 20 cycles of deleting and reinserting 5% of SIFT1M, on statically built HNSW, Vamana and NSG indices at fixed candidate-list sizes — [FreshDiskANN](https://arxiv.org/abs/2105.09613). Its insert path is the Vamana insertion with alpha > 1.
- **HNSW repair and reconnect** are covered in the connectivity section below.

### Inferences
- The strongest documented second-pass rationale in Vamana is adding long-range edges at alpha > 1 on top of an alpha = 1 graph. A second HNSW pass at unchanged RNG pruning cannot be gaining from that mechanism, so it must be gaining from better candidates.
- The diminishing return after two FastNSG iterations suggests testing two extra passes in Epzero before claiming one is the right number.

### Gaps
- No numeric single-pass versus two-pass Vamana result was found in the passages read. Check section 4 of DiskANN and SVS papers if a number is needed.
- The current Microsoft DiskANN repository is now largely Rust and its pass logic was not inspected.
- Maximum degree used in tau-MNG experiments was not extracted.

## Has a full second pass been applied to HNSW itself, at the same degree limit, and measured against HNSW and Glass?

### Takeaway
No published measurement of exactly this was found. There are near precedents: FastHNSW rebuilds HNSW by layer, tau-MNG can start from an HNSW, and a hubness research repository contains an uncalled function doing nearly this. No paper among those downloaded mentions Glass at all.

### Cited Findings
- **FastHNSW (T)** is the closest published result but is not a second pass. It replaces node-by-node insertion with layer-by-layer construction: "FastHNSW finds candidates for each point on the entire dataset in each layer, while OriHNSW only on a subset (the existing nodes in the graph during insertions)." It "shows significantly improved search performance compared to OriHNSW, primarily due to obtaining more accurate [candidate] results" while being 2.4x, 4.6x, 3.0x, 3.7x, 1.6x and 2.1x faster to build on Sift1M, Gist1M, Msong, Crawl, Glove and Deep1M at k = 10 — [Revisiting](https://arxiv.org/abs/2410.01231). The search gain is shown only in figures, with no number in the text.
- **Revisiting on high recall.** "Various approaches exhibit similar performance levels at high recall" because graph structure mainly affects the first phase of search — [Revisiting](https://arxiv.org/abs/2410.01231).
- **tau-MNG** permits HNSW as the starting graph, as quoted above, but the result is a new tau-pruned graph — [tau-MG](https://www.comp.hkbu.edu.hk/~edisonchan/publication/tau-MNG.pdf).
- **Hubness repository (C).** `hnswalg.h` in [GraphANNS-Hubness](https://github.com/H40NaN/GraphANNS-Hubness) has two post-passes:
  - `Optbuild()` takes existing and reverse edges as candidates, weights distance by `((indegree + epsilon) / avg_indegree)^lambda`, and reselects at `maxM0_`. It does no new search. This is the path its published script calls, with epsilon 1.0 and lambda 0.2 on the sample dataset, through `opt_index.cpp`.
  - `BuildFromHnsw()` runs `searchBaseLayer` for every node on the finished graph with `ef_construction_ * 4`, merges in existing neighbors, reselects at `maxM0_` with an in-degree-relaxed rule, and overwrites level 0. Nothing in the repository's tests or scripts calls it.
- **VSAG build cache (C).** `build_with_cache` warm-starts each node's neighbors from a previous build, then refines hit and missed nodes, each round doing parallel search and select, serial writeback and reverse-edge install, then rebuilds route graphs with ODescent — [hgraph.h](https://github.com/antgroup/vsag/blob/main/src/algorithm/hgraph/hgraph.h), [docs](https://github.com/antgroup/vsag/blob/main/docs/docs/en/src/advanced/build_cache.md). VSAG documents it as a build accelerator for recurring snapshot builds, not a quality pass over a finished index.
- **hnswlib primitive (C).** Re-adding an existing label calls `updatePoint(data, id, 1.0)`, which rebuilds neighbors' lists from one-hop and two-hop candidates and then calls `repairConnectionsForUpdate` — [hnswalg.h](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h). No literature measuring a whole-index run of this was found.
- **Keyword search across about 30 downloaded paper texts** finds no occurrence of the word "Glass" as a word, including Revisiting, Azizi, FlatNav, alpha-CNG and DGM.
- **Epzero's pass (C).** `refine_links` walks every node, descends from the entry point, runs `search_layer` at layer 0 with ef defaulting to `ef_construction`, runs `select_neighbors` to `max_links0_`, then `connect` merges with existing links, re-prunes and installs reverse links — `engine/src/hnsw.cpp`. Upper layers are unchanged.

### Inferences
- Epzero's combination appears unmeasured in the literature found: a concurrent in-place second search over a finished HNSW, unchanged RNG selection, unchanged degree, reverse edges kept, base layer only, and comparison against both HNSW and Glass.
- `BuildFromHnsw` shows the idea has occurred to others. A paper should not claim the idea has never been tried. It can say that no published evaluation was found.
- The Revisiting statement about high recall contrasts with Epzero's rise in recall at ef 800, from 0.944 to 0.961. This is a useful result, but it should be stated as a ceiling at fixed ef on GloVe rather than a general claim.

### Gaps
- The hubness paper may report a search-based variant. It must be read before claiming novelty.
- The published FastHNSW figures should be digitized or the code rerun at M = 16 to compare gains directly.
- A paper or workshop could exist outside what search surfaced. Absence here is evidence, not proof.

## Is HNSW's early-inserted-node problem documented, and do insertion orderings matter?

### Takeaway
Yes. The candidate-quality form is stated directly, with an upper bound of 0.5, and order effects up to 12.8 recall points are measured. The fix through a post-build pass on HNSW is the unpublished step.

### Cited Findings
- "The expected value of [candidate] quality in NSWG is only 0.5 even if all the [k-ANN] queries are answered correctly", and "the [candidate] quality of HNSW is limited to 0.5 due to its incremental node-by-node insertion strategy", because construction searches a graph holding only part of the nodes — [Revisiting](https://arxiv.org/abs/2410.01231).
- The same paper shows that candidate quality has a strong positive effect on search performance of the derived graph (its Figure 2) — [Revisiting](https://arxiv.org/abs/2410.01231).
- In NSW, "good candidates [for maximum-degree start nodes] are the first nodes inserted", and hub start points help mainly at low dimension and are worse on high-dimensional data than HNSW — [HNSW](https://arxiv.org/abs/1603.09320).
- Element levels are fixed before insertion, a design the paper links to reducing dependence on insertion order; the extraction of this passage was only partial — [HNSW](https://arxiv.org/abs/1603.09320).
- Inserting by LID changes recall: up to 12.8 percentage points on FAISS (all-MiniLM-L6-v2) and 5.6 on hnswlib (e5-small-v2) at efSearch = 10, and 11.5 and 7.1 at the next efSearch setting. Category order changes recall by up to 8 points — [Impacts](https://arxiv.org/abs/2405.17813). Settings are M = 16, efConstruction = 128.
- Descending-LID insertion had higher recall for all but one model (gte-base with FAISS), and random order fell in between — [Impacts](https://arxiv.org/abs/2405.17813). Average gains were 2.6 and 6.2 points for hnswlib and FAISS in the table at efSearch = 10 (context partly read).
- Claims that early nodes acquire long-range edges and late nodes short-range ones appeared only in low-quality aggregator pages and issue trackers, so they are not cited as findings (S).
- Lucene users report recall loss from graph reuse across merges — [lucene#16552](https://github.com/apache/lucene/issues/16552) (S).

### Inferences
- The 0.5 bound explains Epzero's result. A node inserted at fraction t of the build saw only fraction t of the data, and a post-build search sees all of it. The bound is an argument about expected candidate recall, not a proof that early nodes specifically are damaged.
- A per-node measurement of edge quality by insertion time would turn this argument into evidence, and none was found.

### Gaps
- No primary study was found measuring edge quality or in-degree as a function of insertion position in HNSW.

## Do hnswlib, FAISS, USearch, VSAG, Glass or Milvus Knowhere have a refine or rebuild pass?

### Takeaway
None offers a general post-build quality pass that re-searches every node on a finished HNSW. Knowhere and Lucene have connectivity repair, VSAG has cached refinement, and Glass's HNSW builder is incremental insertion only.

### Cited Findings (all C unless noted)

| Library | What exists | Whole-index quality pass? |
|---|---|---|
| Glass | `HNSW::Build` runs `addPoint` for 0..N-1 in parallel, then copies the base layer and upper-layer lists — [hnsw.hpp](https://github.com/zilliztech/pyglass/blob/master/glass/hnsw/hnsw.hpp). Default R = 32, L = 200. `searcher.optimize()` tunes search on 1000 sampled base points. | No |
| hnswlib | `updatePoint`, `repairConnectionsForUpdate`, `getNeighborsByHeuristic2`, `allow_replace_deleted` — [hnswalg.h](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h). No keepPruned flag found. | No API, though repeated updates could emulate one |
| FAISS | `keep_max_size_level0` default false, `shrink_level_0_neighbors`, `init_level_0_from_knngraph`, `init_level_0_from_entry_points`, `reorder_links`, `link_singletons`, `prune_headroom = 0.2` — [IndexHNSW.h](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.h), [HNSW.h](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.h). Bodies of the reorder and link functions were not read. | None seen; names only for some |
| USearch | `compact` searches for each node's cluster and re-permutes slots for locality, remapping but not choosing links. `isolate` signature only read — [index.hpp](https://github.com/unum-cloud/usearch/blob/main/include/usearch/index.hpp). | No |
| VSAG | Build-cache refinement; ODescent with `turn` default 30, `sample_rate` 0.3, `repair_no_in_edge` with `min_in_degree` 1, then prune and reverse edges — [odescent_graph_builder.cpp](https://github.com/antgroup/vsag/blob/main/src/impl/odescent/odescent_graph_builder.cpp). Experimental adaptive pruning — [docs](https://github.com/antgroup/vsag/blob/main/docs/docs/en/src/indexes/adaptive_pruning.md). | Iterative build, not a pass on a finished HNSW |
| Knowhere | After build, `findUnreachableVectors` then `repairGraphConnectivity` for unreached nodes, in its [hnsw.h](https://github.com/zilliztech/knowhere/blob/main/src/index/hnsw/hnsw.h). | Unreachable nodes only |
| Lucene | `connectComponents` exists but its call in `finish()` is commented out as "exceptionally expensive", see [lucene#14214](https://github.com/apache/lucene/issues/14214). `fixDisconnectedNodes` repairs by search from existing neighbors — [HnswGraphBuilder.java](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphBuilder.java). | No |

- Glass's README states it supports HNSW and NSG; its tree also has NN-Descent and RNN-Descent code under `glass/nsg/` — [pyglass](https://github.com/zilliztech/pyglass).

### Inferences
- Glass's HNSW graph is one-pass, so Epzero's reported 1.36-1.56x over Glass on DEEP10M is plausible as a graph effect in part. It also mixes implementation differences, because Glass and Epzero differ in quantization and traversal code.
- "Refine" in Glass and Knowhere file names (`refiner.hpp`, data-view refiner) appears to mean search-time reranking. Those files were not read.

### Gaps
- Milvus above Knowhere and closed-source Cardinal were not inspected. FAISS `reorder_links` and `link_singletons` bodies were not read.
- A repair pass in pgvector, Qdrant, Weaviate or Vespa was not checked.

## Pruning rules, the fixed degree limit, and filling free slots

### Takeaway
Rule comparisons at degree near 60 favor plain RNG pruning over alpha-relaxation, and alpha and tau wins are reported at larger degrees with adaptive schedules. Filling from pruned candidates is defined in HNSW but no gain is measured there. A direct after-build two-hop versus during-build keep-pruned comparison was not found.

### Cited Findings
- **Rule forms, as stated in** [Azizi et al.](https://arxiv.org/abs/2502.05575): RND keeps a node only if it is closer to the base node than to any kept neighbor; RRND relaxes this by a factor alpha >= 1; MOND uses an angle threshold of at least 60 degrees. In [Revisiting](https://arxiv.org/abs/2410.01231), a pruning angle of 60 equals RNG pruning, and a slightly larger angle improved candidate quality and efficiency up to a dataset-dependent point (about 67 on Glove).
- **alpha-CG rule.** Prune u if an out-neighbor v of p satisfies delta(p,u) > alpha * delta(u,v) + (alpha + 1) * tau — [alpha-CNG](https://arxiv.org/abs/2510.05975).
- **alpha-CNG gain (T).** "Usually reduced distance computations by at least 15% and search steps by over 45% when compared with the best-performing baseline", maximum speedups 2.28x and 2.88x — [alpha-CNG](https://arxiv.org/abs/2510.05975).
- **alpha-CNG parameters (W).** M ranges 50-100 by dataset (SIFT/GIST 50 or 70, WIKI/LAION 70, MSONG/CRAWL 100, DEEP100M 100, BIGANN100M 80), C = 500, K = 200. Baselines include HNSW at M = 32 with efC = 500 and Vamana at M = 70, L = 75, alpha = 1.2. tau mostly below 1 by grid search. The degree bound grows as (alpha * tau)^d.
- **Rule comparison at a fixed limit.** With maximum out-degree 60 and beam 800 on Deep and Sift (25GB, 100GB, 1B), "both RND and MOND consistently outperform, followed by RRND", with NoND worst. Alpha was tuned across 1-2 and 1.3 was best, with theta 60 best in the 50-80 range. Pruning ratios: Deep 20% (RND), 2% (MOND), 0.6% (RRND); Sift 25%, 4%, 0.7% — [Azizi et al.](https://arxiv.org/abs/2502.05575).
- **Tau tuning.** tau-MNG performance "first improves and then deteriorates" with tau — [tau-MG](https://www.comp.hkbu.edu.hk/~edisonchan/publication/tau-MNG.pdf).
- **Vamana on hops.** A search summary reported a stagnation trend in hops for HNSW and NSG while Vamana keeps reducing hops with higher degree and alpha; not checked in the text (S) — [DiskANN](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf).
- **keepPrunedConnections.** The flag "allows getting fixed number of connection per element"; Algorithm 4 line 15 adds discarded connections nearest first while the result is under M. The paper gives no measured benefit in the passages read, and extendCandidates is "useful only for extremely clustered data" — [HNSW](https://arxiv.org/abs/1603.09320).
- **FAISS.** `shrink_neighbor_list` keeps "outsiders" and refills up to `max_size` only when `keep_max_size_level0` is true, the default is false, and the comment gives CAGRA conversion as the reason, not search quality — [HNSW.cpp](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp) (C).
- **VSAG adaptive pruning.** When `0 < |A| < K` it rescans rejected candidates at a relaxed alpha, a schedule step of `alpha0 + delta` at ratio up to 1.5 and `alpha0 + 2*delta` up to 3; experimental, opt-in, L2 only — [docs](https://github.com/antgroup/vsag/blob/main/docs/docs/en/src/indexes/adaptive_pruning.md) (C).
- **Two-hop candidates in repair.** Xiao et al. argue "considering all two-hop neighbors as candidates is inefficient" for update repair, because HNSW already deemed those edges suboptimal — [Xiao et al.](https://arxiv.org/abs/2407.07871).
- **Two-hop candidates elsewhere.** LGP replaces low-value neighbors from directed two-hop non-neighbors using LLM judgment, base layer only on HNSW — [LGP](https://arxiv.org/abs/2609.36359). DGM-Local expands two hops and reaches 97.70-99.78% Recall@10 at ef = 400 in the migration setting — [DGM](https://arxiv.org/abs/2609.23036). NN-Descent is the original neighbors-of-neighbors idea — [Azizi et al.](https://arxiv.org/abs/2502.05575).
- **Pruned edges kept separately.** EnhanceGraph stores pruned edges in a conjugate graph outside the proximity graph, reported to improve Recall@1 from 99.8% to 99.9% on FACE (S for the figure) — [EnhanceGraph](https://arxiv.org/abs/2506.13144).
- **Epzero's fill (C).** `fill_free_links` gathers neighbors of neighbors, drops existing links, sorts by distance, and appends the nearest into free slots. It adds no reverse edges and no diversity check — `engine/src/hnsw.cpp`.

### Inferences
- Azizi's result at degree 60 is the best published support for Epzero's alpha-tau finding, and RRND's near-zero pruning ratio suggests why: a relaxed rule rejects almost nothing, so a hard degree cap truncates to nearest neighbors and loses diversity.
- alpha-CNG's adaptive schedule, which starts near 0.9 and only relaxes on underfill, suggests its authors treat the degree cap as binding. A fixed alpha-tau rule at degree 32 and adaptive alpha-CNG at degree 50-100 are different tests, so Epzero's result does not contradict alpha-CNG.
- Xiao's argument applies to replacing edges under the pruning rule. Epzero fills free slots, so no existing edge is lost. The two findings are compatible.

### Gaps
- No paper was found testing alpha or tau rules at a hard degree of 32 with base recall near 0.9-0.95.
- No measured keepPrunedConnections result was found, either in the HNSW paper's passages or elsewhere.
- Whether a fill measured on HNSW helps equally after a refine pass needs Epzero's own ablation.

## Hubs, hubness-aware construction, in-degree balancing and connectivity repair

### Takeaway
Hubness-aware post-processing is now a published SIGMOD 2026 topic, and in-degree adjustment dates to ONNG. Connectivity repair is well studied for churn, and evidence for timing policies is mixed.

### Cited Findings
- **Hubness paper (S and README).** Hubness creates over-centralized hubs and isolated anti-hubs, which invalidates the greedy heuristic and makes queries targeting anti-hubs performance outliers. It proposes post-processing Hubness-Aware Pruning using in-degree, tested on SIFT, LAION, GIST, DBPedia and Yelp (about 1M each per README; a search summary says up to 100M) — [ACM](https://dl.acm.org/doi/10.1145/3802120), [repo](https://github.com/H40NaN/GraphANNS-Hubness). The code also has an in-degree-relaxed rule of the same shape as alpha-CG, `alpha * d + (alpha + 1) * beta`, and a `LinkAntihubs` function (C).
- **Hub highway (T).** Hubs preferentially connect: at the 95th percentile threshold the null is rejected in all but five cases, and at the 99th in all but one, for both tests. Queries reach highway structures early, with high hub visitation in the first 5-10% of search — [FlatNav](https://arxiv.org/abs/2412.01940).
- **ONNG (A).** Adjusts in-degree and out-degree statically, dynamically by required accuracy, and by removing edges with alternative paths — [ONNG](https://arxiv.org/abs/1810.07355). Implementation is in NGT's [GraphReconstructor.h](https://github.com/yahoojapan/NGT/blob/main/lib/NGT/GraphReconstructor.h) (C, function names).
- **Unreachable points.** Defined as nodes with outgoing but no incoming edges in all layers. Baseline HNSW reaches roughly 3% unreachable in Xiao's replace-update scenario (T for the 3%, W for 3-4% on GIST after 200 iterations); MN-RU variants are 2-4x faster with a backup index and dualSearch (W) — [Xiao et al.](https://arxiv.org/abs/2407.07871).
- **Reconnect.** A global approach re-inserts neighborhood points (S), and local reconnect has "slightly improved recall compared to no patching, yet still worse than others" — [SPatch](https://arxiv.org/abs/2512.18060). LR-RN, strictly one-hop repair, outperforms Native-RU and MN-RU on unreachable growth across 10% cumulative updates — [LR-RN](https://ceur-ws.org/Vol-4250/paper776.pdf).
- **Repair timing.** Signal-triggered repair does not beat fixed cadence at matched budget on SIFT and FashionMNIST. Every 95% CI over four seeds includes zero, and one consolidation pass captures 50.6% of the achievable min-recall gain. The authors withdrew their own v1 claim — [When to Repair](https://arxiv.org/abs/2607.00728). Its concave response also favors a single pass.
- **Survey.** Guaranteed connectivity gives better search than no assurance (C5_NSG) — [WEAVESS](https://arxiv.org/abs/2101.12631).

### Inferences
- If Epzero's refine pass reduces anti-hubs, it would connect to the hubness work. Reporting in-degree distribution before and after is a low-cost addition.
- The matched-budget error in When to Repair is a template for what to avoid when comparing Epzero's pass to a higher-ef build. Run the competing configuration rather than interpolate.

### Gaps
- Hubness-aware gain numbers are unavailable without the paper.
- Epzero's effect on in-degree and unreachable nodes is not in the material given.

## Entry point selection and multiple seeds: does it beat upper layers?

### Takeaway
On high-dimensional data up to 100M, a flat graph matches HNSW with less memory. Random multi-seeding beats hierarchy on small and medium data while hierarchy helps at about 1B, so the literature is size-dependent, not unanimous.

### Cited Findings
- FlatNav reports latency and recall essentially identical to hnswlib on 13 datasets from 1M to 100M at p50 and p99, 38% and 39% lower peak build memory on two Big-ANN datasets and 18% on Microsoft SpaceV. Hierarchy helps only when d < 32 on synthetic data — [FlatNav](https://arxiv.org/abs/2412.01940).
- FlatNav conjectures "the benefits of sophisticated search initialization methods" decay under hub preferential attachment, and that the "largest algorithmic improvements ... should come from optimizations that affect the connectivity and cost of traversal in the base graph, such as link pruning and search algorithm design" — [FlatNav](https://arxiv.org/abs/2412.01940). It compares hierarchy with flat search and does not itself run a seed-strategy contest (W).
- Azizi's evaluation finds K-sampled random (KS) and Stacked NSW (SN) best, with medoid and single fixed random among the least effective. On Deep1M and Deep25GB, SN needs 182 million and 22.3 billion more distance calculations in build than KS, equal to 45K and 1.17M additional 100-NN queries at 0.99 recall — [Azizi et al.](https://arxiv.org/abs/2502.05575).
- "While hierarchical layers ... have shown promise in improving search performance on billion-scale datasets, our experiments demonstrate that a simpler approach like K-random sampling can achieve better results on smaller and medium-sized" collections — [Azizi et al. 2025](https://arxiv.org/abs/2509.05750).
- Lucene's experiment with flat random seeds versus hierarchy on glove-25, glove-200 and sift-128 reports small, dataset-dependent differences; for glove-25 at n_cands = 100, recall 0.946 versus 0.947 and QPS 2,621.846 versus 2,734.787, and 6% more indexing time on glove-100 (W; which side is which should be rechecked) — [PR 416](https://github.com/apache/lucene/pull/416).
- Replacing HNSW's random upper-layer choice with a degree-based selection helped Crawl, hurt Deep1M and cost 2.8% and 8.0% more build time — [Revisiting](https://arxiv.org/abs/2410.01231).
- Adaptive entry point selection (Oguri & Matsui, arXiv 2402.04713) and DiskANN++ (arXiv 2310.00402) are cited by FlatNav and were not read.

### Inferences
- FlatNav's conjecture supports the framing of a base-graph pass: gain should come from base-layer connectivity. Epzero's refine leaves upper layers alone, consistent with this.

### Gaps
- Whether the refined graph needs upper layers at all was not tested. A flat comparison on the refined graph would be a natural extension.

## Epzero findings against the literature

### Takeaway
The mechanism behind each finding has precedents. What appears new is the specific measured result: a same-degree second pass on HNSW compared with HNSW and Glass, and the two-hop-after versus keep-pruned-during comparison. Hold both as "not found in published work" until the hubness paper is read.

### Cited Findings

| Epzero finding | Verdict | Evidence |
|---|---|---|
| Multi-pass construction improves graphs (general idea) | **Already known** | [DiskANN](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf); [Revisiting](https://arxiv.org/abs/2410.01231) |
| Why it helps HNSW: incremental candidates are limited | **Already known** | Bound of 0.5 in [Revisiting](https://arxiv.org/abs/2410.01231) |
| Reselect by searching a finished HNSW, at the same degree | **Partly known** | tau-MNG from a PG; uncalled `BuildFromHnsw` in [GraphANNS-Hubness](https://github.com/H40NaN/GraphANNS-Hubness); VSAG cached refine; [DGM](https://arxiv.org/abs/2609.23036) |
| +28% QPS at recall 0.90 and +37% at 0.93 on GloVe, M = 16 | **Apparently new as a measurement** | No published same-degree HNSW second-pass number found |
| Recall at ef 800 from 0.944 to 0.961 | **Apparently new as a measurement**, in tension with similar high-recall behaviour noted in [Revisiting](https://arxiv.org/abs/2410.01231) | Not found elsewhere |
| 1.36-1.56x Glass on DEEP10M | **Apparently new** | No downloaded paper mentions Glass, and Glass's builder has no post-pass ([hnsw.hpp](https://github.com/zilliztech/pyglass/blob/master/glass/hnsw/hnsw.hpp)) |
| Alpha-tau rule much worse at degree 32 | **Partly known** | RRND trails RND at degree 60 in [Azizi et al.](https://arxiv.org/abs/2502.05575); alpha-CNG wins at M of 50-100 with adaptive pruning ([paper](https://arxiv.org/abs/2510.05975)); no test at degree 32 found |
| Two-hop fill after build beats keeping pruned candidates during it | **Apparently new as a comparison; parts known** | Keep-pruned defined in [HNSW](https://arxiv.org/abs/1603.09320) without a measured gain; two-hop use in [LGP](https://arxiv.org/abs/2609.36359), [DGM](https://arxiv.org/abs/2609.23036), hnswlib `updatePoint`; contrary view for repair in [Xiao et al.](https://arxiv.org/abs/2407.07871) |

The Epzero numbers themselves were supplied in the assignment and were not re-derived here.

### Inferences
Safe wording for the paper:
- Safe: "To our knowledge, no published evaluation applies a full second search-and-reconnect pass to a finished HNSW at the same degree limit and compares it with HNSW and Glass."
- Safe: "Consistent with the candidate-quality limit identified by Yang et al."
- Unsafe: "the first to re-insert nodes after construction", "the first multi-pass HNSW", or "a new idea". Vamana, tau-MNG, FastHNSW, hnswlib's update path and the hubness repository all stand against these.
- The Glass result needs an ablation, because a whole-system ratio cannot be credited to the graph alone. Running the Epzero graph through the same search code with and without the pass, or exporting it to Glass, would isolate the effect.
- The alpha-tau claim should name the exact rule variant and whether alpha was adaptive. If the rule was fixed, the claim does not reach alpha-CNG.
- Compare against a higher-efConstruction or higher-degree build at equal build cost, run directly and not interpolated, because reviewers will ask whether the same build time spent on ef gives the same gain.

### Gaps
- The hubness paper must be read before submission.
- A targeted check of FastHNSW's code at M = 16, and of Glass's RNN-Descent path, would close the two nearest comparisons.
- Build cost of the pass was not supplied, so cost-adjusted gain cannot be assessed here.
