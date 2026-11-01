# ANN Benchmarking Methodology, Threats to Validity, and Top-Venue Expectations (checked 2026-11-02)

**Evidence labels.** The brief asks for three labels. I add a fourth so nothing is overstated. No paper here was read cover to cover, so the label "read in full" is not used anywhere.
- **[FT]** full text retrieved; the stated sections were read directly, but not the whole paper.
- **[ABS]** abstract or landing page only.
- **[SS]** search-engine summary only; treat as a lead, not a fact.
- **[OFFICIAL]** official site or repository page, read through a fetch tool that returns a model summary. Quotes are as that tool reported them.

The source ledger is at the end of the final section. Facts about our repository come from reading it directly and link to no outside source.

## What datasets, metrics, and repetition does a credible evaluation use now?

### Takeaway
The current reference design is VIBE: modern embeddings, in-distribution and OOD queries, held-out test queries, one hyperparameter grid per method across all datasets, Pareto frontiers, single thread on a NUMA-bound socket, five repetitions, and paired significance tests. SIGMOD 2025's graph study adds billion-scale subsets and per-method tuning records. Together they set the bar for a 2027 submission.

### Cited Findings
**VIBE v3 (23 Aug 2026; accepted paper at VecDB@VLDB 2026) [FT]**
- Covers 22 implementations on 11 in-distribution and 8 OOD datasets — [VIBE](https://arxiv.org/html/2505.17810). The VecDB listing of VIBE as accepted comes from the [workshop site](https://vecdb-ws.github.io/vldb2026/index.html).
- Hardware: two Xeon Gold 6230 CPUs (20 cores each, AVX-512), 384 GB RAM. One worker thread on one physical core, SMT off, `numactl` bound to one socket and its local memory — [VIBE](https://arxiv.org/html/2505.17810).
- Default k=100, justified because retrieval systems fetch extra candidates to filter or rerank. Metrics are average recall and QPS — [VIBE](https://arxiv.org/html/2505.17810).
- "Each query hyperparameter combination is run five times, and we use the minimum query time across these runs to compute QPS" — [VIBE](https://arxiv.org/html/2505.17810).
- 1000 test queries per dataset, held out from the corpus for in-distribution data; OOD test queries are held out from the query distribution — [VIBE](https://arxiv.org/html/2505.17810).
- "To keep the comparisons consistent and avoid dataset-specific hyperparameter tuning, we use the same hyperparameter grid for every dataset", with Pareto frontiers reported over all evaluated configurations — [VIBE](https://arxiv.org/html/2505.17810).
- For each dataset it takes each algorithm's fastest configuration reaching at least 95% average recall and compares paired per-query latencies with Wilcoxon signed-rank tests and Holm-Bonferroni correction; 2399 of 2450 comparisons came out significant at the alpha it used — [VIBE](https://arxiv.org/html/2505.17810).
- Index construction was measured on a single CPU core — [VIBE](https://arxiv.org/html/2505.17810).
- It reports hard-versus-easy query breakdowns using relative contrast RC_100 — [VIBE](https://arxiv.org/html/2505.17810).
- It names its own limit: one hardware platform, and relative performance may differ elsewhere (citing Kuffo and Boncz) — [VIBE](https://arxiv.org/html/2505.17810). The repository says the next results update will use AMD Turin 9965 — [VIBE repo](https://github.com/vector-index-bench/vibe).

**Azizi, Echihabi, Palpanas, SIGMOD 2025 [FT]**
- Twelve methods on seven collections, up to 1 billion vectors, per its [abstract](https://arxiv.org/html/2502.05575v2).
- Compiled with GCC 8.2.0 at optimization level 3; a 4-socket Platinum 8276 with 1 thread per core; 48-hour build limit; server used exclusively; caches cleared before each step and kept warm during a workload — [Azizi et al.](https://arxiv.org/html/2502.05575v2).
- Each workload ran six times, the two best and two worst were dropped, and the mean of the rest reported — [Azizi et al.](https://arxiv.org/html/2502.05575v2).
- Queries: 100, processed sequentially; k=10 by default, "per the standard in the community"; reports wall-clock time and distance calculations for both indexing and search — [Azizi et al.](https://arxiv.org/html/2502.05575v2).
- Hard workloads were made by adding Gaussian noise (1% to 10%) to dataset vectors — [Azizi et al.](https://arxiv.org/html/2502.05575v2).

**ann-benchmarks [OFFICIAL]**
- Principles: try multiple parameter values and keep only the frontier; one CPU saturated by default, with `--batch` optional; proper train/test splits; Docker containers; builds should not exceed several hours — [repository](https://github.com/erikbern/ann-benchmarks).
- Runs on r6i.16xlarge with `--parallelism 31` and hyperthreading off — [repository](https://github.com/erikbern/ann-benchmarks).
- The landing site still lists the legacy set: glove, nytimes, fashion-mnist, gist, sift, and so on — [site](https://ann-benchmarks.com/).

**big-ann-benchmarks [OFFICIAL / FT]**
- NeurIPS'23 tracks: filtered, OOD, sparse, streaming, all on small slices of about 10M to 30M. Fixed Azure D8lds v5 (8 vCPU, 16 GB DRAM), 12-hour build limit, ranked by QPS at 90% recall — [track page](https://big-ann-benchmarks.com/neurips23.html).
- Organizers ran entries, measuring 10-recall@10 — [Simhadri et al.](https://arxiv.org/html/2409.17424) [FT].
- The repository points new submissions to an ongoing NeurIPS'23 leaderboard. I found no 2025 or 2026 edition — [repository](https://github.com/harsha-simhadri/big-ann-benchmarks).

**Other current practice**
- SISAP 2026 challenge: 8 vCPU, 24 GB RAM per container, on EPYC 7F72. Fastest solution at or above 0.8 or 0.9 average recall on BGE-M3 Wikipedia (6.4M, 1024-d), an LLM-inspired MIPS workload (about 256k), and SPLADE sparse vectors — [challenge site](https://sisap-challenges.github.io/2026/index.html) [OFFICIAL].
- PAG (arXiv v2, Jul 2026): default K=100, with K=10 and K=1000 also reported; search on one thread, build on all 112 threads; states QPS-recall alone at K=10 is too narrow — [PAG](https://arxiv.org/html/2603.06660) [FT]. Publication venue was not confirmed.
- Filtered ANN benchmark design is called brittle, with implicit or inconsistent choices — [Revisiting Filtered ANN Benchmarks](https://arxiv.org/html/2606.14193v1) [SS]. Out of our scope unless filtering is claimed.

### Inferences
- A safe protocol is the union: VIBE's frontier and grid rules, at least five repetitions, a paired test on per-query latency, and more than one CPU.
- Repetition summary is not standardized. VIBE uses the minimum, Azizi et al. a trimmed mean, and our harness the median. State the choice and show it does not change conclusions.
- Query count varies from 100 to 10,000 across credible work. Ten thousand is not a weakness.

### Gaps
- The EDBT 2026 tutorial [summary](https://helios2.mi.parisdescartes.fr/~themisp/publications/edbt26-FasterAnns-summary.pdf) and [slides](https://helios2.mi.parisdescartes.fr/~themisp/publications/edbt26-FasterAnns-slides.pdf) both returned HTTP 503. Only this much is known [SS]: 1.5 hours, by Chatzakis, Del Gaudio, Sideri (Université Paris Cité), on parallel, distributed, streaming, and early-termination ANN, with closing remarks on quality evaluation. The evaluation content is unread.
- "Revisiting the Index Construction" is a construction paper in PVLDB 18 — [abstract](https://arxiv.org/html/2410.01231v2) [ABS]. Its benchmark setup was not read.

## Why are older datasets criticized, which modern datasets are expected, and where do they come from?

### Takeaway
SIFT, GIST, and GloVe are now treated as secondary. Expected: at least several post-2023 embedding datasets with model named, usually including 1536-d or larger OpenAI or CLIP vectors. Keep GIST and GloVe only as labeled legacy or hardness controls. They are still useful for hardness: GIST has among the highest LID in the Power Law table.

### Cited Findings
**Criticism**
- VIBE: "most of its datasets, such as MNIST, Fashion-MNIST, SIFT, and GIST, are not representative of modern applications", because raw pixels and image descriptors "have been superseded by dense vector embeddings" — [VIBE](https://arxiv.org/html/2505.17810) [FT].
- Iceberg: ann-benchmarks is "constrained to relatively small, legacy datasets" from outdated models — [Chen et al.](https://arxiv.org/html/2512.12980) [FT]. PAG repeats this, noting SIFT and GIST descriptors are over 20 years old — [PAG](https://arxiv.org/html/2603.06660) [FT].
- VIBE still keeps a GloVe-200 Twitter dataset, listed in its repository as `glove-200-cosine` with 1,192,514 vectors — [repo](https://github.com/vector-index-bench/vibe) [OFFICIAL]. PAG and the Power Law paper both still evaluate legacy data — [PAG](https://arxiv.org/html/2603.06660); [Power Law](https://arxiv.org/html/2609.02143) [FT].
- A counterpoint from 2019: Aumüller and Ceccarello are summarized as finding real-world datasets not diverse, so that one dataset predicts the others well — [paper](https://arxiv.org/pdf/1907.07387) [SS].

**Modern datasets and sizes**

| Dataset | n | d | Model | ID/OOD | Where | Evidence |
|---|---|---|---|---|---|---|
| DBpedia OpenAI-1536 / -3072 | 999,000 base, 1,000 queries | 1536 / 3072 | text-embedding-3-large | ID | [Qdrant 1536](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M), [Qdrant 3072](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M) | [PAG Table 2](https://arxiv.org/html/2603.06660) |
| WoltFood | 1,719,611 | 512 | clip-ViT-B-32 | ID | [Qdrant](https://huggingface.co/datasets/Qdrant/wolt-food-clip-ViT-B-32-embeddings) | [PAG](https://arxiv.org/html/2603.06660) |
| AmazonBooks | 15,928,208 | 384 | all-MiniLM-L12-v2 | ID | see PAG citation | [PAG](https://arxiv.org/html/2603.06660) |
| DataCompDr | 12,779,520 | 1536 | PAG table says coca_ViT-L-14; the dataset card says an ensemble of two ViT-L-14 teachers | OOD (text-to-image) | [apple/DataCompDR-12M](https://huggingface.co/datasets/apple/DataCompDR-12M); 2.78 TB in total, Apple license for metadata | [PAG](https://arxiv.org/html/2603.06660) |
| MajorTOM | PAG: 56,506,400. Card: 56,147,150 | 1024 | DINOv2 | ID | [Major-TOM/Core-S2RGB-DINOv2](https://huggingface.co/datasets/Major-TOM/Core-S2RGB-DINOv2), about 240 GB, CC-BY-SA-4.0 | both |
| Cohere Wikipedia 2023-11 | about 247M; English about 41.5M | 1024 | Embed v3 | ID | [Cohere](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3) | card, license not stated |
| OpenAI-ArXiv | not extracted | 1536 | ada-002 | ID | see Power Law citations | [Power Law](https://arxiv.org/html/2609.02143) |
| Wiki-all | not extracted | 768 | MPNet | ID | RAPIDS, see Power Law citations | [Power Law](https://arxiv.org/html/2609.02143) |
| Qdrant-FineWeb-10B | 10B | not read | not read | not read | [Qdrant announcement](https://qdrant.tech/blog/qdrant-fineweb-10b-release/) | [SS] |

VIBE's own datasets, downloadable through [HuggingFace](https://huggingface.co/datasets/vector-index-bench/vibe) [OFFICIAL]:
- In-distribution, as extracted from the paper: DPR-Jina (20,969,760 x 768), MS MARCO-Qwen (8,840,823 x 1024), GooAQ (1,475,024 x 768), arXiv-Nomic (1,344,643 x 768), ImageNet-CLIP (1,281,167 x 512), AGNews-MXBAI (769,382 x 1024), Landmark (760,757 x 768), Yahoo-MiniLM (677,305 x 384), iNaturalist-ResNet (499,000 x 2048).
- OOD: HotpotQA-Harrier (5,233,329 x 640), ImageNet-ALIGN (1,281,167 x 640), LAION-CLIP (about 1M x 512), Yandex (999,000 x 200), two CQADupStack MIPS sets, and Llama and Yi attention MIPS sets. These rows came from the fetch-tool extraction of [VIBE](https://arxiv.org/html/2505.17810); I confirmed HotpotQA-Harrier, LAION, and the repo's agnews, dpr, and glove sizes directly, not every row. Check against the repo before citing.

Other sources: [Marqo benchmark embeddings](https://huggingface.co/datasets/Marqo/benchmark-embeddings) offers 17 models over 7 datasets plus MSMARCO, with LID estimates — [Elliott and Clark](https://arxiv.org/html/2405.17813v1) [FT].

**Intrinsic dimension at 1M, a useful hardness scale** — from [Power Law Table 2](https://arxiv.org/html/2609.02143) [FT]: SIFT 19.2, DEEP 20.7, SpaceV 29.2, Wiki 30.1, GloVe 32.4, Rand64 40.2, GIST 42.5, OpenAI (ada-002 ArXiv) 46.5.

### Inferences
- DBpedia gives us one modern text dataset, though ours is the ada-002 version, not the text-embedding-3-large version in the table above. We have none for images, CLIP, multimodal, or OOD.
- For a laptop-to-node budget, the cheapest strong additions are from VIBE: arXiv-Nomic or GooAQ (text), ImageNet-CLIP or Landmark (image), and LAION-CLIP or HotpotQA-Harrier (OOD). All are near 1M to 5M and ship with a query split.
- DataCompDr and MajorTOM are good 10M+ targets but are heavy to obtain. Only the embeddings are needed; check the file layout first.

### Gaps
- DataCompDr and MajorTOM counts differ slightly between paper and card, and the DataCompDr model label differs. PAG may filter the data, but I did not confirm why.
- I did not verify a DBpedia OpenAI-3072 license or the ann-benchmarks-style HDF5 availability of these sets.
- Big-ANN's "wikipedia-cohere 35M" could not be confirmed from a primary source.

## How do out-of-distribution queries change results, and which benchmarks test them?

### Takeaway
OOD queries slow every method. On the largest shifts general-purpose graphs can fail outright. Corpus-fitted quantizers are specifically named as a cause, which matters directly to an SQ8 engine.

### Cited Findings
- VIBE on LAION-CLIP: throughput falls on OOD queries for every method shown at 95% recall (Figure 8); the factor is in the figure, not the text I read — [VIBE](https://arxiv.org/html/2505.17810) [FT].
- Mechanism: an OOD query can sit far from the corpus, and "if a quantizer is fitted to the corpus, for shifted queries, the quantized vectors can produce larger errors in the estimated candidate distances, steering graph traversal along less useful paths" — [VIBE](https://arxiv.org/html/2505.17810) [FT].
- On text retrieval and text-to-image data, the best general-purpose methods still beat OOD-specific ones, possibly reflecting "implementation maturity and optimization rather than index design alone" (citing Kriegel et al.) — [VIBE](https://arxiv.org/html/2505.17810) [FT].
- On attention MIPS datasets "the graph methods Glass and hnswlib do not reach 50% average recall under any evaluated configuration"; RoarGraph, MLANN, and LoRANN do best — [VIBE](https://arxiv.org/html/2505.17810) [FT].
- ParlayANN on an OOD dataset: at the same recall, graph algorithms were 12.2 to 19.6 times slower than on the other non-OOD datasets — [Manohar et al.](https://arxiv.org/html/2305.04359) [FT].
- Benchmarks that test OOD: VIBE (8 datasets); Big-ANN NeurIPS'23, Yandex Text-to-Image 10M (200-d) — [track page](https://big-ann-benchmarks.com/neurips23.html) [OFFICIAL]; PAG uses DataCompDr — [PAG](https://arxiv.org/html/2603.06660) [FT].
- To show shift, VIBE uses PCA overlap and Mahalanobis distance, with distances in its Appendix C — [VIBE](https://arxiv.org/html/2505.17810) [FT].
- Lightweight alternatives for ID-only datasets: hard query sets chosen by LID or RC — [Aumüller and Ceccarello](https://arxiv.org/pdf/1907.07387) [SS]; Gaussian-noised queries — [Azizi et al.](https://arxiv.org/html/2502.05575v2) [FT].

### Inferences
- An ID-only evaluation of an SQ8 engine invites the exact objection VIBE articulates. One text-retrieval OOD and one cross-modal OOD dataset are the minimum.
- Report re-rank depth (or ef) needed on ID versus OOD at the same recall, so the quantizer effect is measured rather than implied.

### Gaps
- Numeric LAION OOD slowdown factors were in figures, not text.
- No source quantified SQ8 specifically under OOD.

## Recall versus Ratio, k=10 versus k=100, QPS versus latency, threading, memory and build cost

### Takeaway
Keep Recall@k as the primary axis, since reviewers and rankings still use it. Add tail and Ratio views, and report k=10 and k=100, distance computations, memory, and build time. Treat the anti-Recall papers as a reason to add metrics, not to drop Recall.

### Cited Findings
**Ratio (preprint) — [Dimitropoulos and Mamoulis](https://arxiv.org/html/2606.04522), 3 Jun 2026, v1 [FT]**
- 1/Ratio@k = k / sum(retrieved distance i / true distance i).
- At T=0.95 and k=100, satisfying Recall costs on average 1.86x to 9.36x more distance computations (up to 33.3x for HNSW on ImageNet), 3.27x lower QPS on average, and up to 3.63x (HNSW) and 2.82x (SymphonyQG) more build time.
- Rankings stay "largely unchanged"; the relative gaps are not preserved. Near-tied methods may swap.
- HNSW did not reach Recall@100 at or above 0.95 on GIST at any tested configuration.
- Limits: downstream tests use synthesized retrieval sets and one LLM; query heterogeneity is not explored (per the fetch-tool summary of the paper).

**Tail behavior**
- Robustness-delta@K: the fraction of queries with recall above delta. At average Recall@10 of 0.9 on MSMARCO, ScaNN scores 0.991 and DiskANN 0.952 at delta=0.1 — [Wang et al.](https://arxiv.org/html/2507.00379) [FT]. Graph indexes showed more skew, with more high and more low recall queries, and which family wins depends on delta and dataset — [Wang et al.](https://arxiv.org/html/2507.00379).
- With pgvector on GloVe-100, overall recall 0.87 hid clusters at 0.21 versus 0.99 — [RCheck](https://arxiv.org/html/2608.25185) [FT].
- VIBE reports Glass at 81% on the hardest 100 arxiv-nomic queries, with near 100% on easy ones and SymphonyQG more robust — [VIBE](https://arxiv.org/html/2505.17810) [FT].

**Task-level views**
- Iceberg reports method rankings shift when measured by downstream metrics; for instance NSG reaches 99% synthetic recall@100 faster than RaBitQ on Glint360K-ViT yet only 90% label recall, and more slowly — [Chen et al.](https://arxiv.org/html/2512.12980) [FT].
- Semantic Recall and Tolerant Recall, SIGIR 2026, do not penalize misses of irrelevant neighbors — [Kuffo et al.](https://arxiv.org/html/2604.20417) [FT, abstract level].

**k**
- ann-benchmarks and Big-ANN center k=10; Azizi et al. calls 10-NN the community standard; VIBE defaults to 100; PAG treats K as a robustness axis with 10, 100, 1000 — sources as above.
- Ratio-versus-Recall divergence grows with k and, in most cases, LID — [Dimitropoulos and Mamoulis](https://arxiv.org/html/2606.04522).
- PAG finds a leader at K=10 can decline at K=1000 (SymQG) — [PAG](https://arxiv.org/html/2603.06660). On scaling only, k in {1,5,10,20} did not materially change the exponent — [Power Law](https://arxiv.org/html/2609.02143).

**QPS, latency, threads**
- Single-thread QPS is the shared standard (ann-benchmarks, VIBE, PAG). Azizi et al. process queries sequentially to mimic unpredictable arrivals.
- VIBE's per-query paired latencies give a route to significance testing — [VIBE](https://arxiv.org/html/2505.17810).
- QPS reflects CPU, memory bandwidth, SIMD, and parallelism; distance computations isolate algorithmic cost — [Dimitropoulos and Mamoulis](https://arxiv.org/html/2606.04522). The two do not always align, since each quantized comparison costs differently.

**Memory and build**
- PAG reports indexing time, peak build memory, search memory, and a breakdown; Azizi et al. report indexing time and memory footprint.
- VIBE warns that build times under throughput-tuned parameters may vary with parameters — [VIBE](https://arxiv.org/html/2505.17810).
- Build-thread policy varies: VIBE single core, PAG all 112 threads.

### Inferences
- Minimum figure set: QPS-recall at k=10 and k=100; distance computations versus recall; a table of p1, p5, and Robustness-0.5@k at a fixed operating point; build time on one thread and on all threads; peak RSS and on-disk size.
- Add 1/Ratio@k as a secondary column. It costs nothing, because ground-truth distances are available.
- With both 95% and 99% recall reported, GIST at high recall becomes a parameter question: M=16 may not get there at all.

### Gaps
- No primary source gives a standard latency percentile practice for single-thread library evaluation.
- Recall What Matters is a v1 preprint with no venue found; do not cite it as settled.

## Known pitfalls: tuning, unequal parameters, flags, versions, huge pages, one CPU, implementation versus algorithm

### Takeaway
The paper you hoped for exists in classic form: Kriegel, Schubert, and Zimek (2017). The modern evidence is scattered: ParlayANN, Azizi et al., Bang for the Buck, PLASMA, Coleman et al. I found no 2025 to 2026 paper dedicated to quantifying HNSW implementation effects across libraries. That is a real gap, and possibly an opening for this project.

### Cited Findings
**Implementation versus algorithm**
- Kriegel et al., KAIS 52(2), 2017, "The (black) art of runtime evaluation: Are we comparing algorithms or implementations?" [SS; publisher blocked; Semantic Scholar has the record with no abstract]. Described as arguing every new algorithm needs efficiency evaluation with several pitfalls — [record](https://www.datalearner.com/academic/journal-papers/0219-1377/volumes-and-issues/52/paper-detail/16814). **Its contents and numbers are unread. Get the PDF via library access before citing specifics.**
- VIBE invokes that paper to caveat its own OOD comparison — [VIBE](https://arxiv.org/html/2505.17810) [FT].
- Azizi et al. has a section "Implementation Impact": ParlayANN's Vamana_Opt and HNSW_Opt beat originals below 0.97 recall on Deep1B "due to more efficient data structures", while at higher recall "this advantage diminishes as distance computations dominate" — [Azizi et al.](https://arxiv.org/html/2502.05575v2) [FT]. HCNNG_Opt scaled to 1B where the unoptimized version could not pass 25 GB.
- ParlayANN puts four algorithms in one C++ framework, with deterministic parallel builds — [Manohar et al.](https://arxiv.org/html/2305.04359) [FT, parts].
- Graph reordering alone improved query time by up to 40%, and on SIFT100M reduced L1, L2, L3, and TLB misses — [Coleman et al.](https://arxiv.org/pdf/2104.03221) [SS].
- PLASMA (VecDB@VLDB 2026) isolates topology from memory layout on GPU; vertex reordering alone gave up to 80% QPS at equal recall — [PLASMA](https://arxiv.org/pdf/2508.15436) [SS].
- A blog measured Faiss searching at about 1.72x hnswlib's time, attributed to AVX-512 use — [Liu](https://terencezl.github.io/blog/2022/09/28/kids-use-hnswlib/) [SS, not peer reviewed]. A TechRxiv comparison exists — [link](https://www.techrxiv.org/users/922842/articles/1311476-a-comparative-study-of-hnsw-implementations-for-scalable-approximate-nearest-neighbor-search) [SS, preprint of unknown quality].
- Insertion order changed recall by up to 5.6 points in hnswlib and 12.8 in Faiss (from a search summary of [Elliott and Clark](https://arxiv.org/html/2405.17813v1); I did not confirm the figures in the text). Fix and vary seeds and order.

**Baseline bugs**
- PAG reports SymphonyQG gives very low recall at d in {1536, 3072} "due to the use of unsigned 16-bit integers for parallelism in its source code, causing an overflow" — [PAG](https://arxiv.org/html/2603.06660) [FT]. A competitor's result can be a bug, and silent. Note this conflicts in spirit with VIBE ranking it top, though VIBE's in-distribution set peaks at 2048-d and mostly under 1024.

**Hardware and flags**
- Kuffo and Boncz: CPU choice can change queries per dollar about 3x; partition indexes on Zen 4 reach almost 3x QPS, and outcomes depend on data-access patterns and vector size, not SIMD alone — [Bang for the Buck](https://arxiv.org/html/2505.07621) [FT, parts], DaMoN 2025.
- Our source check: Glass's allocator calls `madvise(p, len, MADV_HUGEPAGE)` for allocations with alignment of 2 MB or more (`glass/memory.hpp`, [pyglass](https://github.com/zilliztech/pyglass), HEAD 2025-09-09). Results depend on the host's THP setting.
- General THP guidance: about 10% best case, 0 to 3% more typically, with occasional harm — [Elhage](https://blog.nelhage.com/post/transparent-hugepages/) [SS, a blog]. No vector-search-specific measurement was found.

**Tuning**
- VIBE's unified grids exist to avoid dataset-specific tuning; it warns many methods struggle with common settings on some workloads — [VIBE](https://arxiv.org/html/2505.17810) [FT].
- Unequal baselines: PAG chose HNSW with M=32, efC=1024 — [PAG](https://arxiv.org/html/2603.06660) [FT]. Power Law used M=32 and efc=200 for its examples.
- hnswlib's source comments that high intrinsic dimension needs M of 48 to 64 — reported in [Elliott and Clark](https://arxiv.org/html/2405.17813v1) [FT].
- Fixed-M versus fixed-efc scaling: larger M and efc lower the scaling exponent — [Power Law](https://arxiv.org/html/2609.02143) [FT].

**Hard-to-see problems**
- Average recall hides fragility — [Wang et al.](https://arxiv.org/html/2507.00379) [FT].
- A mismatched metric can yield 99.9% synthetic recall and under 1% label recall (ImageNet, inner product) — [Chen et al.](https://arxiv.org/html/2512.12980) [FT].
- Dataset duplicates: the Power Law paper de-duplicates a dataset that "contains many duplicate points" — [Power Law](https://arxiv.org/html/2609.02143) [FT]. The footnote's dataset was not identified in my extract; check whether it is GloVe.

### Inferences
- Tuning on test queries has an accepted and an unaccepted form. Choosing the frontier from a fixed grid over test queries is standard (ann-benchmarks, VIBE). Iterating the engine's design against the benchmark queries is not covered by that convention, and I would not defend it.
- For an engineering-led paper, implementation is the thesis, so it must be measured rather than controlled away. The strongest design loads one identical graph into each search path, which separates graph quality from search-loop engineering. PLASMA's separation of topology from layout is a precedent.

### Gaps
- No primary source measures huge pages on HNSW. A small experiment (THP on, off, and madvise) would be novel and cheap.
- No source was found quantifying compiler flags across ANN libraries.

## Reproducibility expectations and likely reviewer objections

### Takeaway
PVLDB Volume 20 requires a reproducibility package at first submission, and EA&B papers get no excuses. SIGMOD encourages artifacts at submission, with ARI after acceptance. No published collection of reviewer objections exists, so the objection list below is my inference.

### Cited Findings
**PVLDB**
- "For PVLDB volume 20, already the initial submission must include a link to the full reproducibility package of all experiments, data, and artifacts with meaningful instructions on how to run the experiments" — [guidelines](https://vldb.org/2027/submission-guidelines.html) [OFFICIAL].
- A reasonable explanation excuses a regular paper, but "Experiment, Analysis & Benchmark (EA&B) papers are required to make available all experimental data and related software (there are no excuses)" — [guidelines](https://vldb.org/2027/submission-guidelines.html).
- The post-acceptance process wants source or a fully specified black box, data or its generator, experiment scripts, and figure scripts. Reviewers check overall patterns, since exact numbers depend on hardware — [PVLDB reproducibility](https://vldb.org/pvldb/reproducibility/) [OFFICIAL].

**SIGMOD**
- Sharing is encouraged, in anonymous repositories, but not mandatory; explain if unable. E&A papers must compare existing solutions or provide benchmark data, with public release expected on acceptance — [SIGMOD 2027 call](https://2027.sigmod.org/calls_papers_sigmod_research.shtml) [OFFICIAL].
- ARI awards Artifacts Available, Artifacts Evaluated - Reusable, and Results Reproduced. Authors supply system, input data, experiment scripts, and figure scripts. Docker, ReproZip, CloudLab, and others are suggested — [ARI](https://reproducibility.sigmod.org/) [OFFICIAL].

**Practice to copy**
- VIBE ships Apptainer definitions and configs per method, plus datasets, and supports HPC with NUMA-aware execution — [VIBE](https://arxiv.org/html/2505.17810) [FT]. That suits PACE, where Docker is a poor fit.

### Inferences: likely objections to "new library versus Glass and hnswlib" (my reasoning, not sourced reviewer text)
1. "You tuned your own system and held others at one setting." Answer: a grid and Pareto frontier for each.
2. "Glass and hnswlib are not the state of the art." VIBE names SymphonyQG, NGT-QG, LoRANN, ScaNN, and IVF-PQ among the leaders; PAG adds RaBitQ+ and Vamana.
3. "Your speedup is implementation, not a new idea." For a research-track paper this needs an explicit, publishable novelty, or use the EA&B route.
4. "Your legacy datasets flatter SQ8." Answer: modern and OOD data.
5. "It only holds on Intel with AVX-512." Our three-CPU plan answers this well.
6. "You report averages." Answer: tails.
7. "No statistics." Answer: paired tests.
8. "It does not scale." Answer: a size series.
9. "Quantized graph comparison ignores memory." Answer: report memory.
10. "Glass's tuned parameters or `optimize` step were misused." Answer: follow its documented guidance and show it. The repo says "Use SQ8U for indexing, and SQ4U for searching is almost always a good choice" — [pyglass](https://github.com/zilliztech/pyglass) [OFFICIAL]. Our harness's Glass configuration should be checked against that advice.

### Gaps
- Reviewer reports are not public. No source gives acceptance rates for such papers.

## What do recent scalability results imply for testing at 1M versus 100M?

### Takeaway
A single-size evaluation is now a named weakness. Build a size series at fixed recall. Our DEEP10M run is the natural base for it, and 1M to 100M (subsets of a billion-point set) is what the Power Law paper used.

### Cited Findings
All from [Maghrebi, Eslami, Dayan, arXiv 2609.02143](https://arxiv.org/html/2609.02143), 2 Sep 2026, University of Toronto [FT, major sections]. Venue not stated in what I read.
- "standard benchmarks measure search cost at a single dataset size, not across sizes".
- At fixed recall, distance computations follow dc proportional to N^c, 0<c<1, on all eight datasets, mostly to full size, across recall 90 to 99%, hard and easy queries, and sixteen configurations per index.
- Method: uniform random subsets, each shuffled before build; canonical queries if present, else 10,000 held out; exact ground truth by scan at each size; the smallest ef_search reaching the recall target.
- SIFT, DEEP, SpaceV capped at 100M for the grid (about a week of runtime), then one configuration to 1B: SIFT and DEEP stay straight; SpaceV bends. OpenAI drifts slightly from the power law at its full size.
- The fitted exponent rises with recall target and intrinsic dimensionality, and falls with M and efc. k does not matter. Exponents appear as labels in figures, so I could not extract numbers.
- Exponent model: additive log terms in 1/delta, efc, and M; average R-squared about 0.88. dc versus ef_search is also a power law, R-squared above 0.98.
- At least 80% of cost is the exploration phase across all benchmarks.
- Fixed ef with growing N lowers recall, which is why recall must be held fixed.

Related:
- Azizi et al.: leaders change between 1M and 25 GB, with SSG, NSG, NGT, HCNNG dropping and ELPIS leading — [Azizi et al.](https://arxiv.org/html/2502.05575v2) [FT].
- VIBE concentrates on about 1M so one grid can be used, adds two larger sets, and lists billion-scale as future work — [VIBE](https://arxiv.org/html/2505.17810) [FT].
- Data mimicking to extend benchmark scale appeared at VecDB and in cuVS-bench — [NVIDIA docs](https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide/cu-vs-bench-tool/synthetic-dataset-generation) [SS].

### Inferences
- Use a modest ladder: 1M, 3M, 10M, 30M, 100M on DEEP, plus one modern dataset. Report c at 95% and 99% for each library. This is cheap relative to its value and can be built by re-using scripts.
- Distance computations let us fit c for libraries without hardware noise, but Glass, VSAG, and others need instrumentation, or use QPS-inverse as a proxy, labeled clearly.
- Memory limits the ladder. Our Gold 6226 node has 187 GB RAM, so 100M at 96-d (about 38 GB of raw float32) fits, while 100M at 1536-d (about 614 GB) does not.
- A rank flip between 1M and 100M would be a publishable finding in itself.

### Gaps
- A 1536-d or larger dataset above 10M at this scale is limited: DataCompDr, MajorTOM, and Cohere Wikipedia are the obvious candidates.

## Current venue facts (official pages, checked 2026-11-02)

### Takeaway
PVLDB Volume 20's rolling deadlines are the only confirmed live research route, closing 1 Mar 2027. SIGMOD 2027 research rounds are closed. No Volume 21 or SIGMOD 2028 dates are published.

### Cited Findings
**PVLDB / VLDB**
- Volume 20: deadlines on the 1st of each month at 5:00 PM Pacific, abstract by the 25th of the previous month. First deadline 1 Apr 2026, final 1 Mar 2027 — [guidelines](https://vldb.org/2027/submission-guidelines.html) [OFFICIAL].
- Twelve pages excluding references for Regular and EA&B; single-blind with names on the first page; two per author monthly and twelve yearly; initial reviews usually by the 15th of the next month after the deadline; up to 2.5 months for revisions — [Volume 20 submission](https://www.vldb.org/pvldb/volumes/20/submission) [OFFICIAL].
- 1 Jun is the final revision deadline for VLDB 2027; later rolls to VLDB 2028 — [guidelines](https://vldb.org/2027/submission-guidelines.html).
- VLDB 2027: Athens, 23 to 27 Aug 2027 — [call](https://vldb.org/2027/call-for-research-track.html). VLDB 2028: Marrakech, 28 Aug to 1 Sep 2028 — [site](https://vldb.org/2028/).
- Volume 20 Editors-in-Chief: Katja Hose and Matthias Boehm — [Volume 20](https://vldb.org/pvldb/volumes/20/).
- Volume 21: no deadlines found on [the /volumes/21/ URL](https://vldb.org/pvldb/volumes/21/), which shows generic content listing archives to 20, or on the VLDB 2028 site.

**SIGMOD**
- SIGMOD 2027 research rounds: abstract/paper Jan 10/17, Apr 10/17, Jul 10/17, Oct 10/17, 2026. Final decisions: 12 Jun, 12 Sep, 12 Dec 2026, 12 Mar 2027. All closed — [important dates](https://2027.sigmodconf.hosting.acm.org/calls_papers_important_dates.shtml) [OFFICIAL].
- 12 pages excluding references, double-anonymous, 10 submissions per author, optional appendix, four rounds — [call](https://2027.sigmod.org/calls_papers_sigmod_research.shtml).
- Still open: Industrial, paper 24 Nov 2026; Demonstration, 11 Jan 2027 — [important dates](https://2027.sigmodconf.hosting.acm.org/calls_papers_important_dates.shtml). Conference: 13 to 18 Jun 2027, Huntington Beach.
- SIGMOD 2028: [official site](https://2028.sigmod.org/) is a placeholder. Seoul appears only in a search summary [SS].
- Conflict: one search summary gave a Round 1 of 15/16 Jan 2027 with 8 Mar 2027 notification, attributed to SIGMOD 2027. It contradicts the official page; it may be a different track or year. Ignore unless SIGMOD 2028 confirms it.

**Workshops**
- VecDB, 2nd edition at VLDB 2026, 4 Sep 2026; chairs Matsui, Aumüller, Manohar, Simhadri — [VLDB 2026 workshops](https://vldb.org/2026/workshops.html). Submissions closed 19 Jun 2026 (extended from 12 Jun); it also took dataset announcements — [site](https://vecdb-ws.github.io/vldb2026/index.html). The first edition was at ICML 2025, and no 2027 edition is announced.
- SISAP 2026: Brno, 14 to 16 Oct 2026; challenge submissions in June — [challenge](https://sisap-challenges.github.io/2026/index.html). SISAP 2027 not checked.
- NeurIPS 2026 workshops, 11 to 13 Dec, Sydney, Paris, Atlanta: no vector-search-specific workshop listed; Machine Learning for Systems is the closest, in Sydney; the suggested submission date was 29 Aug 2026 — [announcement](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/) [OFFICIAL].
- DaMoN hosted the hardware paper in 2025 — [Kuffo and Boncz](https://arxiv.org/html/2505.07621).

### Inferences
- Next Volume 20 deadlines by the stated rule: 1 Dec 2026 (abstract 25 Nov), 1 Jan, 1 Feb, 1 Mar 2027.
- With a Round 5 and a re-run ahead, 1 Mar 2027 is realistic only if the redesign starts now. A rejected-without-revision outcome would still leave Volume 21.
- By pattern alone: Volume 20 opened on 1 Apr 2026, so Volume 21 may open around Apr 2027, and SIGMOD 2027's rounds began 17 months ahead of the conference, so SIGMOD 2028 Round 1 could fall around Jan 2027. **Neither date is announced. Check [2028.sigmod.org](https://2028.sigmod.org/) monthly.**
- Double-anonymous SIGMOD conflicts with a public site at epzero.dev under the same name. Check the anonymity rule on pre-existing systems before submitting; PVLDB's single-blind avoids this.
- EA&B has "no excuses" on artifacts, which we can satisfy with a public repo. That sits awkwardly with the memory note that the demo site must not link the repo; those are separate surfaces, so a package link in the paper does not touch the site.

### Gaps
- SIGMOD 2027 workshop call URL returned 404; no workshop list was obtained. VLDB 2027 workshop list not found. VecDB 2027 not announced. DaMoN 2027 not checked.
- The [VLDB 2027 dates page](https://vldb.org/2027/important-dates.html) extraction gave "20th of every month" wording that disagrees with the guidelines. I trust the guidelines page; read the original before relying on either.

## Our setup: what would and would not survive review

### Takeaway
The measurement hygiene is good, in places better than VIBE. The experimental design is the weak part: fixed graph parameters only, three legacy of four datasets, no OOD, no scale series, no memory figures, unpinned versions, and five rounds of variant selection against the same queries. All are fixable before a 1 Mar 2027 target if rounds are spent there first. Findings below come from `bench/compete.py`, `scripts/pace/compete.sbatch`, `scripts/pace/compete-env.sh`, `scripts/pace/round5.sh`, `bench/results/pace/gold6226/hardware.txt`, and result JSON.

### Cited Findings
**Survives review**

| Practice | Why it holds |
|---|---|
| One search thread, `numactl --cpunodebind=0 --membind=0` | Same as [VIBE](https://arxiv.org/html/2505.17810); [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) |
| Exclusive node; 1 thread per core | Matches VIBE's SMT-off; [Azizi et al.](https://arxiv.org/html/2502.05575v2) used exclusive access |
| Interleaved rounds with rotating library order | I did not see this drift control in the VIBE or Azizi et al. sections I read; it is an asset, so say so in the paper |
| Median QPS with raw `qps_samples` kept | Defensible, but see below on count |
| 10,000 queries where available | Above VIBE's 1000 and Azizi et al.'s 100 |
| Three CPUs (Gold 6226, EPYC 7513, Xeon 6740P) | Answers VIBE's stated limitation and [Bang for the Buck](https://arxiv.org/html/2505.07621). A real differentiator |
| Per-node compilation of `-march=native` libraries | Good, once recorded per library |
| Distance kept in L2 on normalized vectors | Ranking-preserving as documented in `compete.py` |

**Will not survive, ranked by risk**

| # | Issue found | Evidence | Fix |
|---|---|---|---|
| 1 | Design iteration on the test set. `-speed-round2` to `round5` compare Epzero variants against Glass on the same query file, and recall is measured on the timed queries | Goes beyond the fixed-grid frontier that [VIBE](https://arxiv.org/html/2505.17810) sanctions | Split queries now into dev and final. Freeze the engine at a tagged commit. Run final once. Disclose the history |
| 2 | Only M=16, efc=200 | ann-benchmarks principle and VIBE use grids and frontiers; [PAG](https://arxiv.org/html/2603.06660) uses M=32, efC=1024; hnswlib's own note wants M 48 to 64 for hard data; [Recall What Matters](https://arxiv.org/html/2606.04522) shows HNSW can fail on GIST | Unified grid for every library, for example M in {8,16,32,48} and efc in {100,200,500}, Pareto frontier. Keep M=16 as a separately labeled "fixed graph budget" experiment. It is a legitimate control, but not a best-versus-best claim |
| 3 | Three of four datasets are legacy; DBpedia is the only modern one. Ours is text-embedding-ada-002 at 1536-d (`scripts/make_dbpedia.py`, from KShivendu/dbpedia-entities-openai-1M), an older model than the text-embedding-3-large DBpedia sets used by PAG, so name the model in the paper | [VIBE](https://arxiv.org/html/2505.17810), [Iceberg](https://arxiv.org/html/2512.12980) | Add about 3 to 5 VIBE datasets; keep GIST and GloVe labeled legacy. DEEP stays as the scale ladder only |
| 4 | No OOD; DBpedia's 10,000 queries come from random split of same corpus (`make-dbpedia.sbatch`). All in-distribution | VIBE's named quantizer mechanism strikes SQ8 directly | LAION-CLIP or Yandex, and HotpotQA-Harrier, at minimum |
| 5 | No size series | [Power Law](https://arxiv.org/html/2609.02143) | DEEP ladder to 100M on one CPU, with shuffled uniform subsets and exact ground truth at each size |
| 6 | No memory in `compete.py` output (no RSS or index size field) | [PAG](https://arxiv.org/html/2603.06660), [Azizi et al.](https://arxiv.org/html/2502.05575v2) | Record peak RSS, on-disk size, and search RSS. SQ8 stores codes plus vectors for rerank, so this is the likely comparison with Glass, and a place where a reviewer will look |
| 7 | Unpinned software. `compete-env.sh` clones pyglass with `--depth 1` and installs usearch and pyvsag unpinned; result JSON has no versions, flags, commit, or THP state; the worktree has an uncommitted change to `engine/src/hnsw.cpp` | [PVLDB reproducibility](https://vldb.org/pvldb/reproducibility/); [Glass's source](https://github.com/zilliztech/pyglass) | Pin by hash. Write `environment` into each JSON: versions, commit, compiler, flags, kernel, THP, microcode. Put every result on a clean tagged commit |
| 8 | Compiler asymmetry: hnswlib and Glass native-built, Faiss, USearch, VSAG from wheels, Epzero by runtime dispatch; each may hit a different SIMD path | [Bang for the Buck](https://arxiv.org/html/2505.07621); [Kriegel et al.](https://www.datalearner.com/academic/journal-papers/0219-1377/volumes-and-issues/52/paper-detail/16814) [SS] | Build everything from source with one compiler and flags, or show wheels are within noise of source builds and report which ISA ran |
| 9 | Huge pages: Epzero (`engine/src/hnsw.cpp:94`) and Glass call `madvise(MADV_HUGEPAGE)`; other libraries unverified; THP not logged | Verified in code | Log `/sys/kernel/mm/transparent_hugepage/enabled`. Run an on/off check. If it matters, enable for all (or report as a factor) |
| 10 | Rounds = 3 | VIBE 5, Azizi et al. 6 | At least 5. Per-query latencies for one configuration per library at 95% recall, then Wilcoxon with Holm-Bonferroni |
| 11 | k=10 only as default | [VIBE](https://arxiv.org/html/2505.17810), [PAG](https://arxiv.org/html/2603.06660) | k=10 and k=100 at least. SQ8 rerank depth is k-sensitive |
| 12 | Mean recall only | [Wang et al.](https://arxiv.org/html/2507.00379), [RCheck](https://arxiv.org/html/2608.25185) | p1, p5, Robustness; hard-versus-easy by LID or RC |
| 13 | Binding is to a socket, not a core | VIBE's text says one physical core and socket-bound numactl, so its exact binding is unclear; our thread may migrate among 12 cores | Bind one core with `--physcpubind`, on a socket with no other jobs |
| 14 | "Eight libraries" overstates it. `compete.py` lists eight configurations from six projects: Epzero (two), hnswlib, Faiss (two), USearch, VSAG, Glass. Only five are outside projects | `compete.py` docstring; [VIBE](https://arxiv.org/html/2505.17810) top list | Say exact counts. Add SymphonyQG and NGT-QG, ScaNN or LoRANN, and ParlayANN or DiskANN Vamana. Test SymphonyQG's 16-bit overflow at 1536-d before trusting it |
| 15 | VSAG measured per query through Python; others batched | `compete.py` notes it | Measure in C++ or drop from QPS ranks |
| 16 | Build timed on all 24 cores only | VIBE single-core, PAG all threads | Report both: lock contention varies |
| 17 | Turbo and frequency: `lscpu` max 2700 MHz, same as nominal, suggests turbo off, but nothing logs it, and the other two CPUs may differ | Absent from scripts (grep of `scripts/pace`) | Log governor and turbo on all three |
| 18 | Single build per library | Build and insertion randomness: order effects in [Elliott and Clark](https://arxiv.org/html/2405.17813v1) [SS for the figure] | Three seeds and insertion orders for each build; report spread of recall at fixed ef |
| 19 | Nothing isolates graph from search | Inference from [Azizi et al.](https://arxiv.org/html/2502.05575v2) and PLASMA | Export one graph, load it into each search path; report distance computations too. This is the core of any implementation claim |

### Inferences
**Order of work**
1. Stop variant search on current queries. Freeze a release candidate.
2. Fix harness: pins, environment block, memory, per-query latency, k=100, dev/final split, core binding.
3. Add VIBE datasets and OOD through the existing ann-benchmarks HDF5 loader.
4. Run the grid, then the size ladder, then all three CPUs.
5. Add the implementation ablation (variants already exist in the `*-variants` runs) and the shared-graph test.
6. Package an Apptainer artifact, following VIBE.

**Positioning.** If the novelty is mainly engineering, EA&B may be stronger than Regular Research: "How much of graph ANN speed is implementation?" across three CPUs answers a gap the literature leaves open and that VIBE explicitly flags. That route requires full artifacts, which the project can meet. A Regular Research paper needs a distinct idea that the ablation shows matters.

**Claims to avoid.** "Fastest library" from one configuration; any result on legacy data alone; a percentage speedup without stating recall, k, dataset, and CPU; "log-scale" scaling.

**Source ledger**

| Paper | Label |
|---|---|
| VIBE (2505.17810 v3) | FT: setup, evaluation, OOD, conclusions; dataset rows partly from extraction |
| Power Law (2609.02143) | FT: sections 1 to 3 and 5, theory skimmed by extraction |
| Recall What Matters (2606.04522) | FT: results and conclusions |
| Azizi et al. SIGMOD 2025 (2502.05575) | FT: setup, implementation impact, recommendations |
| PAG (2603.06660) | FT: section 5 and appendix setup |
| Wang et al., Robustness (2507.00379) | FT: abstract, targeted passages |
| RCheck (2608.25185) | FT: abstract, intro |
| Iceberg, Chen et al. (2512.12980) | FT: abstract, intro, targeted passages |
| Semantic Recall (2604.20417) | FT downloaded; abstract read |
| Big-ANN NeurIPS'23 (2409.17424) | FT: setup, metrics |
| ParlayANN (2305.04359) | FT: OOD sentence, parameter notes |
| Bang for the Buck (2505.07621) | FT: intro, targeted passages |
| Elliott and Clark (2405.17813) | FT: abstract, related work; 5.6 and 12.8 figures from SS |
| Filtered benchmarks (2606.14193) | Downloaded, not read; SS |
| Revisiting Index Construction (2410.01231) | ABS |
| EDBT 2026 tutorial | SS; 503 on both PDFs |
| Kriegel et al. KAIS 2017 | SS; paywalled |
| Aumüller and Ceccarello | SS |
| PLASMA (2508.15436), Coleman et al. (2104.03221), data mimicking | SS |
| Disk-Resident Graph ANN (2603.01779) | Downloaded, not read; out of scope |

### Gaps
- Kriegel et al. should be obtained and read before the paper cites it for anything specific.
- I did not check whether hnswlib, Faiss, USearch, or VSAG use huge pages, nor their per-build SIMD paths.
- GIST's held-out query count limits it (1000 in ann-benchmarks), and I did not check how many dataset files reuse test queries across rounds.
- PACE's THP setting and whether 100M is feasible in its scratch quota remain unmeasured.
- Number of Epzero variants tried in rounds 1 to 5 should be counted, since an honest multiple-comparisons disclosure needs it.
