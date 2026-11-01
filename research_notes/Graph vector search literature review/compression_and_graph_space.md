# Compression and dimensionality reduction in graph ANN: build space versus walk space

Coverage: primary sources through early November 2026. Today is 2026-11-02.

**Reading labels.** The brief asks for "read in full / abstract only / search summary only". No paper here was read line by line, so the top label is narrower than "read in full":
- **[FT]**: full text retrieved as plain text; the relevant sections were read through keyword extraction plus full passages. Quotes come from that raw text. Unread sections may exist. Figures were not seen unless stated, so claims about them rest on captions and body text.
- **[FT+FIG]**: as FT, and the figure itself was viewed (only LeanVec Fig. 14).
- **[ABS]**: abstract or landing-level text only.
- **[SS]**: search-engine summary only. Treat as a lead, not a verified fact.
- **[SRC]**: source code or project documentation read directly.
- **[EPZERO]**: this project's own log, `docs/performance.md`, and source. Not independently verified.

## Which space does each system build in and walk in, and how much does it re-rank?

### Takeaway
Walking on compact codes and re-ranking on full vectors is standard. Building on the same codes is established too: LVQ, LeanVec, Flash, QuIVer and Lucene BBQ all do it. Only a few systems report any evidence about graph quality from that choice. PAG and ADSampling differ in kind, because they keep exact vectors and use projection to skip or shorten distance calculations.

### Cited Findings

| System | Compact code | Graph built on | Walk on | Re-rank |
|---|---|---|---|---|
| LVQ / SVS | LVQ-8, 4x4, 4x8; per-vector scaling then uniform SQ | Can be built on first-level LVQ; streaming work does so by default | First-level LVQ | Second level, optional |
| LeanVec | Primary: LVQ on d-dim projection. Secondary: LVQ8 or FP16 at full D | "Only the primary vectors are used for graph construction", but the headline search results used an FP16 graph | Primary | Example: 50 candidates for recall@10 |
| Flash | PCA, then PQ-style 4-bit codes (K=16), distance tables rounded to 8 bits (H=8) | Codes, using ADT for candidate search and SDT for neighbor selection | CA-style search on codes | Reranking on original vectors; count not found |
| PAG | Random-projection data per edge; no walk code in the QG sense | Original vectors, with exact distances skipped by a test; "first to apply PRT to graph construction" | Original vectors, exact only if the test passes | None needed |
| SymphonyQG | RaBitQ, 1 bit per dim, FastScan batches of 32 | Iterative: candidates from its FastScan search, then NSG pruning, then edge supplementation | Codes for neighbors, exact at each visited vertex | Implicit, no explicit stage |
| RaBitQ | D-bit string | Not graph based; used with IVF; graphs left as future work | Not applicable | Chosen by error bound, not a fixed count |
| ADSampling | None; random orthogonal rotation plus progressive dimension sampling | Standard HNSW construction | Partial dimensions on rotated vectors | Exact results set; HNSW++ decouples two roles |
| ScaNN (SOAR text) | PQ with 16 centers per subspace | Not graph based (VQ partitions) | PQ | Highest-bitrate form, int8 or float32 |
| Glass | SQ8U, SQ8, SQ4U, PQ8 and others | Selectable; bindings default FP32 | Selectable, for example SQ8U | Refiner: `reorder_k = k * reorder_mul`; Epzero's log says 15 |
| VSAG | SQ, for example SQ4 baseline; INT8/INT4 | **Not established from text read** | Low-precision vectors | Selective, "logarithmically scaled candidate subset" |
| QuIVer | 2-bit sign-magnitude, no training | Those codes, with Vamana pruning on them | Same codes, symmetric | float32 on `ef` candidates, e.g. 64 to 256 |

- **LeanVec numbers.** At 10-recall@10 of 0.90, on an FP16-built graph, "up to 10.2x and 3.7x performance gains over FP16 and LVQ" on gist-960-1M; on rqa-768-1M "outperforming LeanVec-ID and LVQ by up to 1.5x and 2.8x"; "does not show significant gains in laion-512-1M"; builds "up to 8.6x and 4.9x faster than FP16 and LVQ" — [LeanVec, FT](https://arxiv.org/html/2312.16335). Selected d: gist-960-1M 160, deep-256-1M 96, open-images-512 160, t2i-200 192, wit-512-1M 256, laion-512-1M 320, rqa-768 160 — same source.
- **LeanVec re-rank.** "we can obtain 50 candidates, recompute their distance using secondary vectors" — [LeanVec, FT](https://arxiv.org/html/2312.16335). Graph settings R=128, L=200, α=1.2 (L2) or 0.95 (IP) — same source.
- **LVQ.** "LVQ enables building graph indices directly from compressed vectors"; memory reduction up to 6.2x; LVQ-8 gives "5.2x more QPS at 0.9 recall" over PQ at similar compression on deep-96-100M — [LVQ, FT](https://arxiv.org/html/2304.04759). Streaming work builds "directly from first-level LVQ encodings" — [SVS streaming, FT](https://arxiv.org/html/2402.02044). SVS docs: "Graph searches are done over this primary dataset" — [SVS docs, SS via fetch summary](https://intel.github.io/ScalableVectorSearch/python/experimental/leanvec.html).
- **Flash.** Build speedups "10.4× to 22.9×"; HNSW-PCA and HNSW-SQ "less than 2× speedup on most datasets"; "we apply an additional reranking step based on original vectors" — [Flash, FT](https://arxiv.org/html/2502.18113). Settings R=32, C=1024, B=16 — same source. Its sensitivity study uses dF=64 and MF=16 — same source.
- **PAG.** "up to 5× faster than HNSW" on modern datasets; PAG-Base "20–40% of the indexing time of HNSW" — [PAG, FT](https://arxiv.org/html/2603.06660). DBpedia1536 is among its datasets. It reports SymQG "very low recall" on 1536-d and 3072-d data, attributing this to 16-bit integer overflow in the SymQG source — same source. That is a code defect claim, not evidence against 1-bit walks.
- **SymphonyQG.** "At 95% recall (K=10), SymphonyQG achieves 1.5x-4.5x QPS compared with the most competitive baselines and achieves 3.5x-17x QPS compared with the classical library HNSWlib"; at least 8x faster index build than NGT-QG — [SymphonyQG, FT](https://arxiv.org/html/2411.12229).
- **RaBitQ.** "We leave it as future work to apply our quantization method in graph-based methods"; re-ranking decided "based on the error bound without tuning hyper-parameters" — [RaBitQ, FT](https://arxiv.org/html/2405.12497).
- **ADSampling.** "up to 2.65x speed-up on HNSW and 5.58x on IVF"; saves "up to 75.3%" of evaluated dimensions for HNSW; accuracy loss on GIST no more than 0.14% — [ADSampling, FT](https://arxiv.org/html/2303.09855). Flash confirms these methods "retain the standard HNSW construction process" — [Flash, FT](https://arxiv.org/html/2502.18113).
- **ScaNN family.** PQ duplicates are cheap next to the "highest-bitrate representation (usually int8 or float32)"; 32,608 QPS on Microsoft Turing-ANNS and 46,712 QPS on Microsoft SPACEV — [SOAR, FT](https://arxiv.org/html/2404.00774). The original ScaNN paper was not read. Google's low-precision paper reports int8 can "reduce memory by 60% and improve query throughput by 30%, while incurring only a 2% reduction in recall" — [Low-Precision Quantization, FT](https://arxiv.org/html/2110.08919).
- **Glass.** `Index` binding default `quant="FP32"`; `create_hnsw` also accepts quantized builders such as `SQ8QuantizerUniform`; HNSW builds through `quant.get_sym_computer()` — [pyglass source, SRC](https://github.com/zilliztech/pyglass). Glass has no paper that I found. Epzero's `compete.py` calls `Index("HNSW","L2",R=...,L=...)` with no `quant`, then walks SQ8U and refines FP32, so its Glass graphs take the installed version's default build — [EPZERO]. See the last section's gaps.
- **VSAG.** 2,167 QPS vs 511.9 on GIST1M; L3 miss rate 93.89% to 39.23% — [VSAG, FT](https://arxiv.org/html/2503.17911). The recall target of 90% and "built on full-precision vectors" came only from a fetch summary **[SS]**. The paper's stated baseline is "HNSW with SQ4 quantization", so the multiple's reference should be checked before citing.
- **QuIVer.** ≥91% Recall@10 at 16–39K QPS on MiniLM-1M, Cohere-1M and DBpedia-OpenAI-1M, "outperforming hnswlib by ∼16× and USearch HNSW by ∼5×" — [QuIVer, FT](https://arxiv.org/html/2605.02171v1).

### Inferences
- Two different designs get called "build on compressed": build and walk both on codes (LVQ, Flash, QuIVer), or build on codes with a separate full graph as the evaluated baseline (LeanVec). A paper must say which it means.
- LeanVec's d values (gist 160) and Epzero's best d (96 to 192 on GIST1M) are in the same range, which supports treating the walk result as a reproduction on HNSW.

### Gaps
- VSAG's build space and exact selective re-rank rule were not confirmed from raw text.
- Flash's re-rank count was not found in the passages read.
- SymphonyQG's pruning inputs: its querying algorithm computes exact distances at visited vertices, so pruning probably sees exact distances, but I did not confirm this.

## Has anyone compared code-built and full-built graphs under the same walk, and did search ever improve?

### Takeaway
Yes, same-walk comparisons exist. LVQ Fig. 4 and LeanVec Fig. 14 both show essentially no change. No clean same-walk *improvement* was found. Flash reports "maintaining or improving" search, but no Flash experiment searches code-built and exact-built graphs with the same routine, so its gain cannot be assigned to the graph.

### Cited Findings
- **LVQ, same walk.** Figure 4 searches with float32 "to normalize for compute differences" across build bit widths on deep-96-100M: "almost no decrease in throughput in graphs built with 4 or more bits with LVQ", but "a sharp drop in throughput for graphs built using global quantization with 4 bits" — [LVQ, FT](https://arxiv.org/html/2304.04759).
- **LVQ mechanism.** Candidate-list recall against full precision is almost one at 8 and 16 bits, 0.82 for LVQ-4, and 0.6 for 4-bit global quantization; Proposition 1 bounds when pruning errors leave graphs "equally accurate" — [LVQ, FT](https://arxiv.org/html/2304.04759). This explanation is about preservation, not improvement.
- **LeanVec, same walk.** Fig. 14: "We observe no noticeable differences in the graphs built with and without dimensionality reduction." — [LeanVec, FT+FIG](https://arxiv.org/html/2312.16335). Body text: "without degrading their quality, i.e., their search accuracy and speed", then "The accuracy preservation is a surprising fact, as the graph, being related to the Delaunay graph, is heavily related to the local topology of the data. Further theoretical studies to understand this phenomenon are required." — same source.
- **Flash.** Flash uses PCA then PQ-style codes, builds on them, and reports "maintaining or improving search performance", with construction speedups of 10.4× to 22.9×. No experiment searches code-built and exact-built graphs with the same routine — [Flash, FT](https://arxiv.org/html/2502.18113); coordinator-verified. Its search is on codes plus re-rank, while baseline HNSW searches on full vectors, so walk and graph differ together.
- **Flash's reasons.** It credits fewer random memory accesses and better SIMD use. On dimension: "focusing on principal components is advisable when vector representation is constrained by limited bit allocation", because less significant components "suffer from poor bit usage". On Figure 16: quality rises then falls with dF, from "information loss at lower dimensions and redundancy at higher dimensions", and "While the search focuses on nearest neighbors, index construction requires navigable neighbors, even if more distant" — [Flash, FT](https://arxiv.org/html/2502.18113). The bit-allocation claim is about compressing with few bits, not about noise in the data.
- **Flash baselines.** "HNSW-PCA achieves the best performance on ARGILLA but the worst on LAION, indicating sensitivity to data distribution"; HNSW-PQ is "inferior to standard HNSW due to degraded index quality"; HNSW-SQ "shows consistent search performance gains on most datasets" — [Flash, FT](https://arxiv.org/html/2502.18113). The walk differs here too. Flash also reports improved search on NSG and τ-MG on LAION-10M, with the same confound.
- **QuIVer.** It asks whether binary quantization can "build the graph, instead of merely accelerating graph search". It has no float-built graph under the same walk in the passages read. Recall@10 at ef=64: GIST-960 1.00%, SIFT 5.68%, GloVe 54.74%, DBpedia 94.63%, Cohere 95.12% — [QuIVer, FT](https://arxiv.org/html/2605.02171v1). It claims "all existing systems treat quantization as a post-hoc optimization", which LVQ (2023), LeanVec and Flash contradict.
- **MonaVec, opposing view.** "FP32 build is necessary"; 4-bit build scores risk "corrupting the graph topology irreparably" — [MonaVec, FT](https://arxiv.org/html/2606.19458). This is argued from noise magnitudes in the text read, and LVQ-4 evidence runs against it as a general claim.
- **Bi-metric framework.** Theory: a graph built using only a cheap proxy metric d can answer to 1+ε under expensive D with sublinear D calls, for DiskANN and Cover Tree — [Bi-metric, FT](https://arxiv.org/html/2406.02891). On HotpotQA, baselines at 8,000 D calls match the method at under 2,000, ">4x fewer evaluations". On recall it beat single-metric "in most cases, except for FEVER, FiQA2018, and HotpotQA" — same source. Here d and D are different embedding models, and the walks differ.
- **IVF parallel.** MRQ's projected centroids incur "almost no recall loss on the GIST dataset and even slightly outperform the original centroids on the OpenAI-1536 dataset" — [MRQ, FT](https://arxiv.org/html/2411.06158). That is IVF, not a graph, and the dataset direction is the reverse of Epzero's.
- **Distance-preserving case.** A full-rotation PCA gives the same index, since "the vector index built from the transformed dataset will be the same as the original one" — [DR survey, FT](https://arxiv.org/html/2403.13491). Only truncation or quantization can change a graph.

### Inferences
- LeanVec's result is a null at throughput on a multi-threaded system with R=128. Epzero's is a signed effect at M=16-style degree (32 base links) on one thread. Differences in degree, thread count and index type could explain the disagreement, and none has been tested.
- Dataset dependence is the common signal: Flash's HNSW-PCA swings by dataset, QuIVer collapses on GIST while Epzero gains there, and MRQ's centroids favor OpenAI. No paper explains this pattern.

### Gaps
- I found no paper reporting a same-walk gain from a code-built graph. Absence from my search is not proof of absence.
- Flash's Figures 8, 13, 14 and 16 were not viewed.

## Has anyone crossed graph-build space with walk space in a factorial experiment?

### Takeaway
Yes, in part, so "first factorial" is **not a safe claim**. LeanVec Fig. 14 crosses three build spaces with two walks on four datasets, as an appendix figure with no numbers. I found no factorial with a full-vector walk cell, a plain-quantization build level, or a measured interaction effect.

### Cited Findings
- **LeanVec Fig. 14, viewed on page 23 of the PDF.** Legend: search {LeanVec-ID, LeanVec-OOD} crossed with build {FP16, LeanVec-ID, LeanVec-OOD}, on deep-256-1M, gist-960-1M, t2i-200-1M and rqa-768-1M. That is 2×3 cells per dataset, as QPS against 10-recall@10 from 0.80 upward — [LeanVec PDF, FT+FIG](https://arxiv.org/pdf/2312.16335).
- **What the plot shows.** Curves sharing a search method lie close together; search method separates curves on t2i-200-1M and rqa-768-1M. Build curves spread somewhat on deep-256-1M at the low-recall end. This is my reading of the plot. The paper gives no table or effect size — same source.
- **Limits of that design.** No walk on full vectors or on full-dimension LVQ is in the figure, and FP16 is the only non-reduced build. Reduced build and walk share d, and no variation of d is crossed with build.
- **LVQ.** One factor only: build bits vary, walk fixed at float32 — [LVQ, FT](https://arxiv.org/html/2304.04759).
- **Bi-metric.** Compares proxy-built and single-metric graphs, with different walks — [Bi-metric, FT](https://arxiv.org/html/2406.02891).
- **Searches.** Targeted queries on factorial or decoupled build/search spaces returned nothing beyond the above, including QuIVer and unrelated storage-decoupling work ([SmartANN, SS](https://arxiv.org/pdf/2609.08240), [PiPNN, SS](https://arxiv.org/pdf/2602.21247)).

### Inferences
- A safe framing is: "LeanVec crossed reduced build and walk spaces and saw no noticeable difference on its Vamana graphs. We extend this to full, PCA-float, PCA-8-bit and plain-SQ8 levels on HNSW, with effect sizes and repeated builds, and find a dataset-signed interaction."
- A design that separates the competing explanations: build ∈ {full, PCA float, PCA 8-bit, SQ8} crossed with walk ∈ {full, SQ8, PCA 8-bit}. If PCA-float builds gain under a full walk, that favors denoising. If gain appears only when walk matches build, that favors space alignment, Epzero's untested working explanation.
- Build seeds must be repeated. HNSW insertion order alone moved recall by up to 7.1 points on HNSWLib and 11.5 on FAISS in one study — [Elliott & Clark, FT](https://arxiv.org/html/2405.17813).

### Gaps
- SmartANN and PiPNN were not read beyond search summaries; a crossed design could still exist there or in unindexed venues.

## Which theory explains graph quality in a lower-dimensional projection?

### Takeaway
No accepted theory exists. Existing work explains why quantization can preserve comparisons and why hubness and recall track intrinsic dimension. Nothing read predicts a gain, and LeanVec explicitly calls for more theory.

### Cited Findings
- **Hubness.** For PCA, ICA, SNE and real data, skewness of N_k "stays relatively constant until a small percentage of features is left, after which it suddenly drops", which the authors describe as the point where "the intrinsic dimensionality of data sets is reached, and further reduction of dimensionality may incur loss of valuable information". i.i.d. data instead falls steadily — [Radovanović et al. 2010, FT via crude PDF text extraction, Section 6](https://jmlr.org/papers/v11/radovanovic10a.html). They left the interaction with dimensionality reduction as "future work".
- **Hub highways.** HNSW's hierarchy gives little in high dimensions because hubness induces "easily-traversed highways" — [Hub Highway, FT](https://arxiv.org/html/2412.01940). This is a hypothesis with empirical support, and it was not tested under projection.
- **Intrinsic dimensionality and recall.** Recall fell "by approximately 50% as synthetic data approaches full rank"; ordering by local intrinsic dimensionality affected recall — [Elliott & Clark, FT](https://arxiv.org/html/2405.17813).
- **Distance-comparison preservation.** Flash's Theorem 1 keeps comparisons correct when the compression error is small relative to a margin — [Flash, FT](https://arxiv.org/html/2502.18113). LVQ's Proposition 1 does the same for pruning — [LVQ, FT](https://arxiv.org/html/2304.04759). Both bound harm rather than predict benefit.
- **Proxy metrics.** Bi-metric results depend on C-approximation and doubling dimension — [Bi-metric, FT](https://arxiv.org/html/2406.02891).
- **Noise claims.** QPAD's abstract says embeddings "suffer from noise" — [QPAD, FT](https://arxiv.org/html/2504.16335). QuIVer decomposes recall as "50%" from low-rank structure plus "45%" from contrastive geometry — [QuIVer, FT](https://arxiv.org/html/2605.02171v1). I did not find a test that denoising improves graph links.
- **Variance.** After PCA, OpenAI-1536's first 512 dims hold nearly 90% of variance — [MRQ, FT](https://arxiv.org/html/2411.06158). Epzero measures top-256 variance at 94% for GIST1M and 80% for a DBpedia sample, top-512 at 98% and 92% — [EPZERO].

### Inferences
- Variance retained fits Epzero's direction: GIST at `nav_dim` 128 keeps 87%, DBpedia at 384 keeps something between 80% and 92%, and the 11% loss on plain SQ8 is a different mechanism because no dimensions are dropped. It explains which dataset is safer, not why GIST *gains*.
- Three candidate explanations remain open and testable: removal of low-variance noise, alignment of graph and walk metrics, and hubness change. No source read tests any of them for graph links.

### Gaps
- Hubness of the projected spaces has not been measured on GIST or DBpedia in anything I read.
- The Hubs quotes come from crude text extraction. Check them against the published PDF before quoting.

## How is query projection cost handled, has anyone quantized the projection, and how is target dimension chosen?

### Takeaway
Most papers call projection cheap or switch to structured rotations. LoRANN (NeurIPS 2024) is the closest prior art for int8: it uses a float PCA projection, then int8 query and int8 model matrices on VPDPBUSD. In the text read, nobody rounds the PCA matrix itself to int8 for a graph walk. Target dimension is chosen by sweep or by a variance rule, and is still called an open problem.

### Cited Findings
- **Called negligible.** "done only once per search incurring a negligible overhead" — [LeanVec, FT](https://arxiv.org/html/2312.16335). That was with d of 160 and 72-thread throughput runs. Epzero measures 28 µs at 256 directions of 1,536 dimensions and 48 µs at 384 on a laptop — [EPZERO].
- **Cost stated.** PCA needs "D2 floats memory cost and O(D2) extra query cost" — [DR survey, FT](https://arxiv.org/html/2403.13491). ADSampling's rotation is "at most 6.18%" of HNSW's total cost, 2.02% at 95% recall — [ADSampling, FT](https://arxiv.org/html/2303.09855).
- **Structured rotation.** SymphonyQG uses a Fast JLT to go "from O(D2) to O(D log D)" — [SymphonyQG, FT](https://arxiv.org/html/2411.12229). The RaBitQ Library's default is FHT/Kac — [RaBitQ Library docs, SRC](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/docs/rabitq/rotator.md). [RaBitQ-H, SS](https://www.emergentmind.com/topics/rabitq-h). These are random rotations and do not apply directly to a PCA basis.
- **Avoiding rotation.** LVQ "does not require the rotation of the query vectors", which helps it at low probe counts — [SAQ, FT](https://arxiv.org/html/2509.12086).
- **LoRANN.** Query projected as x̃ = W_s^T x onto top eigenvectors. "we quantize the matrices A and B to 8-bit integer precision", and "By also quantizing the query vector" prediction takes "two 8-bit integer vector-matrix products", using absmax scaling to [−127,127], computed with "VPDPBUSD instructions in the AVX-512 VNNI". It uses a "mixed-precision decomposition by not quantizing the first component" and a random rotation "at no extra cost before quantization". "Incorporating 8-bit quantization improves not only the memory usage but also the query latency." — [LoRANN, FT](https://arxiv.org/html/2410.18926v1). LoRANN is clustering plus reduced-rank regression, not a graph; its QPS-recall curves cross Glass "between 80% and 99% on most of the data sets", with k=100.
- **Integer random projection.** VQ-bench notes random projections moved to "i.i.d. ±1" matrices — [VQ-bench, FT](https://arxiv.org/html/2608.11240). Those are data-independent, unlike a rounded PCA basis.
- **Target dimension by sweep.** LeanVec increases "d in steps of 32 starting from 64" on the same uncompressed graph, and says users can pick d∈[160,256] — [LeanVec, FT](https://arxiv.org/html/2312.16335).
- **By variance.** HNSW-PCA uses cumulative variance above 0.9, giving dPCA 420 on LAION; Flash chooses dF by grid search — [Flash, FT](https://arxiv.org/html/2502.18113). MRQ uses about 80% variance aligned to a multiple of 64: GIST to 128, OpenAI-1536 from 354 to 512 — [MRQ, FT](https://arxiv.org/html/2411.06158). MRQ is inconsistent here: one line says GIST needs 78 dims, another 128.
- **At search time.** LeanVec-Sphering "allows to set the target dimensionality during search" — [GleanVec paper, FT](https://arxiv.org/html/2410.22347).
- **Open.** "finding the optimal hyper-parameters, such as the dimensionality of the representations, is a challenging open problem" — [DR survey, FT](https://arxiv.org/html/2403.13491). Epzero also finds fewer directions win at lower recall — [EPZERO]. LeanVec's fixed 90% target hides this.

### Inferences
- LoRANN's text suggests W_s stays float, since quantization applies to x̃ and to A, B. That is my reading of the passages, and should be confirmed in the LoRANN code before relying on it.

### Gaps
- No rule in the literature predicts the best d from data alone. No source treats d as a function of recall target for graphs.
- Other searches for fixed-point PCA in ANN found nothing beyond LoRANN, but its discovery came late, on a differently worded query, so a similar miss is possible.

## Do learned or supervised projections beat PCA for ANN?

### Takeaway
Evidence is thin. They win clearly when queries differ from the data (OOD). In-distribution gains over PCA are small or come from small experiments. RAE and QPAD report no graph QPS-recall against PCA, and QPAD excludes standard PCA altogether.

### Cited Findings
- **RAE.** 10,000 to 20,000 samples across CelebA, IMDb, ImageNet-Tiny and Flickr30k, measuring k-NN preservation, not ANN throughput. ImageNet(384d), Euclidean: PCA 70.76 vs RAE 71.23 at dim 128; 88.21 vs 88.65 at 256. On the other three datasets, on cosine, it surpasses PCA "by at least 12%"; on Euclidean it is "comparable". CelebA dim 128, Euclidean: PCA 86.10 beats RAE 84.38 — [RAE, FT](https://arxiv.org/html/2509.25839).
- **QPAD.** Claims it beats 11 baselines, but "Kernel PCA is included instead of standard PCA". First place in 100% of configurations on Fasttext, 97.7% on Isolet, 59.9% on Arcene, 63.7% on PBMC3k. Runtime 1,445.2 s on SIFT-1M — [QPAD, FT](https://arxiv.org/html/2504.16335). Its comparison says nothing about plain PCA.
- **Matryoshka.** Trained into the model. Adaptive retrieval with shortlist K=200, Ds=16, Dr=2048 gives "up to 14×" lower cost in MFLOPs on ImageNet, measured as mAP@10, with no graph index — [Matryoshka, FT](https://arxiv.org/html/2205.13147). It beats post-hoc SVD on ResNet50 ImageNet-1K. Applying this to OpenAI text embeddings is not covered there.
- **LeanVec-OOD.** Frank-Wolfe on query and database matrices: up to 1.5x over LeanVec-ID on rqa-768-1M. On ID data, LeanVec-ID and OOD "show similar performance"; LeanVec-ID has "accuracies on par with" deep alternatives — [LeanVec, FT](https://arxiv.org/html/2312.16335). Transformer inference "exceeds the runtime of the search itself" — same source.
- **Sphering and GleanVec.** On OOD sets, SVD "clearly underperforms"; GleanVec beats LeanVec-Sphering in all reported cases, with wider gaps at lower d — [GleanVec paper, FT](https://arxiv.org/html/2410.22347).
- **LoRANN.** Regression trained on the query distribution beats corpus training on Yandex T2I OOD, especially without re-ranking — [LoRANN, FT](https://arxiv.org/html/2410.18926v1).
- **Survey.** On GIST, PCA, OPQ, ADSampling, DWT and SEANet* all give about 1.5x over raw HNSW; best alternatives reach "up to 6x", but with "no improvement at all" in some cases — [DR survey, FT](https://arxiv.org/html/2403.13491).

### Inferences
- PCA remains the right baseline for Epzero's in-distribution data. An OOD test such as a text-to-image dataset would need LeanVec-OOD or Sphering as a comparator.

### Gaps
- No paper found tests RAE, QPAD or Matryoshka with graph build and walk on the reduced vectors against PCA.

## Who does integer scoring of 8-bit codes against per-query rounded weights?

### Takeaway
Integer scoring is established, and Glass already does int8 × int8 on HNSW with a global scale. LoRANN does per-query rounding against 8-bit corpus factors, in IVF. RaBitQ does it for 1-bit codes. Epzero's specific variant is the only one found for graphs: per-dimension SQ scales folded into weights, one per-query scale, and outliers kept in float.

### Cited Findings
- **Glass SQ8U.** `to[j] = calibrator.transform(from[j])` with `helpa::dot_s8_s8` or `helpa::l2_s7_s7`, returning `int32_t`. The query is encoded by `this->encode`, with `calibrator` set at training, so the scale is global, not per query. Glass SQ8 instead keeps a float query against per-dimension uint8 codes — [pyglass source, SRC](https://github.com/zilliztech/pyglass).
- **LoRANN.** Absmax per vector: x_i8 = ⌊(127/‖x_f32‖∞)·x_f32⌉, so the scale is per query. r is re-quantized and scored against B_i8, whose columns correspond to corpus points, on VPDPBUSD — [LoRANN, FT](https://arxiv.org/html/2410.18926v1).
- **RaBitQ.** Uniform SQ of the rotated query to B_q-bit unsigned integers, with Δ and v_l known at query time, against 1-bit codes — [RaBitQ, FT](https://arxiv.org/html/2405.12497).
- **SAQ.** Multi-bit codes: "if we quantize each dimension of a query vector to integer, Eq. 13 will mostly use integer computation" — [SAQ, FT](https://arxiv.org/html/2509.12086). It is tested with IVF; proximity graphs are left to future work.
- **BBQ.** 1-bit documents, int4 queries — [Elastic, primary blog](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch). It also says "allowing the graph to be built with the quantized vectors" — same source.
- **Lucene.** int8 via `byte = (float - minQuantile) * 127/(maxQuantile - minQuantile)`, with corrective offsets; quantiles are index-level — [Lucene API, SRC](https://lucene.apache.org/core/10_2_2/core/org/apache/lucene/util/quantization/ScalarQuantizer.html). I did not confirm the query path in code.
- **LUT rounding.** FastScan-style PQ quantizes lookup tables to 8-bit unsigned integers — [RaBitQ, FT](https://arxiv.org/html/2405.12497). Flash rounds per-vector tables to 8 bits during build — [Flash, FT](https://arxiv.org/html/2502.18113). This is rounded per-query tables, not a dot product.
- **Low-Precision.** int8 database vectors on HNSW, FAISS and NGT; the authors extended HNSW's inner-product code to int8 arrays. The query path was not confirmed — [Low-Precision Quantization, FT](https://arxiv.org/html/2110.08919).
- **MonaVec, opposing.** Keeps queries float: "errors would accumulate on both sides of the dot product" — [MonaVec, FT](https://arxiv.org/html/2606.19458).
- **QuIVer.** Replacing symmetric navigation with ADC "decreased QPS by 9.4× while improving recall by only 3.2%" — [QuIVer, FT](https://arxiv.org/html/2605.02171v1).
- **Epzero.** Weights are `query[i] * sq8_scale_[i]`, rounded to int8 "on one scale per query", with up to 8 largest kept in float when the top weight is over 4x the next; recall was capped at 0.93 on OpenAI embeddings before that fix, with 15% error falling to 0.3% — [EPZERO, `engine/src/hnsw.cpp`, `docs/performance.md`].

### Inferences
- Epzero's outlier split resembles LoRANN's mixed-precision principal-component treatment. A paper should cite LoRANN for the general idea.

### Gaps
- Faiss's SQ query path was not confirmed; its source was refactored and I did not trace it.
- SAQ's actual kernel and MRQ's B_q settings were not extracted.

## How do the Epzero findings compare with this literature?

### Takeaway
None of the three is new as a technique. The defensible new contributions are measurements: a dataset-signed build effect under a fixed walk, HNSW-specific gains beyond a tuned SQ8 competitor, and int8 rounding of the PCA matrix itself on a graph. Each needs the checks below.

### Cited Findings

Epzero figures reconciled to `docs/performance.md`, Gold 6226, one thread — [EPZERO]:
- GIST1M at 0.95: 2,590 vs Glass 723 (3.58×, `nav_dim` 192). DBpedia: 1.16× at 0.85, 1.21× at 0.90, 1.28× at 0.95, 1.57× at 0.98, 2.07× at 0.99. The 1.16× point is just below the brief's "1.2x".
- These multiples come from the round with the second pass. The earlier round without it gave 3.15× on GIST and 0.89× to 1.18× on DBpedia, so they mix walk and graph refinement effects.
- Build on codes: GIST1M `nav_dim` 128, 1,266 vs 987 QPS at 0.95 (+28.3%), 47 s vs 223 s. DBpedia `nav_dim` 384, 3,438 vs 3,635 (−5.4%), 95 s vs 307 s. Plain SQ8: −11% on DBpedia, −5% on DEEP10M, unchanged on SIFT1M and GloVe-100 (laptop).
- Int8 projection: 28 µs and 48 µs float cost; +31-57% at 256 directions and +35-72% at 384 on the 37,000-vector DBpedia sample. I read only the start of that passage, so its recall range and conditions need checking.

| Finding | Status | Evidence | Safe wording |
|---|---|---|---|
| Walk on 8-bit PCA codes plus re-rank | **Already known** as a method; **new** as a measurement | LeanVec has the same design with 10.2x and 3.7x on GIST at 0.90 on Vamana. Flash's HNSW-PCA is a close HNSW variant. No paper found measures this design against Glass | "We confirm LeanVec-style walking on HNSW, measured against Glass" |
| DBpedia gain, rising with recall | **Partly known** | PAG, QuIVer and MRQ all use this data, none in this design. Dependence of d on recall target is not studied for graphs | "The best d rises with the recall target" as an observation |
| Build on codes, +28% on GIST | **Partly known; gain apparently new** | Building is known. LeanVec Fig. 14 crossed this and saw no noticeable difference, including on gist-960-1M. Flash's gain has no same-routine test | "Unlike LeanVec's null on Vamana, we find..." once confirmed |
| Build on codes, −5% on DBpedia | **Partly known** | Flash reports data sensitivity and PQ-built degradation. No same-walk loss from PCA build on OpenAI embeddings found | Report with multi-seed error bars |
| Signed interaction by dataset under a fixed walk | **Apparently new** | LVQ and LeanVec show flat results; Flash is confounded | Avoid "first factorial"; LeanVec crossed 2×3 |
| Int8 projection matrix and query | **Partly known; narrow part apparently new** | LoRANN has int8 query, int8 matrices, VPDPBUSD and absmax scaling, after a float PCA. Integer random matrices are older | "We apply integer projection, as in LoRANN's regression factors, to the PCA basis of a graph walk" |
| Integer SQ8 with per-query scale | **Partly known** | Glass (global scale), LoRANN (per query, IVF), RaBitQ (1-bit) | Cite all three; claim only the graph-specific combination |

### Inferences
- Before the +28% goes in a paper, four things must hold. The re-rank must match between graphs, or the result must be stated at equal recall. Builds must be repeated with seeds, since a 5% difference can sit inside order effects. `fast_build` must be separated into its search and pruning parts, as Epzero notes it uses codes for both. The `refine_links` and `fill_free_links` steps must stay out of the comparison.
- The strongest paper claim is an explanation, not a speedup. LeanVec named the open question: "Further theoretical studies to understand this phenomenon are required." A factorial that separates denoising from alignment would address it directly.
- GIST as the positive case carries risk. It is a legacy dataset, and PAG argues such datasets are outdated. A modern high-variance-concentration dataset would make the result stronger.

### Gaps
- Epzero's `compete.py` calls Glass's build without a `quant` argument, so it uses the installed version's default. That is FP32 in the master bindings I read, but `compete.py` passes five `Searcher` arguments where those bindings show four, so the installed Glass may differ from master. Confirm the version and its build default and re-rank multiplier, and state them next to "3.6× Glass".
- I did not rerun or inspect the raw JSON behind the Epzero numbers; I used the log.
- LeanVec's and Epzero's GIST results differ in graph type, degree, thread count and code type, so they are not a direct contradiction until tested on one system.
