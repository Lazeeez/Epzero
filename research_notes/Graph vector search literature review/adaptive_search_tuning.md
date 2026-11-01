# Adaptive search and automatic tuning for vector indexes: review through late 2026, with novelty checks for Epzero

Read-status labels used below. The brief named three; a fourth, PARTIAL, is added because most papers were read section by section, and calling those "full" would overstate it.
- FULL: every section relevant to the question read as text.
- PARTIAL: named sections or keyword-located passages read as text, the rest not.
- ABSTRACT: abstract or an introduction summary only.
- SEARCH: search-engine summary or a third-party index only.
- NOT READ: known only through another paper's description.

Verification note: an automated page summariser misquoted one number (the DCO benchmark's "6%-36% of dimensions" came back as "66%-36%", and it omitted that the sentence is about Deep only). Numbers below were therefore rechecked against raw text unless marked "(summariser only)". Percentages marked "computed" are my arithmetic from the source's table.

## Early termination: what is gained against an index tuned to the same recall

### Takeaway
Headline speedups of 3-15x are mostly measured against a search configured for a higher recall than the target, or against a hard query mix. Where the baseline really is a fixed ef tuned to the same recall on a default workload, the only clean graph-index numbers found are VSAG's +0.6% to +7.2% (computed) and Epzero's own 5-10%. The exception is LAET at recall-at-1, up to 7.1x, because k=1 tails are extreme.

### Cited Findings

Baseline honesty table (HNSW or graph unless noted):

| Method (status) | Baseline | Same recall as baseline? | Result |
|---|---|---|---|
| LAET, SIGMOD 2020 (PARTIAL) | Fixed efSearch or nprobe, swept | Yes | Up to 7.1x (86% latency cut) HNSW, 95-100% recall-at-1, DEEP10M/SIFT10M/GIST1M, no compression |
| VSAG QLP tuner, VLDB 2025 (PARTIAL) | FIX: smallest ef reaching target | Yes | +4.4%, +7.2% GIST1M; +0.6%, +2.8% SIFT1M, recall@10 94% and 97% (computed) |
| DARTH vs REM, SIGMOD 2026 (PARTIAL) | HNSW with efS tuned per target and per noise level | Yes, on a hard workload | Up to 5.8x, 3.1x on average, Rt=0.90, noise=12%, k=50 |
| DARTH vs plain HNSW (PARTIAL) | efS giving recall >=0.99 | No | Up to 14.6x, avg 6.8x, median 5.7x |
| Patience in Proximity, ECIR 2025 (FULL) | Same ef, no retuning | Approximately, by accident of plateau | +18% to +61% QPS (computed), BGE on BEIR, Lucene |
| Patience, Busolin et al., CIKM 2024 (PARTIAL) | Fixed clusters N reaching R*@1=0.95 | Yes, but IVF | 2.95x, 4.04x, 5.13x on MS-MARCO |
| Terminus, MLSys 2026 (PARTIAL, through 6.3) | IO-Budget | Yes, on RAG accuracy, not recall | 1.4x (Pythia-1B, accuracy 0.16); 1.16x (LLaMA-2-7B, 0.31) |
| Terminus vs Starling (PARTIAL) | No early termination | No | Up to 3.2x with "minimal impact" |
| Ada-ef, SIGMOD 2026 (PARTIAL) | DARTH | Yes, same average recall | Up to 4x lower workload latency |

- LAET's reported gains: "the average latency is reduced by up to 86% (7.1 times speedup)" on HNSW for 95-100% recall-at-1, and up to 52% (2.1x) for IVF+OPQ at 95-100% recall-at-100, up to 59% (2.4x) for IMI+OPQ — [LAET](https://www.pdl.cmu.edu/ftp/BigLearning/mod0246-liA.pdf)
- LAET's motivation data: 80% accuracy needs searching up to 0.13% of graph nodes, while 95-100% needs up to 11.83% — [LAET](https://www.pdl.cmu.edu/ftp/BigLearning/mod0246-liA.pdf)
- LAET uses gradient-boosted trees to predict a stop point once, with a tuned multiplier on the prediction — [LAET](https://www.pdl.cmu.edu/ftp/BigLearning/mod0246-liA.pdf)
- DARTH's plain-HNSW baseline is configured for "average recall >=0.99", and its speedup at k=50 is stated as "up to 15x, on average 6.75x, and median 5.7x" (the abstract says 14.6x); "the speedup decreases for higher recall targets" — [DARTH](https://arxiv.org/html/2505.19001v2)
- DARTH's comparison with REM is explicitly for a hard workload, Rt=0.90 and noise=12%, k=50: "up to 280QPS (100QPS on average) more queries than REM, while being up to 5.8x (3.1x on average) faster than REM". The stated reason for using REM here is that "the competing approaches do not provide the required accuracy" — [DARTH](https://arxiv.org/html/2505.19001v2)
- DARTH's per-query reliability on DEEP100M at average recall about 0.95: 28% of queries below target for Baseline, 22% for LAET, 21% for REM. Worst query recall was 0.46 for Baseline and LAET, 0.55 for REM — [DARTH](https://arxiv.org/html/2505.19001v2). The figure of 13% for DARTH is (summariser only).
- DARTH's predictor features are index progress (nstep, ndis, ninserts), nearest-neighbour distances, and summary stats including mean, variance, median and quartiles of result distances. Invocation intervals are ipi=distsRt/2 and mpi=distsRt/10 — [DARTH](https://arxiv.org/html/2505.19001v2)
- DARTH's 2-part offline cost: training data generation "20-36 minutes", training "85-155 seconds" (summariser only) — [DARTH](https://arxiv.org/html/2505.19001v2)
- Ada-ef contradicts DARTH's reliability claim on modern embeddings: at target 0.95 "both methods reach the declarative target recall of 0.95 on five out of" its datasets, with recall "considerably lower" on GloVe-100, and DARTH "achieves higher recall than LAET, but at the expense of substantially higher query latency" — [Ada-ef](https://arxiv.org/pdf/2512.06636)
- Ada-ef is rule-based: it models the query-to-data distance distribution as normal, scores query difficulty, and looks up ef from a table built from 200 sampled vectors. It claims 50x lower offline time and 100x lower offline memory than DARTH/LAET on average, and it adapts to updates — [Ada-ef](https://arxiv.org/pdf/2512.06636)
- VSAG's oracle claim versus its result: "assigning personalized optimal retrieval parameters to each query can improve retrieval performance by 3-5 times", while Table 7's measured QPS is 1469 to 1534 and 902 to 967 on GIST1M, and 4027 to 4050 and 2834 to 2912 on SIFT1M. The text says "over a 5% increase", which the SIFT1M columns do not support — [VSAG](https://arxiv.org/pdf/2503.17911)
- PiP stops when at least Φ% of the k results stay unchanged for Δ consecutive candidates, with Φ=95 shown. Table 1 QPS, plain then patience at search time: Robust04 73.7 to 87.1, HotpotQA 70.6 to 100.1, DBPedia 47.9 to 77.1, Climate-FEVER 73.4 to 102.5, TREC-COVID 62.4 to 73.9. HotpotQA R@100 falls 0.8487 to 0.8422. The text's "64%" for DBPedia matches the index-and-search row (78.5), not the search-only row, which is 61% (computed) — [PiP](https://cs.uwaterloo.ca/~jimmylin/publications/Teofili_Lin_ECIR2025.pdf)
- PiP compares at a single fixed setting. It does not retune ef down to the same recall, and the ef value is not stated in the paper text — [PiP](https://cs.uwaterloo.ca/~jimmylin/publications/Teofili_Lin_ECIR2025.pdf)
- Ada-ef notes that PiP, unlike the learned methods, needs a preset ef — [Ada-ef](https://arxiv.org/pdf/2512.06636)
- The patience idea originated for IVF: Φ=95 with Δ of 7, 12 and 14, speedups 2.95x to 5.13x at R*@1=0.95 — [Busolin et al.](https://arxiv.org/html/2408.04981v1)
- Patience is in production: Lucene ships [PatienceKnnVectorQuery](https://lucene.apache.org/core/10_2_2/core/org/apache/lucene/search/PatienceKnnVectorQuery.html). Elasticsearch 9.2 had static patience, and 9.3 enables an adaptive form by default, where saturation threshold is the rolling mean plus SD of discovery rate (Welford) and patience scales inversely with SD — [Elastic](https://www.elastic.co/search-labs/blog/hnsw-elasticsearch-adaptive-early-termination) (summariser only; the blog gives charts, and no extracted percentages)
- Terminus defines per-I/O utility as a sum of exponentially decaying rank weights over new queue insertions, with sigma=1.8, beta=0.5, and stops when utility stays below a threshold for a window X=2. Overhead is at most 2% of latency — [Terminus](https://www.cs.princeton.edu/~mfreed/docs/terminus-mlsys26.pdf)
- Terminus's key caveat: "the Recall-I/O curves across all systems are nearly identical", so its advantage exists only under rank-weighted metrics. Also, "Terminus delivers the highest gains in the medium-high accuracy range (i.e., 0.6 - 0.9)". Setting is on-disk (Starling), Wikipedia-20M and SPACEV-100M, k=20 — [Terminus](https://www.cs.princeton.edu/~mfreed/docs/terminus-mlsys26.pdf)
- Terminus reports 1.6x over VBASE at accuracy 0.33 on LLaMA-2-7B — [Terminus](https://www.cs.princeton.edu/~mfreed/docs/terminus-mlsys26.pdf)
- Adaptive Beam Search replaces beam-width termination with a distance rule (stop when a candidate exceeds (1+gamma) times a current result distance, with a guarantee on navigable graphs), and "consistently outperforms classic beam search in terms of total distance computations" for equal recall — [ABS, NeurIPS 2025](https://arxiv.org/html/2505.15636v1) (PARTIAL; no percentage extracted)
- ConANN gives distribution-free FNR control for IVF by conformal risk control, with worst-case deviation 0.03 FNR from target, up to 62.8% fewer probes, overhead under 1.97 ms, and calibration under 4 minutes — [ConANN, PVLDB 19(1)](https://www.vldb.org/pvldb/vol19/p29-horchidan.pdf) (PARTIAL). A follow-on project lists extending this to HNSW and LSH — [project page](https://nim.nsc.liu.se/projects/7862/) (SEARCH)
- Tao replaces runtime features with Local Intrinsic Dimension (NOT READ; only via [Ada-ef](https://arxiv.org/pdf/2512.06636))

### Inferences
- A paper should report gains only against fixed ef tuned to the same mean recall on the same query set. On that basis a 5-15% gain is normal for rule-based methods on default workloads, and is not weak compared with 14.6x.
- VSAG's oracle claim (3-5x) versus realised 0.6-7.2% marks the gap between per-query potential and what a two-class predictor captures. That gap is an open and citable problem.
- Gains scale with query-hardness skew. LAET's k=1 result and DARTH's noise=12% result both sit in high-skew regimes, which is the likely reason they are large.
- Terminus's near-identical Recall-I/O curves are independent evidence that a rule-based stop may give small gains on plain recall.

### Gaps
- No same-recall DARTH-versus-REM QPS on the default (non-noisy) workload was found in the passages read. Its per-Rt plots were not read as numbers.
- Ada-ef's percentage against tuned HNSW at equal recall was not found as a number in the text.
- DARTH+ (a follow-up, [HAL preprint](https://helios2.mi.parisdescartes.fr/~themisp/publications/hal26-darthplus.pdf)) was NOT READ: the server blocked automated download. A related [VLDB 2026 PhD workshop paper](https://www.vldb.org/2026/Workshops/VLDB-Workshops-2026/PhD/PhD26_3.pdf) also failed to download.

## Adaptive re-ranking: how many, error bounds, calibration, stopping early, and distance-comparison methods

### Takeaway
"The right re-rank depth varies per query" is established, and early stops exist in four forms: rank-change stability, fixed-threshold separation tests, per-candidate theoretical bounds, and offline-fitted error models. No source was found that estimates the signed gap between exact and estimated distance online, from the same query's already-scored candidates, and uses it as a stopping test. Distance-comparison methods are real but conditional: at most 1.1-2.1x on HNSW with SIMD.

### Cited Findings

Re-rank stopping:
- FusionANNS (FAST 2025, PARTIAL): "the minimum re-ranking numbers for different ANNS queries usually vary significantly". It sets a conservative large depth, runs mini-batches in ascending estimated order, computes the change rate of the top-k id set, and terminates after the rate stays below epsilon for beta consecutive batches. With I/O deduplication this cuts I/O further: "up to 30% and 23%, respectively" — [FusionANNS](https://arxiv.org/pdf/2409.16576)
- AQR-HNSW (2026 arXiv, PARTIAL): tests separation between positions k and k+1 on asymmetric distances, using gap and ratio metrics against fixed thresholds. If either passes, "exact reranking is limited to at most k candidates". It reports 35% fewer computations from multi-stage re-ranking, and 1.8-2.5x throughput — [AQR-HNSW](https://arxiv.org/pdf/2602.21600)
- VSAG Selective Re-rank (PARTIAL): "The optimal number of candidates requiring re-ranking varies significantly depending on query characteristics, quantization error distribution, and search requirement k." It implements the DDC scheme for this — [VSAG](https://arxiv.org/pdf/2503.17911). DDC itself was NOT READ.
- MRQ (PARTIAL): on PCA-projected data, estimates a lower bound and "check[s] whether this bound exceeds the current top-k threshold"; if so the candidate "can be safely pruned". Stages run quantized, projected, then exact. The bound comes from a known quantization error and offline residual variance. Up to 3x faster — [MRQ](https://arxiv.org/pdf/2411.06158)
- Extended RaBitQ (PARTIAL) describes the mechanism from original RaBitQ: if "a lower bound of the estimated distance is greater than the distance of the NN that has been searched so far", discard; otherwise refine. The empirical error formula is fitted offline against the 99.9% quantile — [Extended RaBitQ](https://arxiv.org/pdf/2409.09913). Original RaBitQ, SIGMOD 2024, was NOT READ directly.
- FaTRQ (DATE 2026, PARTIAL): "learns a lightweight linear calibration model offline" on about 0.3% of the database, over features including the coarse distance, to predict true distance near the decision boundary. "Refinement stops early once a candidate is provably outside the top-k", per candidate, in a CXL far-memory design. Refinement cost cut by 85% — [FaTRQ](https://arxiv.org/pdf/2601.09985)
- Elasticsearch DiskBBQ auto-calibration (July 2026): a Gaussian model of quantization noise, with SD measured from "the gap between exact and quantized dot products on a sample", at merge time per segment. It picks encoding and an oversampling factor from 1.25 to 3.0 to hit 90% recall@10. This is a per-segment, index-time decision. Reported: average 16.7% QPS, and 15 of 18 datasets better on both axes (summariser only) — [Elastic](https://www.elastic.co/search-labs/blog/vector-quantization-auto-calibration-diskbbq)
- Panorama (PARTIAL; upstreamed into Faiss) prunes during refinement with strict Cauchy-Schwarz lower bounds along PCA, deterministic per candidate, up to 28.9x end-to-end — [Panorama](https://arxiv.org/pdf/2510.00566)
- EAHR (2026 arXiv, ABSTRACT) uses bounds to stop deep retrieval in hybrid fusion, a different problem — [EAHR](https://arxiv.org/abs/2608.07152)

Distance-comparison methods:
- ADSampling: up to 2.65x on HNSW and 5.58x on IVF "while providing the same accuracy", and saves up to 75.3% of dimensions on HNSW with accuracy loss at most 0.14% — [ADSampling](https://arxiv.org/pdf/2303.09855) (PARTIAL)
- DADE: over 40% QPS on HNSW over ADSampling on DEEP at the same accuracy, using PCA and a hypothesis test — [DADE](https://arxiv.org/pdf/2411.17229) (PARTIAL)
- ICDE 2026 benchmark, 8 algorithms and 10 datasets (PARTIAL): on a SIMD CPU, "PDScanning+, ADSampling, DADE, DDCres, DDCpca, and DDCopq improved QPS by up to 1.9, 1.4, 1.8, 1.9, 2.1, and 1.6x on HNSW". "Enabling SIMD narrows the efficiency advantage... reducing the speedup from 1.9-3.1x to 1.1-2.1x on HNSW." On XUltra, SOTA QPS drops "by up to 77%-79%" against FDScanning. "FDScanning outperforms ADSampling on the GIST dataset" under AVX2. DDCpca is up to 1.6x slower than FDScanning on OOD queries. Takeaway #1: "DCO is no silver bullet." — [DCO benchmark](https://arxiv.org/html/2604.02801v1)
- The same benchmark's Takeaway: tuning Delta0 and Deltad "can substantially boost QPS", but "their optimal values are determined by datasets, indexes, and specific DCO methods" — [DCO benchmark](https://arxiv.org/html/2604.02801v1)
- SkipComputing, SIGMOD 2026 (SEARCH; DOI 10.1145/3769753): "estimation-and-testing DCOs become costly under SIMD"; it uses subspace filtering, distance-decomposition lower bounds and layout changes; "up to 6x vs HNSW and 2.7x vs SOTA" — [index summary](https://rmarcus.info/dbscholar/papers/hcba1211e9aa94b13). Full text not read, so dataset and recall target are unknown.
- FINGER, WWW 2023 (SEARCH): estimates angles between residuals, 20-60% wall-clock improvement — [FINGER](https://arxiv.org/pdf/2206.11408). CRouting reports that FINGER costs 14-78% more construction time and 56-295% more memory on HNSW — [CRouting](https://arxiv.org/html/2509.00365v1)
- CRouting (PARTIAL): cosine-theorem estimate with stored neighbour distances, QPS 1.12x to 1.48x on HNSW, distance computations down by up to 41.5% — [CRouting](https://arxiv.org/html/2509.00365v1)
- PEOs, ICML 2024 (SEARCH): probabilistic routing, HNSW throughput 1.6x to 2.5x — [PEOs](https://proceedings.mlr.press/v235/lu24l.html)
- PAG, 2026 arXiv (PARTIAL): probabilistic routing test with a test feedback buffer, up to 5x faster than HNSW in QPS-recall — [PAG](https://arxiv.org/pdf/2603.06660)
- PDX (PARTIAL): vertical layout restores dimension pruning under SIMD, ADSampling benefit 2-7x (SEARCH for that figure) — [PDX](https://arxiv.org/pdf/2503.04422)

### Inferences
Novelty matrix for the z=3 stop. The Epzero rule, from `engine/src/hnsw.cpp`: score blocks in ascending estimate order, keep the mean and SD of (exact minus estimate) over all scored so far, and stop when next_estimate + mean - z*SD exceeds the current k-th best exact distance.

| Work | Signal | Fitted when | Scope | Needs |
|---|---|---|---|---|
| Epzero | Distance gap, signed mean and SD | Online | Per query | One parameter z, no training |
| FusionANNS | Top-k id change | Not fitted | Per query | epsilon, beta |
| AQR-HNSW | Estimated gap at k | Not fitted | Per query | 2 thresholds, one decision |
| RaBitQ, MRQ | Theoretical or offline bound | Offline or a priori | Per candidate | Quantizer-specific factors |
| FaTRQ | Linear calibration | Offline | Dataset | 0.3% sample |
| DDC (via VSAG) | Learned model | Offline | Dataset | Not read |
| Elastic | Gaussian noise SD | Merge time | Segment | Sample, sets depth once |

**Verdict on z=3 re-rank stop: partly known in aim, apparently new in mechanism.** Safe claim: "We are not aware of prior work that calibrates the estimate-to-exact gap online from the query's own re-ranked candidates and uses it as a stopping test." Unsafe claims: "first adaptive re-rank", "first per-query re-rank stop", "first early-stop for re-ranking"; FusionANNS and AQR-HNSW refute these. Its practical distinctions are that it corrects for per-query bias (the mean), is estimator-agnostic, and needs no training. Confidence is moderate: DDC and RaBitQ 2024 were not read, so check both before submission.

- Compare it with FusionANNS in the paper as the closest baseline, because that method is effectively patience applied to re-ranking. An ablation on the same index would be stronger than any claim of novelty.
- Statistical threats a reviewer will raise (mine, not sourced): candidates are sorted by estimate, so the scored head is selection-biased, which likely inflates the mean gap and makes stopping eager. The test checks only the next candidate, not all remaining ones. Gaps are probably heteroscedastic with distance. z=3 therefore has no calibrated coverage meaning, and "within 0.0004 recall" should stand as an empirical result, not a guarantee.
- The gains' pattern (GIST at low ef, DBPedia at higher ef) needs an explanation, since it may reflect where the fixed re-rank budget was oversized. A same-recall comparison against a retuned fixed re-rank depth is the honest one.

### Gaps
- DDC (SIGMOD 2024) and original RaBitQ were not read. RaBitQ's arXiv text was not fetched after the download batch containing a blocked retry was refused.
- Whether Faiss Refine, ScaNN reordering, Qdrant oversampling or Milvus refine have online per-query stopping in source was not checked.
- Absolute conclusions about non-existence cannot be drawn from web search. Non-English and patent literature were not systematically searched.

## Automatic parameter tuning, including tuners adapting to data and CPU

### Takeaway
Black-box tuners are mature and reach 98-99.97% of an oracle. Measurement-based CPU adaptation exists in VSAG, SVS and Glass. No tuner was found that transfers across CPUs without re-measuring, or that models hardware explicitly.

### Cited Findings

| Tuner (status) | Tunes | Adapts to data | Adapts to CPU | Result |
|---|---|---|---|---|
| VDTuner, 2024 (PARTIAL) | 16 params in Milvus 2.3.1 by multi-objective BO | Yes | Only by measuring on the machine | 14.12% speed and 186.38% recall over default; tuning 1.48x-3.57x faster |
| CHAT, SIGMOD 2027 (PARTIAL) | M, efC, boundary search on efS, resource filter | Yes | Not addressed; one Threadripper 3970X | 98.03-99.97% of oracle QPS; 1-45% over baselines; 99% of oracle in 48 min avg (13-123) |
| VSAG, VLDB 2025 (PARTIAL) | Three levels: ELP (prefetch), QLP (ef per query), ILP (index) | Yes | Yes: grid search of environment parameters | See below |
| SVS `experimental_calibrate` (docs) | Window, buffer, prefetch lookahead and step at a target recall | Yes | By timing, with the assigned threads | No published gain found |
| Glass/pyglass `optimize` (source) | Prefetch po and pl only | Partly | Yes: picks minimum elapsed time | No recall measured; ef not changed |
| Milvus/Zilliz AUTOINDEX (docs) | Single `level` | Yes | Not stated | Level 1 about 90% recall |
| Elastic auto-calibration (blog) | Bits, oversampling | Per segment | Costs, not CPU model | 16.7% QPS (summariser only) |

- VSAG's tuner: strategy row "Above + 4. ELPs Auto Tuner" moves QPS 1517 to 2052 on GIST1M and 3565 to 4946 on SIFT1M at unchanged recall of 89.8% and 98.4%, or +35% and +39% (computed). Prefetch "heavily depends on environment-level parameters". ELP tuning took 10s, ILP 68s, all tuning 128s, against 5998s to build. Online QLP overhead is 0.001s versus 0.217s of search — [VSAG](https://arxiv.org/pdf/2503.17911)
- VDTuner's numbers — [VDTuner](https://arxiv.org/pdf/2404.10413)
- CHAT: "up to 45% higher throughput or 11% higher recall"; converges "1.5x-44x faster" than the best competitor; supports warm-start retuning when corpus or workload changes; the testbed is single-machine, "CPU-only" — [CHAT](https://arxiv.org/html/2607.04630v1)
- SVS: "calibration will find parameters that optimize performance at this recall level"; takes `prefetch_steps`; marked experimental — [SVS docs](https://intel.github.io/ScalableVectorSearch/python/vamana.html)
- Glass: sweeps try_pos and try_pls, baseline po=1, pl=1, keeps the lowest elapsed — [pyglass source](https://raw.githubusercontent.com/zilliztech/pyglass/master/glass/searcher.hpp) (summariser only). pyglass also has level 0-2, where a higher level "may cause unstable recall" — [pyglass](https://github.com/hhy3/pyglass)
- AUTOINDEX "analyzes the distribution of users' datasets during index building" — [Zilliz docs](https://docs.zilliz.com/docs/autoindex-explained). It is proprietary, and Cardinal is its engine — [Zilliz](https://zilliz.com/cardinal) (SEARCH). Open-source Knowhere internals were not inspected.

### Inferences
- Tuners that find a fixed best configuration are the correct baseline for adaptive stopping. A paper should run its stop on top of a tuned index, since a CHAT-style oracle is the real competitor.
- Two tuning layers are complementary, not rival: prefetch and environment tuning gave VSAG +35-39%, larger than any per-query adaptive gain at same recall above.
- "Adapts to both data and CPU" is partly known by measurement. A claim of a tuner that predicts hardware behaviour would be new, but a claim that merely re-measures on the host would not.

### Gaps
- No primary performance evaluation of SVS calibration, Glass optimize or Knowhere found.
- Whether CHAT handles a machine change was not found beyond its corpus and workload retuning.

## Partial-distance or two-stage scoring on PCA-ordered dimensions: when it fails

### Takeaway
Failure is well evidenced in general, and conditions are identifiable. Epzero's 10-20% slowdown fits the failure conditions, but its exact setting (8-bit PCA-ordered codes in HNSW with a statistically predicted remainder) was not found reported. Positive results exist and must be cited: PCA gives real gains on full-precision vectors, so this is not "PCA pruning does not work".

### Cited Findings

Failure conditions:
- **Compute already cheap, or memory-bound.** "Raw Panorama-HNSW drops to 0.80x, whereas adding PCA yields a 2.83x speedup. Because pointer-based graph traversals suffer a cache miss on nearly every candidate look-up, runtime is heavily dominated by memory latency", so by Amdahl's law "raw pruning's minor arithmetic savings cannot offset its overhead" — [Panorama](https://arxiv.org/pdf/2510.00566). The passage opens with a GIST example, and it does not explicitly state the dataset for the HNSW figures.
- **Check overhead versus saved dimensions.** On Deep "the time saved by skipping 6%-36% of dimensions is marginal (except for DDCopq), since the accumulated overhead of these methods including distance estimation, threshold checking, and conditional branching outweighs the benefits of dimension pruning" — [DCO benchmark](https://arxiv.org/html/2604.02801v1)
- **SIMD on the baseline.** HNSW speedup falls from 1.9-3.1x to 1.1-2.1x; FDScanning beats ADSampling on GIST under AVX2 — [DCO benchmark](https://arxiv.org/html/2604.02801v1). SkipComputing states the same point (SEARCH).
- **Very high dimension.** At Trevi "the O(D^2) online pre-processing time per query becomes the efficiency bottleneck"; on XUltra QPS drops 77-79% — [DCO benchmark](https://arxiv.org/html/2604.02801v1)
- **Out-of-distribution queries.** DDCpca is up to 1.6x slower than FDScanning — [DCO benchmark](https://arxiv.org/html/2604.02801v1)
- **Non-contiguous layout.** HNSW gets level batch size 1 in Panorama, "the only feasible mode under scattered memory" — [Panorama](https://arxiv.org/pdf/2510.00566). PDX exists to restore pruning by changing layout — [PDX](https://arxiv.org/pdf/2503.04422)
- **Compression conflict.** PCA "breaks the uniform-variance assumption required by Product Quantization", which Panorama fixes by variance reshaping — [Panorama](https://arxiv.org/pdf/2510.00566)
- **Parameter sensitivity.** Delta values depend on data, index and method — [DCO benchmark](https://arxiv.org/html/2604.02801v1)

Positive results:
- PDScanning+ up to 1.9x on HNSW with SIMD — [DCO benchmark](https://arxiv.org/html/2604.02801v1)
- On GIST at recall at least 0.95, PCA scales Panorama QPS 12.3x on IVFFlat and 7.3x on IVFPQ — [Panorama](https://arxiv.org/pdf/2510.00566)
- MRQ succeeds with PCA prefix plus quantization, but on IVF — [MRQ](https://arxiv.org/pdf/2411.06158)
- DADE shows PCA is the optimal orthogonal transform for its estimator — [DADE](https://arxiv.org/pdf/2411.17229)

### Inferences
**Verdict on two-stage PCA 8-bit failure: partly known.** The mechanism is documented, while the specific negative result looks unreported. Those positive results scan full-precision floats, where a skipped dimension saves a 32-bit load and arithmetic. On 8-bit codes the full scan is already cheap, so saved work shrinks while the check and remainder prediction stay. That reading is mine, not stated in any source. A safe wording is "consistent with the overhead effects of [DCO benchmark] and [Panorama]; we extend them to 8-bit codes". It is a useful negative result, but not a headline.

**Verdict on patience at ef/3: already known.** Stop-after-unchanged is PiP with Φ=100 and Busolin before it, and Lucene and Elasticsearch ship it. Scaling patience with ef is a minor variant; Elastic adapts per query. What Epzero can contribute is the measurement: 5-10% against a tuned same-recall baseline, compared with 18-61% in PiP without retuning. That needs confirming that Epzero's figure is measured on that baseline; the brief does not state it.

### Gaps
- No source tested two-stage scoring on SQ8 codes within HNSW with a predicted remainder.
- Baseline hardware differs across sources, so ratios are not comparable with Epzero's AVX-512 kernels. The DCO benchmark used SSE and AVX2.
