# Hardware dependence of graph-based ANN search: prefetching, memory, SIMD, microarchitecture

Coverage: searches through 2 November 2026.

Reading labels. No paper here was read cover to cover, so "read in full" is not used.
- **[passages]**: full text downloaded and the relevant sections read directly, with numbers checked against the raw text.
- **[summary]**: page fetched but read only through an automated summary, so quotes are less certain.
- **[search]**: search-result summary only.
- **[code]**: source file read directly.
- **[Epzero]**: this repository's code or `docs/performance.md`, or results supplied by the coordinator.

Automated summaries of VSAG and of the CCD paper returned generic figures ("typically 30-50%", "10-40%") that are not in the papers. Those were discarded and both papers re-read from raw text. Any secondary quotation of those figures should be treated as wrong.

## Which papers benchmark ANN across several CPUs, and does a tuning that is best on one CPU harm another?

### Takeaway
Only Bang for the Buck is a true multi-CPU graph-search study, and it tests no prefetch or layout setting. Reversals across CPUs are documented for SIMD and quantization choices. I found no published reversal of a prefetch setting across CPUs with the same code.

### Cited Findings
- **Bang for the Buck** (Kuffo and Boncz, DaMoN 2025) [passages]: six AWS CPUs at 8 vCPUs and 64 GB: Graviton3, Graviton4, Zen 3 (EPYC 7R13), Zen 4 (EPYC 9R14), Sapphire Rapids (8488C), SPR Z (6455B). Datasets are OpenAI/1536, arXiv/768, SIFT/128. FAISS IVF and USearch HNSW (m=16, ef_construction=128), single-threaded, with Table 4 at the highest possible recall. — [arXiv](https://arxiv.org/html/2505.07621v1)
- Bang, ranking reversal: "Zen4 is the clear winner" on IVF float32 with 3x both Intels, yet "Contrary to IVF indexes, Zen4 performs the worst" on HNSW. HNSW float32 QPS on OpenAI: SPR Z 228.7, Graviton3 192.0, SPR 182.0, Zen 4 165.6. — [arXiv](https://arxiv.org/html/2505.07621v1)
- Bang, technique that helps on one CPU and hurts on another: FAISS IVF on OpenAI going from float32 to SQ8 gives Zen 4 38.9 to 57.3 QPS, SPR 13.3 to 28.9, and Graviton3 27.4 to 9.6. The cause is software: FAISS "uses scalar code" on ARM for the decode. — [arXiv](https://arxiv.org/html/2505.07621v1)
- Bang, float16 in USearch HNSW: SIFT on SPR Z 959.7 to 2866.2 QPS, OpenAI on SPR 182.0 to 270.2, Zen 4 165.6 to 168.0 ("the Zens struggle due to the lack of float16 SIMD"). — [arXiv](https://arxiv.org/html/2505.07621v1)
- Bang, newer is not faster: "SVE is 37% faster than NEON on Graviton3", NEON and SVE are equal on Graviton4, and Graviton3 is 31% faster than Graviton4 with SVE (256-bit against 128-bit registers). — [arXiv](https://arxiv.org/html/2505.07621v1)
- Bang's stated limits: no multi-threading, and it does not vary prefetching (the text never mentions it in the passages searched). — [arXiv](https://arxiv.org/html/2505.07621v1)
- **Distance Comparison Operations Are Not Silver Bullets** (coordinator gives venue as ICDE 2026; arXiv 2604.02801) [passages]: "diverse hardware" means one CPU model, an AMD Ryzen 9 5950X, at varied SIMD levels, plus an RTX 3090. It is not a multi-CPU study. Enabling SIMD lifts DDCopq "from the least to the most efficient method" on SIFT, and shrinks DCO advantages from 2.0-4.4× to 1.2-3.8× on IVF and 1.9-3.1× to 1.1-2.1× on HNSW. No discussion of prefetching in the passages searched. — [arXiv](https://arxiv.org/html/2604.02801v1)
- **KBest** (KDD 2026) [passages]: compares different code on different CPUs: KBest on Kunpeng 920 (80C/160T, 2.9 GHz) against x86 libraries on an EPYC 9654 "overclocked to 3.7GHz". QPS at Recall@10=0.95, KBest against Vamana: Glove 170K/145K, Deep 628K/459K, Text-to-Image 113K/94K, BigANN 357K/305K. Its CPU comparison confounds library and hardware, so it cannot show a tuning reversal. — [arXiv](https://arxiv.org/html/2508.03016v2)
- **VIBE** [passages]: one platform, Xeon Gold 6230 (Cascade Lake), one thread, SMT off, local NUMA node. It says outright: "relative implementation performance may differ on other hardware", and calls repeating across architectures "valuable". — [arXiv](https://arxiv.org/abs/2505.17810)
- **PDX** [passages]: auto-vectorized kernels on several architectures (Zen 4, Zen 3, Graviton 4 and others), about 1.5x faster on average above 32 dimensions; it attributes differences to instruction latencies, register widths and cache sizes. This is exact and partition search, not graph traversal. — [arXiv](https://arxiv.org/abs/2503.04422v1)
- **VSAG** asserts prefetch parameters "exhibit substantial variance in optimal configurations in different testing environments" but its ablation table is one machine; see the next section. — [arXiv](https://arxiv.org/pdf/2503.17911v3)

### Inferences
- Published reversals come from kernel availability (ISA and library support). None comes from a memory-system setting on identical code.
- The citable gap is stated by the field itself: VIBE names it, and Bang lists only SIMD and data-access explanations.

### Gaps
- I found no paper running one engine with a varied prefetch setting on two or more CPUs. This is a search result, not proof of absence. Vendor reports and non-English venues were not swept.

## Software prefetching: settings, gains, cache lines, and first line versus whole vector

### Takeaway
Prefetching is established at roughly +15-20% alone and up to 2x when distance is tuned. Libraries disagree on how many lines to fetch. The only hard number found for first-line versus whole-vector is a 2026 pull request in Intel's SVS: 946 to 729 QPS on one unstated CPU.

### Cited Findings
- **VSAG** (PVLDB 18(12), 2025) [passages]: Xeon Platinum 8163, `-Ofast`, AVX-512. Table 5 cumulative QPS (GIST1M at 89.8% recall, SIFT1M at 98.4%):

| Step | GIST1M | SIFT1M | L3 miss rate GIST / SIFT |
|---|---|---|---|
| Baseline | 510 | 1695 | 93.89% / 77.88% |
| + Quantization | 1272 | 2881 | 67.42% / 52.09% |
| + Software prefetch | 1490 (+17%) | 3332 (+16%) | 71.71% / 53.86% |
| + Stride prefetch | 1517 (+2%) | 3565 (+7%) | 64.57% / 19.26% |
| + ELP auto-tuner | 2052 (+35%) | 4946 (+39%) | 45.88% / 32.65% |
| + Deterministic access | 2167 (+6%) | 5027 (+2%) | 39.23% / 20.98% |

Percentages are computed here from the table. — [arXiv](https://arxiv.org/pdf/2503.17911v3)
- VSAG on why: stride alone "shows minimal difference" because "prefetch efficiency heavily depends on environment-level parameters". The stride depends on CPU speed against prefetch latency, and the depth "is primarily determined by the CPU's native prefetch patterns combined with vector dimensionality". Tuning is a grid search on sampled base vectors. — [arXiv](https://arxiv.org/pdf/2503.17911v3)
- VSAG's current code differs from the paper: depth defaults to the whole code in lines, and HGraph's optimizer tunes only stride 1-10 (`TODO(ZXY): optimize PREFETCH_DEPTH_CODE`). [code] — [hgraph_build.cpp](https://github.com/antgroup/vsag/blob/main/src/algorithm/hgraph/hgraph_build.cpp)
- **Glass** [code]: defaults `po=1, pl=1`; `Optimize()` grid-searches `po` to min(20, K) and `pl` to min(20, code lines), then prints the gain against `po=pl=1`. No paper. — [graph_searcher.hpp](https://github.com/zilliztech/pyglass/blob/master/glass/searcher/graph_searcher.hpp)
- **NGT** [code]: `adjustPrefetchParameters` searches offset 1-10 and size in bytes up to min(4× object size, 64×28). — [GraphOptimizer.h](https://github.com/yahoojapan/NGT/blob/main/lib/NGT/GraphOptimizer.h)
- **hnswlib** [code]: prefetches one vector ahead, first line only. — [hnswalg.h](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h)
- **DiskANN** [code]: `prefetch_vector` fetches floor(size/64) lines, with a note that it works best on multiples of 64. — [utils.h](https://github.com/microsoft/DiskANN/blob/cpp_main/include/utils.h)
- **SVS** paper [passages]: Xeon Platinum 8380. Prefetching gives "up to 2x" over none; prefetch-step=1 "shows no performance gain"; "there is not a universally best combination"; offset=0, step=2 "works well in many cases". Parameters are in vectors, not lines. — [arXiv](https://arxiv.org/abs/2304.04759)
- **SVS** code [code]: prefetches ceil(bytes/64) lines, plus one extra when size is not a multiple of 64 and under 3 lines, because "stream prefetch kicks in after few consecutive accesses". — [prefetch.h](https://github.com/intel/ScalableVectorSearch/blob/main/include/svs/lib/prefetch.h)
- **Closest evidence on first line versus whole vector**: SVS PR #361 (merged 18 August 2026). GCC 12 dropped all but the first prefetch per vector. On cohere-768 fp16 inner product at recall 0.95 with 1 thread, QPS was 946 (gcc11), 729 (gcc12 stock), 925 (gcc12 with fix), about -23% from first-line-only. Redis end-to-end: 9332 to 8251 RPS. CPU not stated; recall bit-identical. [code/PR] — [PR 361](https://github.com/intel/ScalableVectorSearch/pull/361)
- Spalding-Jamieson et al. [passages]: name hnswlib's first-line-only prefetch as a flaw ("may be mitigated by smart compilers or hardware prefetchers"), default to 4 vectors ahead, and find "a number of values seemed to exhibit essentially the same performance". No first-line-versus-whole number in the passages read. — [arXiv](https://arxiv.org/pdf/2502.06163)
- **KBest** [passages]: batch B = floor(α·L1d/(d·s)), α typically 0.5; prefetch adds "more than 20%" on BigANN. — [arXiv](https://arxiv.org/html/2508.03016v2)
- **AverSearch** ("Efficient Graph-Based ANN Search Achieving Low Latency Without Throughput Loss") does not address prefetching [summary, keyword search of full text]. — [arXiv](https://arxiv.org/html/2504.20461v2)
- **Graph Reordering** (NeurIPS 2022) [passages]: Xeon E5-2697, 36 MB L3. "up to 40%". SIFT100M L1 misses 19.53% to 14.46%, TLB 3.85% to 2.14% (Gorder). No prefetch discussion found. — [arXiv](https://arxiv.org/pdf/2104.03221)
- Hardware background: the Intel manual's spatial prefetcher "strives to complete every cache line fetched to the L2 cache with the pair line that completes it to a 128-byte aligned chunk" [search]. — [quoted here](https://issues.dlang.org/bugs/20055/)
- MSR 0x1A4 bit 1 controls the L2 adjacent-line prefetcher [search]. — [Intel forum](https://community.intel.com/t5/Intel-ISA-Extensions/Disabling-HW-prefetcher/m-p/1176713) A SPEC platform note gives the BIOS default as enabled [search]. — [SPEC flags](https://www.spec.org/cpu2017/flags/NEC-Platform-Settings-V1.2-D120h-RevA.html)

### Inferences
- Cross-CPU sensitivity of prefetch is widely assumed: three auto-tuners exist because of it. It is asserted, not measured, in what I found.
- VSAG's largest prefetch gain comes from tuning, not from the mechanism, which supports treating the setting as hardware-dependent.
- First-line-only is plausibly fine where a hardware prefetcher fetches the rest, and costly where it does not. The SVS comment and PR bracket this, but neither names a CPU.

### Gaps
- VSAG's tuned stride and depth values per CPU: none reported in the sections read.
- The CPU behind SVS PR #361 is not stated.
- Fill-buffer counts on Skylake-SP from a primary source: not found. Golden Cove's 16 fill buffers and 48 outstanding misses are search-level only. — [AnandTech](https://at-web1.www.anandtech.com/show/16881/a-deep-dive-into-intels-alder-lake-microarchitectures/3)
- Intel's behaviour for software prefetches that miss the TLB: only forum-level material found.

## Memory as the bottleneck: where search time goes

### Takeaway
Fetch and compute are often lumped together as "distance", which explains the apparent conflict. Where they are separated, memory access is the larger share, and near-memory papers report 5× gains in simulation.

### Cited Findings
- VSAG [passages]: baseline "over 2,959 random vector accesses (total 1.4 MB), causing a 67.42% L3 cache miss rate. The memory-access operations consume 63.02% of the search time" (its Table 1 is GIST1M). — [arXiv](https://arxiv.org/pdf/2503.17911v3)
- ADSampling [passages]: DCOs take 77.2% to 87.6% of HNSW runtime at 256-960 dimensions, 95% recall, k=100. This includes fetching. — [arXiv](https://arxiv.org/pdf/2303.09855)
- SVS [passages]: with AVX, "search is heavily dominated by the memory access time"; hardware prefetchers "are ineffective" on graph access. — [arXiv](https://arxiv.org/abs/2304.04759)
- ANSMET (ISCA 2025) [passages]: FAISS on Xeon 5120; 617 vectors per query on HNSW-SIFT, over 300 kB per query; 50% to over 90% of comparisons rejected. Simulated NDP gives 5.26× on average, up to 6.40×, and 72.8% lower per-query latency than CPU-Base on SIFT. — [PDF](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/ansmet.isca25.pdf)
- Cosmos [passages]: "ANNS remains bottlenecked by memory bandwidth, primarily due to distance calculations loading large vectors". Its breakdown is figure-only (no numbers in text). — [arXiv](https://arxiv.org/html/2505.16096v1)
- Bang [passages]: past L3, sequential latency on SPR is "almost 2x higher than the Zens and almost 4x higher than the Gravitons". This is cited as the reason SPR's kernel lead shrinks on IVF. — [arXiv](https://arxiv.org/html/2505.07621v1)
- Flash [passages]: distance computation exceeds 90% of HNSW construction time on Xeon E5-2620 v3. — [arXiv](https://arxiv.org/abs/2502.18113)

### Inferences
- Claims of "distance dominates" and "memory dominates" are compatible. A paper should state which split it uses.
- Near-memory results are simulated against old CPUs, so they do not show what a modern CPU loses.

### Gaps
- No top-down (frontend, backend, memory-bound) analysis of HNSW across CPU generations was found.
- A search summary attributed "85.4% to 94.9%" to some paper, but it was not in the two PDFs checked. Not used.

## SIMD: AVX-512 frequency, VNNI, masked loads, AMX, FP16

### Takeaway
AVX-512 downclocking on Skylake and Cascade Lake is well established and largely gone from Ice Lake on. Masked loads are cheap in microbenchmarks, and I found no ANN work calling them a bottleneck. AMX exists in VSAG for contiguous scans, but no graph-search evaluation with numbers was found.

### Cited Findings
- Downs, Skylake W-2104 [summary]: L0 3.2, L1 2.8, L2 2.4 GHz; about 10 µs halt on a frequency transition; about 680 µs before the license relaxes; dispatch at 1/4 speed during voltage transition. — [blog](https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html)
- Downs, Ice Lake i5-1035G4 [summary]: "only 100 MHz of license-based downclock with 1 active core", 3,700 to 3,600 MHz, and none elsewhere. — [blog](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html)
- Cloudflare, Xeon Silver 4116 [summary]: AVX-512 at 1.8 GHz single-core, 1.4 GHz all-core; 10% fewer requests per second with ChaCha20-Poly1305. — [blog](https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/)
- OpenJDK, Cascade Lake [summary]: 3.1 to 2.7 GHz observed; SPECjvm xml.transform 776.00 to 894.07 ops/m moving to AVX2. The post says "~23%", which does not match 3.1 to 2.7. — [mail](https://mail.openjdk.org/pipermail/hotspot-dev/2022-May/060369.html)
- Bang [passages]: downclocking is "no longer an issue in modern CPUs (Zen4, Sapphire Rapids)". — [arXiv](https://arxiv.org/html/2505.07621v1)
- uops.info, measured [passages], load throughput in cycles and µops:

| Instruction | Cascade Lake | Ice Lake | Alder Lake-P | Zen 4 |
|---|---|---|---|---|
| VMOVDQU8 ymm{k}, m256 | 0.50, 2 (p015+p23) | 0.50, 2 | 0.40, 2 | 0.50, 1 |
| VMOVDQU32 ymm{k}, m256 | 0.50, 2 | 0.50, 2 | 0.38, 2 | 0.50, 1 |
| VMOVDQU64 ymm, m256 | 0.50, 1 (p23) | 0.50, 1 | 0.33, 1 | 0.50, 1 |
| VPDPBUSD ymm (EVEX) | 0.50, 1 (p01) | 0.50, 1 | 0.50, 1 | 0.50, 1 |
| VPDPBUSD zmm | 0.50, 1 (p05) | 1.00, 1 | 1.00, 1 | 1.00, 1 |

Cascade Lake's masked byte load uses no microcode (MS 0). — [VMOVDQU8](https://uops.info/html-instr/VMOVDQU8_YMM_K_M256.html), [VPDPBUSD](https://uops.info/html-instr/VPDPBUSD_ZMM_ZMM_ZMM.html)
- Peter Cordes on Skylake-AVX512: prefer `vmovdqu32`/`64` to `vmovdqu8` unless masking [summary]. — [GCC bug thread](https://gcc.gnu.org/ml/gcc-bugs/2017-10/msg00360.html)
- SimSIMD treats masked tails as a benefit: AVX-512 6.1M ops/s against AVX2 3.3M on 1535-d vectors, Xeon 8480+ [summary]. That is a whole-kernel comparison, not a tail measurement. — [post](https://ashvardanian.com/posts/simsimd-faster-scipy/)
- Fault suppression: masked-out elements raise no faults [search]. Nothing found on whether masked-out bytes in a split load still touch the second line. A possible lead, unread: — [AVX timing side channels](https://arxiv.org/pdf/2304.07940)
- AMX: CABANA reports up to 32.6× throughput over SIMD, by batching queries per cluster (IVF-style) [search]. — [DGIST](https://scholar.dgist.ac.kr/handle/20.500.11750/60390)
- VSAG ships an AMX SQ8 kernel: 16 codes in a tile via `_tile_dpbuud`, loaded with a row stride, used by the batch scan path. [code] — [amx.cpp](https://github.com/antgroup/vsag/blob/main/src/simd/amx.cpp)
- SVS's paper reports LVQ up to 2.12x faster than float16, and float16 at over 90% of peak memory bandwidth. — [arXiv](https://arxiv.org/abs/2304.04759)
- FP16 recall: SingleStore and pgvector report near-identical recall at half the memory [search]. — [SingleStore](https://www.singlestore.com/blog/float16-vector-type-support-in-singlestore-cheaper-faster-better/), [Supabase](https://supabase.com/blog/pgvector-0-7-0)

### Inferences
- A masked load costs about one extra ALU µop on Cascade Lake. That cannot by itself explain a 31% search slowdown, so a different mechanism must be involved (see the Epzero comparison).
- VNNI's 512-bit Cascade Lake throughput (p05) is better than Ice Lake's or Alder Lake's on paper, so kernel throughput does not explain Cascade Lake's weakness on short vectors.
- AMX tile loads need a regular stride, so graph search would need codes stored together, as in Flash or VSAG's redundant storage.

### Gaps
- Gold 6226 AVX-512 turbo table: WikiChip refused connections. Intel's spec page was seen only in search: 12 cores, 2.7 GHz base, 3.7 GHz turbo, 19.25 MB L3. — [Intel](https://www.intel.com/content/www/us/en/products/sku/193957/intel-xeon-gold-6226-processor-19-25m-cache-2-90-ghz/specifications.html)
- Agner Fog and the Intel optimization manual were not read directly.
- No AVX512-FP16 re-ranking study found.

## Multi-thread scaling and NUMA for in-memory vector search

### Takeaway
Memory bandwidth limits high-dimensional scaling, chiplet L3 layout now matters, and most HNSW papers avoid NUMA by staying on one socket. I found no local, remote and interleaved comparison for graph search on x86.

### Cited Findings
- SVS [passages]: Platinum 8380, 174 GB/s peak per socket. Search reaches 160 GB/s (float16) and 135 GB/s (LVQ-8). Scaling reaches 23.5x and 33x; float16 saturates at 40 threads as bandwidth approaches peak. Experiments stay on one socket to avoid NUMA. — [arXiv](https://arxiv.org/abs/2304.04759)
- AverSearch [passages]: Xeon 8457C, 48 cores, 266 GB/s by MLC. iQAN throughput falls 37.7%-76.7% from 1 to 24 intra-query threads. At 24×2, bandwidth is 108.39 GB/s, under 40% of maximum. Its own results are 2.1-8.9× throughput at comparable latency. — [arXiv](https://arxiv.org/abs/2504.20461)
- CCD paper (RedNote) [passages]: EPYC 9654 and 7K62, hnswlib- and FAISS-derived, 64-256 dimensions, M=32. Naive HNSW scaling gives 2× to 9.9× from 16 to 96 cores, 82% of ideal at 96. CCD-aware dispatch reaches over 100 KQPS at 96 cores, and up to 3.7× throughput with 30-90% lower P50 and P999. — [arXiv](https://arxiv.org/pdf/2605.10090)
- Quake [passages]: partitioned index, not graph: 20× and 4× lower latency against single-threaded and non-NUMA configurations on MSTURING100M. — [arXiv](https://arxiv.org/abs/2506.03437)
- UDON [passages]: FAISS HNSW on dual-socket Arm, with remote NUMA emulating CXL; offload up to 6.87×. — [arXiv](https://arxiv.org/pdf/2404.02868)
- VIBE and LeanVec also bind to one socket. — [VIBE](https://arxiv.org/abs/2505.17810), [LeanVec](https://arxiv.org/abs/2312.16335)

### Inferences
- Single-thread prefetch results may not carry to full-core runs, because bandwidth limits differ. Line-count settings should be tested under load.

### Gaps
- No multi-thread cross-CPU comparison of the same graph engine found. Bang lists it as future work.

## Index layout: separating vectors from links, and huge pages

### Takeaway
Layout effects are real but conditional. Co-locating data helps or hurts depending on dataset, and huge pages grow more valuable with scale.

### Cited Findings
- VSAG PRS [passages]: storing neighbour vectors redundantly lifts GIST1M from 2167 to 2377 in Table 5 (the text says 2337), and drops SIFT1M from 5027 to 4640, "because CPU pressure outweighs memory pressure". — [arXiv](https://arxiv.org/pdf/2503.17911v3)
- Flash [passages]: stores neighbour IDs followed by neighbour codewords, so distances use register tables and avoid random access. — [arXiv](https://arxiv.org/abs/2502.18113)
- Glass keeps graph and vectors in separate structures and prefetches each [code]; hnswlib stores them inline [code]. KBest uses CSR plus a flat vector table [passages]. — [Glass](https://github.com/zilliztech/pyglass), [KBest](https://arxiv.org/html/2508.03016v2)
- Huge pages, SVS [passages]: 20% on deep-96-100M, 90% on deep-96-1B at 0.90 recall. — [arXiv](https://arxiv.org/abs/2304.04759)
- KBest [passages] and Glass [code] use `madvise(MADV_HUGEPAGE)`. — [arXiv](https://arxiv.org/html/2508.03016v2)
- Epzero [Epzero]: +13% QPS on the laptop; BFS reordering gave 0% with 704-byte records, against up to 40% in Graph Reordering. — `docs/performance.md`

### Inferences
- Huge-page policy is a hidden benchmark variable. Epzero's lead was 1.44-1.55× on the laptop and 1.06-1.08× on the server with THP `always` [Epzero]. Cross-CPU comparisons should fix and report THP.

### Gaps
- No controlled study of inline against separated storage across CPUs was found.

## Comparison with Epzero findings

### Takeaway
The prefetch-line reversal is the strongest novelty candidate, and the new GloVe grid strengthens it. The masked-load finding should not be published as "masked loads cost 31%" until it is retested, because Epzero's record layout suggests a cache-line split instead.

### Cited Findings
**1. Prefetch lines, new grid [Epzero]: apparently new as a cross-CPU reversal; partly known otherwise.** One prefetched line was best on Raptor Lake under WSL2. On Cascade Lake, a GloVe grid measured 1 line at 0.60-0.75x Glass and 2 lines (the whole 104-byte record) at 1.02-1.21x.
- Known: prefetch settings are said to depend on environment. — [VSAG](https://arxiv.org/pdf/2503.17911v3)
- Known: first-line-only cost about 23% in one case. — [SVS PR 361](https://github.com/intel/ScalableVectorSearch/pull/361)
- Not found: the same code with the best line count reversing between CPUs.

**2. Masked 4-byte tail [Epzero]: observation apparently new; attribution unsafe.** The 256-bit VNNI kernel was 31% slower than AVX2 on 100 dimensions and 2.5% faster on 96. No ANN work reports this, and uops.info shows the load is cheap. — [uops.info](https://uops.info/html-instr/VMOVDQU8_YMM_K_M256.html)

**3. Batching and smaller beam items, laptop null [Epzero]: partly known.** KBest folds 1-to-B batching into a SIMD gain of 60% average and 92% maximum, on Kunpeng. — [arXiv](https://arxiv.org/html/2508.03016v2) Lucene says bulk scoring lets the scorer "prefetch and pipeline work", with no isolated figure [summary]. — [Elastic](https://www.elastic.co/search-labs/blog/apache-lucene-wrapped-2025) SVS prefers a sorted linear buffer to a heap, with no number. I found no isolated measurement of batching with unchanged vectors on x86, and nothing on beam item size.

**4. 16-bit-float re-ranking speed neutral, identical recall [Epzero]: mostly known.** Near-identical float16 recall is reported in industry. — [SingleStore](https://www.singlestore.com/blog/float16-vector-type-support-in-singlestore-cheaper-faster-better/) Speed depends on CPU: Bang shows large gains where f16 SIMD exists, none on Zen 4. — [arXiv](https://arxiv.org/html/2505.07621v1) I found nothing specific to re-ranking, but this is a weak novelty claim.

Also, the Epzero prefetch tuner picks 2 neighbours ahead on the Gold 6226 and all up front on the laptop, 13-17% and 15-17% faster on GloVe-100 and DEEP10M [Epzero]. This is a second reversal, in depth rather than lines.

### Inferences
- **Masked-tail mechanism (my analysis, untested).** In the SQ8 record, `[norm 4][codes][links]` gives links at byte 104 and a 256-byte stride for GloVe-100. The removed kernel's tail load was `_mm256_maskz_loadu_epi8(m, codes + 96)`, which is record bytes 100-131, so it crosses the line boundary at 128 on every scored node. On 96 dimensions there is no tail, so no such load exists. That matches the +2.5% and -31% split exactly. The third line holds links and is not in the 2-line prefetch, so the load may cause a split, a demand miss, or both, if masked-out bytes still access the line. The 512-bit kernel's tail appears to span bytes 68-131 on both 96 and 100 dimensions, which fits its 61-73% loss there and its 3-5% win at 960 and 1,536 dimensions, where no tail exists. `docs/performance.md` and the comment in `distance_avxvnni.cpp` currently attribute the cost to the masked load itself.
- **Test for this:** (a) use a 128-bit masked load at the same offset, which stays inside line 2; (b) move the norm after the codes; (c) count `mem_inst_retired.split_loads` and L1 misses per scored node. If the cost disappears, the finding is a layout and split effect, which is a more interesting and more defensible result.
- **Prefetch mechanism (hypothesis).** With 256-byte stride and 2 MiB alignment, line 2 is the 128-byte pair of line 1, so the L2 adjacent-line prefetcher could supply it on one CPU and not the other. Reading MSR 0x1A4 on both machines, or offsetting records so line 2 is not the pair, would separate this from latency and fill-buffer explanations. Under WSL2, nested paging is a further uncontrolled difference.
- **Safe wording for now:** "We are not aware of prior work showing the best software-prefetch line count reversing between CPUs on identical search code." Avoid "first" without the qualifier. Name two CPUs only, and do not attribute it to a microarchitecture until a third CPU is measured. The 2-line result is on a 104-byte record, so state it as whole-record against first-line.
- Ratios to Glass are relative to a binary compiled per CPU, so they mix Glass's own tuning with Epzero's. Absolute QPS should accompany them.

### Gaps
- Whether the 0.60-0.75x and 1.02-1.21x ranges pair at the same recall points. Paired values are needed before quoting a single ratio.
- Whether the GloVe grid includes huge-page state, thread count and the kernel in use. These should be reported with it.
- No third CPU result was available to me.
- Whether a masked-out split load touches the second line on Cascade Lake: no source found. The experiment above would settle it for this code.
