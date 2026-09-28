<script lang="ts">
  import { compare, operating, speedup } from "../lib/bench";
  import Lab from "../components/bench/Lab.svelte";
  import Optimizations from "../components/bench/Optimizations.svelte";
  import Quantization from "../components/bench/Quantization.svelte";
  import RecallChart, { type Series } from "../components/bench/RecallChart.svelte";
  import { reveal } from "../lib/reveal";
  import { router } from "../lib/router.svelte";

  const libs = compare.libraries;
  const series: Series[] = [
    { name: "Oryx", color: "var(--accent)", points: libs.oryx.sweep },
    { name: "FAISS", color: "var(--series-3)", points: libs.faiss.sweep },
    { name: "hnswlib", color: "var(--series-2)", points: libs.hnswlib.sweep },
  ];

  const vsHnswlib = speedup("hnswlib");
  const vsFaiss = speedup("faiss");
  const builds = [
    { name: "Oryx", s: libs.oryx.build_seconds, accent: true },
    { name: "hnswlib", s: libs.hnswlib.build_seconds },
    { name: "FAISS", s: libs.faiss.build_seconds },
  ];
  const maxBuild = Math.max(...builds.map((b) => b.s));

  const removed = [
    {
      idea: "BFS graph reordering",
      result: "0%",
      why: "Node records are 704 bytes, so neighbours never share cache lines, and huge pages had already fixed the TLB misses.",
    },
    {
      idea: "Exact early abandoning",
      result: "−10%",
      why: "Lossless, but the horizontal sum at each checkpoint breaks the FMA pipeline, and the prefetcher already pulled the lines it would skip.",
    },
    { idea: "LTO and PGO", result: "±2%", why: "The code is memory-bound; better branch layout doesn't shorten DRAM latency." },
    {
      idea: "Chunked snapshot I/O",
      result: "0.31 s → 0.31 s",
      why: "Loading was dominated by allocating and zeroing 764 MB, not by parsing. Kept for its error handling, without claiming a speedup.",
    },
  ];
</script>

<div class="bench">
  <header class="head container">
    <p class="eyebrow">Benchmarks</p>
    <h1><span class="gradient-text">Faster than the libraries</span><br />it's measured against.</h1>
    <p class="section-lede">
      SIFT1M, the standard ANN benchmark: {compare.n.toLocaleString()} × {compare.dim}-D vectors, {compare.queries.toLocaleString()}
      queries, exact ground truth. Every library builds the same graph (M = {compare.M}, efConstruction = {compare.ef_construction}) and
      is queried from one process on one thread, in {compare.rounds} interleaved rounds.
    </p>
    <dl class="stats">
      <div><dt>vs hnswlib</dt><dd class="num">{vsHnswlib.toFixed(2)}×</dd><span>QPS at equal recall ({operating.recall.toFixed(3)})</span></div>
      <div><dt>vs FAISS HNSW</dt><dd class="num">{vsFaiss.toFixed(2)}×</dd><span>QPS at equal recall ({operating.recall.toFixed(3)})</span></div>
      <div><dt>SQ8 + re-rank</dt><dd class="num">+48%</dd><span>QPS, identical recall</span></div>
      <div><dt>Graph build</dt><dd class="num">{libs.oryx.build_seconds.toFixed(1)} s</dd><span>on {compare.build_threads} threads</span></div>
    </dl>
  </header>

  <section class="container block" aria-labelledby="curve-title">
    <div class="card panel" use:reveal>
      <div class="panel-head">
        <div>
          <h2 id="curve-title">Recall vs. throughput</h2>
          <p>Up and to the right is better. Hover a point to compare libraries at the same <code>ef</code>.</p>
        </div>
      </div>
      <RecallChart {series} label="Recall versus queries per second on SIFT1M for Oryx, FAISS and hnswlib; Oryx is highest at every recall" />
      <details>
        <summary>Show the data</summary>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>ef</th>{#each series as s (s.name)}<th>{s.name} recall</th><th>{s.name} QPS</th>{/each}</tr>
            </thead>
            <tbody>
              {#each libs.oryx.sweep as row, i (row.ef)}
                <tr>
                  <td>{row.ef}</td>
                  {#each series as s (s.name)}
                    <td>{s.points[i].recall.toFixed(4)}</td><td>{Math.round(s.points[i].qps).toLocaleString()}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </details>
    </div>
    <p class="caption" use:reveal>
      All three implement the same algorithm, and graph quality is essentially identical: recall matches at every <code>ef</code>. The
      difference is memory-system engineering, mainly batched prefetching and huge pages, which hnswlib and FAISS don't use by default.
    </p>
  </section>

  <section class="container block two" aria-label="Where the speed comes from">
    <div class="card panel" use:reveal>
      <h2>Where the 31% came from</h2>
      <p class="sub">Profiling showed search is bound by memory latency (~63 ns per distance), not arithmetic. So each change targets memory.</p>
      <Optimizations />
    </div>
    <div class="card panel" use:reveal={0.1}>
      <h2>Build time</h2>
      <p class="sub">1M inserts on {compare.build_threads} threads, same graph parameters.</p>
      <div class="builds">
        {#each builds as b (b.name)}
          <div class="build-row">
            <span>{b.name}</span>
            <span class="track"><i class:accent={b.accent} style:width="{(b.s / maxBuild) * 100}%"></i></span>
            <b class="num">{b.s.toFixed(1)} s</b>
          </div>
        {/each}
      </div>
      <p class="sub small">
        Parallel inserts take fine-grained per-node locks; the entry point and max level are updated under a separate lock only when
        a node lands on a new top layer.
      </p>
    </div>
  </section>

  <section class="container block" aria-labelledby="sq8-title">
    <div class="card panel" use:reveal>
      <h2 id="sq8-title">8-bit scalar quantization</h2>
      <p class="sub">
        Codes are scored with an asymmetric kernel (the query stays in float), then the beam is re-ranked with full-precision vectors
        when they're kept.
      </p>
      <Quantization />
    </div>
  </section>

  <section class="container block" aria-labelledby="removed-title">
    <h2 id="removed-title" class="block-title" use:reveal>Tried and removed</h2>
    <p class="caption lead" use:reveal>A change was kept only if the interleaved A/B showed a gain beyond ~3% noise. These didn't.</p>
    <div class="removed">
      {#each removed as r, i (r.idea)}
        <article class="card" use:reveal={i * 0.06}>
          <header><strong>{r.idea}</strong><span class="num">{r.result}</span></header>
          <p>{r.why}</p>
        </article>
      {/each}
    </div>
  </section>

  <section class="container block" id="lab" aria-labelledby="lab-title">
    <div use:reveal>
      <p class="eyebrow">Run it yourself</p>
      <h2 id="lab-title" class="block-title">Benchmark Oryx on your device.</h2>
      <p class="caption lead">
        The same C++ engine, compiled to WebAssembly with threads and SIMD, generates a clustered dataset, computes exact ground truth,
        builds the graph on every core and sweeps <code>ef</code>. Nothing leaves your browser.
      </p>
    </div>
    <Lab />
  </section>

  <section class="container block method" aria-labelledby="method-title">
    <h2 id="method-title" class="block-title" use:reveal>Methodology</h2>
    <ul use:reveal>
      <li><strong>Hardware.</strong> Intel i5-13420H laptop (P- and E-cores) under WSL2. Hyper-V hides core placement, so runs are interleaved rather than pinned.</li>
      <li><strong>Same graph for every search change.</strong> Built once, saved, and reloaded by every candidate binary, so recall is identical by construction.</li>
      <li><strong>Interleaved A/B.</strong> Candidates run in alternating order for 5–7 rounds; the median is reported. Differences under ~3% are treated as noise.</li>
      <li><strong>Library comparison.</strong> All three libraries are loaded into one process and queried through their own C++ batch loops, so Python overhead isn't timed.</li>
    </ul>
    <p class="caption">
      Want to see it on real data? <a href={router.href("try")}>Search 29,825 photos in the playground →</a>
    </p>
  </section>
</div>

<style>
  .bench {
    padding: calc(var(--nav-h) + 72px) 0 40px;
  }
  .head h1 {
    font-size: clamp(38px, 6vw, 72px);
    letter-spacing: -0.045em;
    line-height: 1.02;
  }
  .stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 48px 0 0;
  }
  .stats div {
    display: grid;
    gap: 2px;
    padding: 18px 20px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
  }
  dt {
    color: var(--text-2);
    font-size: 13px;
  }
  dd {
    margin: 0;
    font-size: 36px;
    font-weight: 700;
    letter-spacing: -0.04em;
  }
  .stats div:first-child dd {
    background: linear-gradient(90deg, var(--accent-strong), var(--accent-2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .stats span {
    color: var(--muted);
    font: 12px var(--mono);
  }
  .block {
    margin-top: clamp(56px, 8vw, 96px);
  }
  .panel {
    padding: clamp(18px, 3vw, 28px);
    background: color-mix(in srgb, var(--surface-solid) 72%, transparent);
  }
  .panel h2 {
    font-size: 22px;
  }
  .panel-head {
    margin-bottom: 18px;
  }
  .panel-head p,
  .sub {
    margin: 6px 0 18px;
    color: var(--text-2);
    font-size: 14px;
  }
  .sub.small {
    margin: 16px 0 0;
    color: var(--muted);
    font-size: 13px;
  }
  .caption {
    max-width: 760px;
    margin: 18px 0 0;
    color: var(--text-2);
    font-size: 15px;
    line-height: 1.65;
  }
  .caption.lead {
    margin: 10px 0 24px;
  }
  .block-title {
    font-size: clamp(26px, 3.4vw, 38px);
  }
  .two {
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 16px;
    align-items: start;
  }
  .builds {
    display: grid;
    gap: 12px;
  }
  .build-row {
    display: grid;
    grid-template-columns: 70px 1fr 60px;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    color: var(--text-2);
  }
  .build-row b {
    color: var(--text);
    font: 600 13.5px var(--mono);
    text-align: right;
  }
  .track {
    height: 26px;
    border-radius: 8px;
    background: var(--surface-2);
    overflow: hidden;
  }
  .track i {
    display: block;
    height: 100%;
    border-radius: 8px;
    background: var(--axis);
  }
  .track i.accent {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
  }
  details {
    margin-top: 16px;
  }
  summary {
    color: var(--text-2);
    font-size: 14px;
    cursor: pointer;
  }
  .table-wrap {
    margin-top: 12px;
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font: 12.5px var(--mono);
  }
  th,
  td {
    padding: 7px 10px;
    border-bottom: 1px solid var(--border);
    text-align: right;
    white-space: nowrap;
  }
  th {
    color: var(--muted);
    font-weight: 500;
  }
  th:first-child,
  td:first-child {
    text-align: left;
  }
  .removed {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
  }
  .removed article {
    padding: 20px;
  }
  .removed header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 8px;
  }
  .removed header span {
    color: var(--warn);
    font: 600 13px var(--mono);
    white-space: nowrap;
  }
  .removed p {
    margin: 0;
    color: var(--text-2);
    font-size: 14px;
    line-height: 1.6;
  }
  .method ul {
    display: grid;
    gap: 12px;
    max-width: 820px;
    margin: 20px 0 0;
    padding-left: 20px;
    color: var(--text-2);
    line-height: 1.6;
  }
  .method strong {
    color: var(--text);
  }
  @media (max-width: 900px) {
    .stats {
      grid-template-columns: 1fr 1fr;
    }
    .two,
    .removed {
      grid-template-columns: 1fr;
    }
    dd {
      font-size: 30px;
    }
  }
</style>
