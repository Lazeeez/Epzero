<script lang="ts">
  import {
    DRIFT_POLICIES,
    FILTER_RECALL,
    FILTER_STRATEGIES,
    MACHINE,
    churn,
    compare,
    drift,
    driftEstimates,
    driftSummary,
    field,
    filterTable,
    operating,
    speedup,
  } from "../lib/bench";
  import HeroRace from "../components/bench/HeroRace.svelte";
  import { countUp } from "../lib/count-up";
  import { spotlight } from "../lib/spotlight";
  import DriftChart from "../components/bench/DriftChart.svelte";
  import FieldChart from "../components/bench/FieldChart.svelte";
  import Lab from "../components/bench/Lab.svelte";
  import Optimizations from "../components/bench/Optimizations.svelte";
  import Quantization from "../components/bench/Quantization.svelte";
  import RecallChart, { type Series } from "../components/bench/RecallChart.svelte";
  import { reveal } from "../lib/reveal";
  import { router } from "../lib/router.svelte";

  const libs = compare.libraries;
  const series: Series[] = [
    { name: "Epzero", color: "var(--accent)", points: libs.epzero.sweep },
    { name: "FAISS", color: "var(--series-3)", points: libs.faiss.sweep },
    { name: "hnswlib", color: "var(--series-2)", points: libs.hnswlib.sweep },
  ];

  const vsHnswlib = speedup("hnswlib");
  const vsFaiss = speedup("faiss");
  const builds = [
    { name: "Epzero", s: libs.epzero.build_seconds, accent: true },
    { name: "hnswlib", s: libs.hnswlib.build_seconds },
    { name: "FAISS", s: libs.faiss.build_seconds },
  ];
  const maxBuild = Math.max(...builds.map((b) => b.s));
  const maxDriftCost = Math.max(...driftSummary.map((p) => p.cost));

  const [quiet, busy] = churn.mixed;
  const cycles = churn.cycles.filter((c) => !c.compacted);
  const worn = cycles[cycles.length - 1];
  const compacted = churn.cycles.find((c) => c.compacted)!;
  const pct = (x: number) => `${(x * 100).toFixed(x < 0.01 ? 1 : 0)}%`;

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
    <div class="hero-art">
      <HeroRace
        recall={operating.recall}
        lanes={[
          { name: "Epzero", qps: operating.qps, lead: true },
          { name: "FAISS", qps: operating.qps / vsFaiss },
          { name: "hnswlib", qps: operating.qps / vsHnswlib },
        ].sort((a, b) => b.qps - a.qps)}
      />
    </div>
    <p class="eyebrow">Benchmarks</p>
    <h1><span class="gradient-text">Faster than hnswlib and FAISS.</span><br />Measured against the field.</h1>
    <p class="section-lede">
      SIFT1M, the standard ANN benchmark: {compare.n.toLocaleString()} × {compare.dim}-D vectors, {compare.queries.toLocaleString()}
      queries, exact ground truth. Every library builds the same graph (M = {compare.M}, efConstruction = {compare.ef_construction}) and
      is queried from one process on one thread, in {compare.rounds} interleaved rounds, on a dedicated server node ({MACHINE}).
    </p>
    <dl class="stats">
      <div use:spotlight><dt>vs hnswlib</dt><dd class="num" use:countUp>{vsHnswlib.toFixed(2)}×</dd><dd class="note">QPS at equal recall ({operating.recall.toFixed(3)})</dd></div>
      <div use:spotlight><dt>vs FAISS HNSW</dt><dd class="num" use:countUp>{vsFaiss.toFixed(2)}×</dd><dd class="note">QPS at equal recall ({operating.recall.toFixed(3)})</dd></div>
      <div use:spotlight><dt>SQ8 + re-rank</dt><dd class="num" use:countUp>+48%</dd><dd class="note">QPS, identical recall</dd></div>
      <div use:spotlight><dt>Graph build</dt><dd class="num" use:countUp>{libs.epzero.build_seconds.toFixed(1)} s</dd><dd class="note">on {compare.build_threads} threads</dd></div>
    </dl>
  </header>

  <section class="container block" aria-labelledby="curve-title">
    <div class="card panel" use:reveal use:spotlight>
      <div class="panel-head">
        <div>
          <h2 id="curve-title">Recall vs. throughput</h2>
          <p>Up and to the right is better. Hover a point to compare libraries at the same <code>ef</code>.</p>
        </div>
      </div>
      <RecallChart {series} label="Recall versus queries per second on SIFT1M for Epzero, FAISS and hnswlib; Epzero is highest at every recall" />
      <details>
        <summary>Show the data</summary>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>ef</th>{#each series as s (s.name)}<th>{s.name} recall</th><th>{s.name} QPS</th>{/each}</tr>
            </thead>
            <tbody>
              {#each libs.epzero.sweep as row, i (row.ef)}
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
      difference is memory-system engineering: batched prefetching, the node layout and the visited set. On this server every library
      gets 2 MiB huge pages, because the kernel enables them for all programs. Where huge pages are opt-in, as on many desktops, only
      Epzero asks for them, and its lead grows to about 1.46× (measured on a laptop).
    </p>
  </section>

  <section class="container block" aria-labelledby="field-title">
    <h2 id="field-title" class="block-title" use:reveal>Against the field</h2>
    <p class="caption lead" use:reveal>
      Four more ann-benchmarks datasets, from 96 to 1,536 dimensions and up to 10 million vectors, against five other libraries: hnswlib,
      FAISS (HNSW and a tuned IVF-PQ), USearch, VSAG and Glass. Same rules as above: the same graph degree (M = 16) and build beam
      (200), one process, one search thread, the median of three interleaved rounds, on the same server.
    </p>
    <div class="card panel" use:reveal use:spotlight>
      <FieldChart />
      <details>
        <summary>Show the data</summary>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Library</th>
                {#each field as d (d.key)}<th>{d.name} QPS (recall {d.recall.toFixed(2)})</th><th>{d.name} build</th>{/each}
              </tr>
            </thead>
            <tbody>
              {#each field[0].entries as lib (lib.id)}
                <tr>
                  <td>{lib.name}</td>
                  {#each field as d (d.key)}
                    {@const e = d.entries.find((x) => x.id === lib.id)!}
                    <td>{e.qps !== null ? Math.round(e.qps).toLocaleString() : "—"}</td><td>{e.build.toFixed(0)} s</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </details>
    </div>
    <p class="caption" use:reveal>
      Epzero SQ8 is the fastest of the eight on GIST1M (1.08× Glass at recall 0.95) and DEEP10M (1.15× Glass). Glass leads on GloVe
      (1.18×) and DBpedia's OpenAI embeddings (1.31×); its runs vary by about 10%, as it rebuilds its graph each time. Like Epzero SQ8 and VSAG, it walks the graph on 8-bit codes and re-ranks with the full vectors; it also tunes
      its prefetch distance to the machine before searching. Epzero is ahead of hnswlib, FAISS, VSAG and USearch on all four.
      Uncompressed, it is level with hnswlib on the high-dimensional sets (GIST, DBpedia: within 2%) and ahead on the low-dimensional
      ones (DEEP10M, GloVe: 7–15%), where the memory work around each distance matters most. GloVe is compared at recall 0.90: at
      M = 16 no library but VSAG reaches 0.95 by <code>ef</code> = 800. How the engine got here, and what didn't help:
      <a href={router.href("blog")}>Vector search is a memory problem</a>.
    </p>
  </section>

  <section class="container block" aria-labelledby="drift-title">
    <h2 id="drift-title" class="block-title" use:reveal>Recall that holds as the workload drifts</h2>
    <p class="caption lead" use:reveal>
      An approximate index trades accuracy for speed, and in production nobody measures the accuracy. Here, {drift.batch.toLocaleString()}
      queries at a time stream through four kinds of drift on SIFT1M: steady traffic, harder queries, 20% of the vectors deleted and
      re-inserted, and a 5% filter. A beam width tuned once falls below the {drift.target} target without anyone noticing. The recall
      guard measures recall on a {(drift.sample_rate * 100).toFixed(0)}% sample of queries and moves the beam width to hold the target.
    </p>
    <dl class="stats drift" use:reveal>
      <div use:spotlight>
        <dt>Phases at target</dt>
        <dd class="num" use:countUp>{driftSummary[0].adapted} of {driftSummary[0].phases}</dd>
        <dd class="note">guard, once adapted · tuned-once: {driftSummary[1].adapted} of {driftSummary[1].phases}</dd>
      </div>
      <div use:spotlight>
        <dt>Cost vs over-provisioning</dt>
        <dd class="num" use:countUp>−{((1 - driftSummary[0].cost / driftSummary[2].cost) * 100).toFixed(0)}%</dd>
        <dd class="note">distances per query, whole stream</dd>
      </div>
      <div use:spotlight>
        <dt>Guard's own estimate</dt>
        <dd class="num" use:countUp>±{driftEstimates.error.toFixed(3)}</dd>
        <dd class="note">mean error · truth inside its interval {driftEstimates.covered}/{driftEstimates.total}</dd>
      </div>
    </dl>
    <div class="card panel" use:reveal use:spotlight>
      <div class="panel-head">
        <h2>Recall per batch, by policy</h2>
        <p>Hover a batch to compare the policies; the shaded band is the interval the guard measured for itself.</p>
      </div>
      <DriftChart
        batches={drift.batches}
        series={DRIFT_POLICIES}
        target={drift.target}
        label="Recall per batch of queries through steady, harder, churn and filtered phases: the fixed beam tuned once falls below the target in two phases; the recall guard holds it after adapting"
      />
      <div class="drift-cost">
        <h3>Search cost</h3>
        {#each driftSummary as p (p.policy)}
          <div class="build-row wide">
            <span>{p.name}</span>
            <span class="track"><i class:accent={p.policy === "guard"} style:width="{(p.cost / maxDriftCost) * 100}%"></i></span>
            <b class="num">{Math.round(p.cost).toLocaleString()}</b>
          </div>
        {/each}
        <p class="sub small">Mean distance computations per query over the whole stream. {MACHINE}.</p>
      </div>
    </div>
  </section>

  <section class="container block" aria-labelledby="filter-title">
    <h2 id="filter-title" class="block-title" use:reveal>Filtered search: no single strategy wins</h2>
    <p class="caption lead" use:reveal>
      "Only results from this category" turns a graph search into a different problem at every filter size. Each strategy's best
      throughput at recall {FILTER_RECALL} on SIFT1M, by the share of vectors the filter allows. Scanning wins when few pass, walking
      past the excluded ones wins when most pass, and the two-hop walk wins in between. It falls apart below about 1.5%, never
      reaching the target. The engine's automatic choice follows this table.
    </p>
    <div class="card panel" use:reveal use:spotlight>
      <div class="table-wrap">
        <table class="filter-table">
          <caption class="visually-hidden">Queries per second at recall {FILTER_RECALL}, by filter selectivity and strategy</caption>
          <thead>
            <tr>
              <th scope="col">Filter allows</th>
              {#each FILTER_STRATEGIES as s (s.strategy)}<th scope="col">{s.name}</th>{/each}
            </tr>
          </thead>
          <tbody>
            {#each filterTable as row (row.selectivity)}
              <tr>
                <th scope="row">{pct(row.selectivity)} <span>{row.allowed.toLocaleString()}</span></th>
                {#each FILTER_STRATEGIES as s (s.strategy)}
                  {@const r = row.best[s.strategy]}
                  <td class:win={row.winner === s.strategy}>
                    {#if r}{Math.round(r.qps).toLocaleString()}<small>ef {r.ef}</small>{:else}<em>never</em>{/if}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      <p class="sub small">Queries per second on one thread, best over beam widths 10–320; the fastest per row is highlighted. {MACHINE}.</p>
    </div>
  </section>

  <section class="container block" aria-labelledby="churn-title">
    <h2 id="churn-title" class="block-title" use:reveal>Searching while the data changes</h2>
    <p class="caption lead" use:reveal>
      Reads never take a lock, so writers slow them only by competing for memory. {churn.readers} search threads ran
      for 30 seconds alone, then alongside 2 threads deleting and re-inserting vectors. Separately, a fresh index went through five
      cycles of deleting and re-inserting 10% of its vectors.
    </p>
    <dl class="stats drift" use:reveal>
      <div use:spotlight>
        <dt>Throughput under writes</dt>
        <dd class="num" use:countUp>{pct(busy.search_qps / quiet.search_qps)}</dd>
        <dd class="note">
          of read-only, with {Math.round(busy.writes_per_s).toLocaleString()} writes/s · recall {quiet.recall.toFixed(3)} → {busy.recall.toFixed(3)}
        </dd>
      </div>
      <div use:spotlight>
        <dt>p99 latency under writes</dt>
        <dd class="num" use:countUp>{Math.round(busy.p99_us)} µs</dd>
        <dd class="note">from {Math.round(quiet.p99_us)} µs read-only · p99.9 {Math.round(busy.p999_us)} µs</dd>
      </div>
      <div use:spotlight>
        <dt>After 5 churn cycles</dt>
        <dd class="num" use:countUp>−{((cycles[0].recall - worn.recall) * 100).toFixed(1)} pt</dd>
        <dd class="note">
          recall, and −{pct(1 - worn.qps / cycles[0].qps)} speed, with {pct(worn.tombstone_share)} tombstones · a {churn.compaction_seconds.toFixed(0)} s
          compaction restores {compacted.recall.toFixed(3)}
        </dd>
      </div>
    </dl>
  </section>

  <section class="container block two" aria-label="Where the speed comes from">
    <div class="card panel" use:reveal use:spotlight>
      <h2>Where the 31% came from</h2>
      <p class="sub">
        Profiling showed search is bound by memory latency (~63 ns per distance), not arithmetic. So each change targets memory.
        Measured on the development laptop, one search thread.
      </p>
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
    <div class="card panel" use:reveal use:spotlight>
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
        <article class="card" use:reveal={i * 0.06} use:spotlight>
          <header><strong>{r.idea}</strong><span class="num">{r.result}</span></header>
          <p>{r.why}</p>
        </article>
      {/each}
    </div>
  </section>

  <section class="container block" id="lab" aria-labelledby="lab-title">
    <div use:reveal>
      <p class="eyebrow">Run it yourself</p>
      <h2 id="lab-title" class="block-title">Benchmark Epzero on your device.</h2>
      <p class="caption lead">
        The same C++ engine, compiled to WebAssembly with threads and SIMD, takes a clustered dataset it generates or a file of your own
        embeddings, computes exact ground truth, builds the graph on every core and sweeps <code>ef</code>. Nothing leaves your browser.
      </p>
    </div>
    <Lab />
    <p class="caption next">
      Want to see it on real data? <a href={router.href("try")}>Search 29,825 photos in the playground →</a>
    </p>
  </section>

</div>

<style>
  .bench {
    padding: calc(var(--nav-h) + 72px) 0 40px;
  }
  .head {
    position: relative;
  }
  .hero-art {
    position: absolute;
    top: 64px;
    right: 0;
    width: 340px;
  }
  @media (max-width: 1240px) {
    .hero-art {
      display: none;
    }
  }
  /* Beside the race, the headline and intro keep to their own column (the race is positioned out
     of the flow, so nothing else stops a longer headline running under it). */
  @media (min-width: 1241px) {
    .head h1,
    .head .section-lede {
      max-width: calc(100% - 340px - 56px);
    }
    .bench .head h1 {
      font-size: clamp(48px, 4.4vw, 56px);
    }
  }
  .head h1 {
    font-size: clamp(38px, 6vw, 72px);
    letter-spacing: -0.045em;
    line-height: 1.02;
    text-wrap: balance;
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
  .stats div:first-child dd:not(.note) {
    background: linear-gradient(90deg, var(--accent-strong), var(--accent-2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .stats dd.note {
    color: var(--muted);
    font: 12px var(--mono);
    letter-spacing: 0;
  }
  .block {
    margin-top: clamp(56px, 8vw, 96px);
  }
  .stats.drift {
    grid-template-columns: repeat(3, 1fr);
    margin: 0 0 16px;
  }
  .drift-cost {
    display: grid;
    gap: 10px;
    max-width: 640px;
    margin-top: 28px;
  }
  .drift-cost h3 {
    margin: 0 0 4px;
    font-size: 15px;
  }
  .build-row.wide {
    grid-template-columns: 150px 1fr 60px;
  }
  @media (max-width: 640px) {
    .stats.drift {
      grid-template-columns: 1fr;
    }
    .build-row.wide {
      grid-template-columns: 112px 1fr 52px;
      font-size: 13px;
    }
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
  .caption.next {
    margin-top: 28px;
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
  .filter-table th[scope="row"] {
    color: var(--text);
    font-weight: 600;
  }
  .filter-table th[scope="row"] span {
    margin-left: 6px;
    color: var(--muted);
    font-weight: 400;
  }
  .filter-table td small {
    margin-left: 6px;
    color: var(--muted);
    font-size: 11px;
  }
  .filter-table td em {
    color: var(--muted);
    font-style: normal;
  }
  .filter-table td.win {
    color: var(--accent-strong);
    font-weight: 700;
    background: var(--accent-wash);
  }
  @media (max-width: 640px) {
    .filter-table td small,
    .filter-table th[scope="row"] span {
      display: none;
    }
    .filter-table th,
    .filter-table td {
      padding: 7px 5px;
    }
    .filter-table thead th {
      white-space: normal;
      vertical-align: bottom;
    }
  }
  .visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
    white-space: nowrap;
  }
</style>
