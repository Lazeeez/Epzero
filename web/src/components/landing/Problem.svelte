<script lang="ts">
  import { engine } from "../../lib/engine/engine.svelte";
  import { agreement } from "../../lib/engine/trace-analysis";
  import { reveal } from "../../lib/reveal";

  // Measured on this device by the hero's live queries: HNSW and an exact scan, same query.
  const trace = $derived(engine.last?.trace ?? null);
  const speedup = $derived(trace ? trace.exactMicros / Math.max(trace.micros, 1) : 0);
  const share = $derived(trace && engine.count ? trace.distanceComputations / engine.count : 0);
  let shown = $state(false);

  const time = (us: number) => (us >= 1000 ? `${(us / 1000).toFixed(1)} ms` : `${Math.round(us)} µs`);
</script>

<section class="problem section" id="problem" aria-labelledby="problem-title">
  <div class="container grid">
    <div use:reveal>
      <p class="eyebrow">The problem</p>
      <h2 id="problem-title" class="section-title">Search by meaning is a nearest-neighbour problem.</h2>
      <p class="section-lede">
        Modern search, recommendations and RAG all work the same way: embed everything as vectors, then find the vectors
        closest to the query's. The exact answer means comparing against every vector. At scale, that's too slow, so every
        vector database answers approximately, and the engineering is in how close and how fast.
      </p>
    </div>

    <div class="race card" use:reveal={0.12} {@attach (node: HTMLElement) => {
      const observer = new IntersectionObserver(([e]) => e.isIntersecting && ((shown = true), observer.disconnect()), { threshold: 0.4 });
      observer.observe(node);
      return () => observer.disconnect();
    }}>
      <header>
        <span>Measured just now, in this tab</span>
        <span class="muted num">{engine.count ? engine.count.toLocaleString() : "29,825"} × 512-D</span>
      </header>
      {#if trace}
        <div class="row">
          <div class="label"><strong>Exact scan</strong><span class="num">{engine.count.toLocaleString()} comparisons</span></div>
          <div class="bar"><i class="exact" style:width={shown ? "100%" : "0%"}></i></div>
          <span class="value num">{time(trace.exactMicros)}</span>
        </div>
        <div class="row">
          <div class="label">
            <strong class="accent">Oryx HNSW</strong><span class="num">{trace.distanceComputations.toLocaleString()} comparisons</span>
          </div>
          <div class="bar"><i class="hnsw" style:width={shown ? `${Math.max((100 * trace.micros) / trace.exactMicros, 0.6)}%` : "0%"}></i></div>
          <span class="value num">{time(trace.micros)}</span>
        </div>
        <dl class="facts num">
          <div><dt>Faster</dt><dd>{speedup.toFixed(0)}×</dd></div>
          <div><dt>Vectors compared</dt><dd>{(share * 100).toFixed(1)}%</dd></div>
          <div><dt>Same top {trace.results.length}</dt><dd>{agreement(trace)}/{trace.results.length}</dd></div>
        </dl>
      {:else}
        <p class="waiting">
          {engine.status === "error" ? engine.error : `${engine.stage || "Starting the engine"}… the comparison appears once the index is built.`}
        </p>
      {/if}
      <footer>
        On SIFT1M (1M × 128-D), an exact scan is 128 million multiply-adds per query. Oryx answers in about 0.1 ms per query
        on one core at 97% recall.
      </footer>
    </div>
  </div>
</section>

<style>
  .problem {
    background: linear-gradient(180deg, transparent, color-mix(in srgb, var(--surface-2) 40%, transparent) 50%, transparent);
  }
  .grid {
    display: grid;
    grid-template-columns: 1.05fr 1fr;
    align-items: center;
    gap: clamp(32px, 6vw, 88px);
  }
  .race {
    display: grid;
    gap: 18px;
    padding: 24px;
  }
  header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    color: var(--text-2);
    font: 12.5px var(--mono);
  }
  .muted {
    color: var(--muted);
  }
  .row {
    display: grid;
    grid-template-columns: 150px 1fr 70px;
    align-items: center;
    gap: 14px;
  }
  .label {
    display: grid;
    font-size: 14px;
  }
  .label span {
    color: var(--muted);
    font-size: 12px;
  }
  .accent {
    color: var(--accent-strong);
  }
  .bar {
    height: 12px;
    border-radius: 99px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .bar i {
    display: block;
    height: 100%;
    border-radius: 99px;
    transition: width 1.6s var(--ease);
  }
  .bar .exact {
    background: linear-gradient(90deg, var(--muted), var(--text-2));
  }
  .bar .hnsw {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    transition-delay: 0.3s;
  }
  .value {
    text-align: right;
    font: 600 14px var(--mono);
  }
  .facts {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    margin: 4px 0 0;
    border-top: 1px solid var(--border);
    padding-top: 16px;
  }
  .facts div {
    display: grid;
    gap: 2px;
  }
  dt {
    color: var(--muted);
    font-size: 12px;
  }
  dd {
    margin: 0;
    font: 650 28px var(--font);
    letter-spacing: -0.03em;
  }
  .waiting {
    margin: 0;
    padding: 28px 0;
    color: var(--muted);
    font: 13px var(--mono);
  }
  footer {
    color: var(--muted);
    font-size: 12.5px;
    line-height: 1.55;
  }
  @media (max-width: 900px) {
    .grid {
      grid-template-columns: 1fr;
    }
    .row {
      grid-template-columns: 110px 1fr 62px;
      gap: 10px;
    }
    dd {
      font-size: 22px;
    }
  }
</style>
