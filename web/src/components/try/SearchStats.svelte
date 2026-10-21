<script lang="ts">
  import { engine, type SearchOutcome } from "../../lib/engine/engine.svelte";
  import { agreement } from "../../lib/engine/trace-analysis";

  let { outcome, encodeMs = null }: { outcome: SearchOutcome; encodeMs?: number | null } = $props();

  const trace = $derived(outcome.trace);
  const hits = $derived(agreement(trace));
  const k = $derived(trace.results.length);
  const speedup = $derived(trace.exactMicros / Math.max(trace.micros, 1));
  const share = $derived((trace.distanceComputations / engine.count) * 100);

  const time = (us: number) => (us >= 1000 ? `${(us / 1000).toFixed(1)} ms` : `${us.toFixed(0)} µs`);

  // With a filter, the engine picks how to honor it from how many photos pass.
  const STRATEGY = {
    exact: { name: "exact scan", why: "few enough photos match to score each one" },
    inline: { name: "graph walk", why: "most photos match, so it walks the graph and skips the rest" },
    "two-hop": { name: "two-hop walk", why: "it walks only matching photos, reaching past the others to their neighbours" },
  } as const;
  const filtered = $derived(trace.strategy ? { ...STRATEGY[trace.strategy], allowed: trace.allowed ?? 0 } : null);
  // Hybrid: how many of the shown results only the words found.
  const wordsOnly = $derived(
    trace.hybrid ? Array.from(trace.hybrid.fromWords).filter((w, i) => w && !trace.hybrid!.fromVector[i]).length : 0,
  );
</script>

<div class="stats" aria-live="polite">
  <div class="primary">
    <div class="stat hero">
      <span class="label">{filtered ? "Oryx, filtered" : "HNSW search"}</span>
      <strong class="num">{time(trace.micros)}</strong>
    </div>
    <div class="vs" aria-hidden="true">vs</div>
    <div class="stat">
      <span class="label">Exact scan</span>
      <strong class="num dim">{time(trace.exactMicros)}</strong>
    </div>
    <div class="stat accent">
      {#if trace.strategy === "exact"}
        <!-- So few photos match that the engine scored each one: the answer is exact, not faster. -->
        <span class="label">Answer</span>
        <strong>Exact</strong>
      {:else}
        <span class="label">Faster</span>
        <strong class="num">{speedup.toFixed(0)}×</strong>
      {/if}
    </div>
  </div>
  <div class="bar" aria-hidden="true">
    <i style:width="{Math.min(share, 100)}%"></i>
  </div>
  <p class="detail">
    Scored <strong class="num">{trace.distanceComputations.toLocaleString()}</strong> of {engine.count.toLocaleString()} photos
    (<span class="num">{share.toFixed(1)}%</span>){trace.strategy === "exact" ? "" : ` across ${trace.entryLayer + 1} layers`} ·
    <strong class="num" class:perfect={hits === k}>{hits}/{k}</strong> same as the exact scan{trace.hybrid ? " (image search)" : ""}
    {#if encodeMs !== null && outcome.query.kind === "vector"}
      · CLIP {outcome.query.source === "text" ? "text" : "image"} encoder {encodeMs.toFixed(0)} ms
    {/if}
  </p>
  {#if trace.hybrid}
    <p class="keywords">
      <span class="tag">hybrid</span>
      <span>
        {#if trace.hybrid.matches}
          “{trace.hybrid.words}” matched <span class="num">{trace.hybrid.matches}{trace.hybrid.matches >= 30 ? "+" : ""}</span>
          captions (BM25), merged with the image search by rank fusion: {wordsOnly
            ? `${wordsOnly} of the ${trace.hybrid.results.length} came from the words alone`
            : "the words agreed with the image search"}
          · +{time(trace.hybrid.micros)}.
        {:else}
          No caption contains “{trace.hybrid.words}”, so these are the image search's results.
        {/if}
      </span>
    </p>
  {/if}
  {#if filtered}
    <p class="strategy">
      <span class="tag">{filtered.name}</span>
      <span>
        {filtered.allowed.toLocaleString()} photos match the filter ({((filtered.allowed / engine.count) * 100).toFixed(1)}%):
        {filtered.why}.
      </span>
    </p>
  {/if}
</div>

<style>
  .stats {
    display: grid;
    gap: 10px;
    padding: 14px 16px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--surface);
  }
  /* Two rows: every label on one line, every value on one shared baseline (the values differ in
     size, so aligning each column's bottom would leave the labels at different heights). */
  .primary {
    display: grid;
    grid-template-columns: auto auto auto minmax(0, 1fr);
    grid-template-rows: auto auto;
    align-items: baseline;
    column-gap: 14px;
    row-gap: 2px;
  }
  .stat {
    display: contents;
  }
  .stat:nth-child(1) > * {
    grid-column: 1;
  }
  .stat:nth-child(3) > * {
    grid-column: 3;
  }
  .stat:nth-child(4) > * {
    grid-column: 4;
    justify-self: end;
  }
  .label {
    grid-row: 1;
  }
  .stat strong {
    grid-row: 2;
    white-space: nowrap;
  }
  .label {
    white-space: nowrap;
    color: var(--muted);
    font-size: 11.5px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  strong {
    font-size: 20px;
    font-weight: 650;
    letter-spacing: -0.02em;
  }
  .hero strong {
    font-size: 26px;
    background: linear-gradient(90deg, var(--accent-strong), var(--accent-2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .dim {
    color: var(--text-2);
  }
  .vs {
    grid-column: 2;
    grid-row: 2;
    color: var(--muted);
    font-size: 12px;
  }
  .accent strong {
    color: var(--good);
  }
  .bar {
    height: 4px;
    border-radius: 99px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .bar i {
    display: block;
    height: 100%;
    min-width: 3px;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    transition: width 0.5s var(--ease);
  }
  .detail {
    margin: 0;
    color: var(--text-2);
    font-size: 12.5px;
    line-height: 1.5;
  }
  .detail strong {
    font-size: inherit;
    color: var(--text);
  }
  .perfect {
    color: var(--good) !important;
  }
  .strategy,
  .keywords {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin: 0;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    color: var(--text-2);
    font-size: 12.5px;
    line-height: 1.5;
  }
  .tag {
    flex: none;
    padding: 1px 8px;
    border-radius: 99px;
    background: var(--accent-wash);
    color: var(--accent-strong);
    font: 600 11px var(--mono);
    white-space: nowrap;
  }
</style>
