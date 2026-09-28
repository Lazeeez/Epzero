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
</script>

<div class="stats" aria-live="polite">
  <div class="primary">
    <div class="stat hero">
      <span class="label">HNSW search</span>
      <strong class="num">{time(trace.micros)}</strong>
    </div>
    <div class="vs" aria-hidden="true">vs</div>
    <div class="stat">
      <span class="label">Exact scan</span>
      <strong class="num dim">{time(trace.exactMicros)}</strong>
    </div>
    <div class="stat accent">
      <span class="label">Faster</span>
      <strong class="num">{speedup.toFixed(0)}×</strong>
    </div>
  </div>
  <div class="bar" aria-hidden="true">
    <i style:width="{Math.min(share, 100)}%"></i>
  </div>
  <p class="detail">
    Scored <strong class="num">{trace.distanceComputations.toLocaleString()}</strong> of {engine.count.toLocaleString()} photos
    (<span class="num">{share.toFixed(1)}%</span>) across {trace.entryLayer + 1} layers ·
    <strong class="num" class:perfect={hits === k}>{hits}/{k}</strong> same as the exact scan
    {#if encodeMs !== null && outcome.query.kind === "vector"}
      · CLIP {outcome.query.source === "text" ? "text" : "image"} encoder {encodeMs.toFixed(0)} ms
    {/if}
  </p>
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
  .primary {
    display: flex;
    align-items: flex-end;
    gap: 14px;
  }
  .stat {
    display: grid;
    gap: 2px;
  }
  .label {
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
    padding-bottom: 5px;
    color: var(--muted);
    font-size: 12px;
  }
  .accent {
    margin-left: auto;
    text-align: right;
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
</style>
