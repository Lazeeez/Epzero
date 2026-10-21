<script lang="ts">
  // Search latency (p50 and p99) and throughput over the last two minutes, one point a second.
  import type { Second } from "../../lib/cluster/protocol";

  let { seconds }: { seconds: Second[] } = $props();

  const W = 300;
  const H = 64;
  const rows = $derived(seconds.slice(-90));
  const top = $derived(Math.max(5, ...rows.map((s) => s.p99)) * 1.1);
  const x = (i: number) => (rows.length > 1 ? (i / (rows.length - 1)) * W : 0);
  const y = (ms: number) => H - (Math.min(ms, top) / top) * (H - 4) - 2;
  const path = (key: "p50" | "p99") => rows.map((s, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(s[key]).toFixed(1)}`).join(" ");
  // Seconds with incomplete answers, as marks along the bottom.
  const gaps = $derived(rows.map((s, i) => ({ i, bad: s.partial + s.failed > 0 })).filter((g) => g.bad));
</script>

<figure>
  <figcaption>
    <span><i class="p99"></i>p99</span>
    <span><i class="p50"></i>p50</span>
    <span><i class="gap"></i>incomplete</span>
    <span class="scale num">{top.toFixed(0)} ms</span>
  </figcaption>
  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="img" aria-label="Search latency over the last {rows.length} seconds">
    {#each gaps as g (g.i)}
      <rect x={x(g.i) - 1.5} y={H - 4} width="3" height="4" class="gap" />
    {/each}
    {#if rows.length > 1}
      <path d={path("p99")} class="p99" />
      <path d={path("p50")} class="p50" />
    {/if}
  </svg>
</figure>

<style>
  figure {
    margin: 0;
  }
  figcaption {
    display: flex;
    gap: 12px;
    margin-bottom: 6px;
    color: var(--muted);
    font-size: 11.5px;
  }
  figcaption span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  figcaption i {
    width: 10px;
    height: 2px;
    border-radius: 2px;
  }
  .scale {
    margin-left: auto;
    font-family: var(--mono);
  }
  svg {
    display: block;
    width: 100%;
    height: 64px;
    border-bottom: 1px solid var(--axis);
  }
  path {
    fill: none;
    stroke-width: 1.6;
    vector-effect: non-scaling-stroke;
  }
  .p99 {
    stroke: var(--series-2);
    background: var(--series-2);
  }
  .p50 {
    stroke: var(--accent);
    background: var(--accent);
  }
  .gap {
    fill: var(--warn);
    background: var(--warn);
  }
</style>
