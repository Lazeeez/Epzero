<script lang="ts">
  // The Benchmarks header's graphic: each library's throughput at the same recall, as bars that
  // race out on arrival, Oryx's lit. Decorative: the same figures are in the stats and charts below.
  let { lanes, recall }: { lanes: { name: string; qps: number; lead?: boolean }[]; recall: number } = $props();
  const top = $derived(Math.max(...lanes.map((l) => l.qps)));
</script>

<figure class="race" aria-hidden="true">
  <figcaption>Queries/s at recall {recall.toFixed(2)} · one thread</figcaption>
  {#each lanes as lane, i (lane.name)}
    <div class="lane" class:lead={lane.lead} style:--w={lane.qps / top} style:--d="{0.2 + i * 0.12}s">
      <span class="name">{lane.name}</span>
      <span class="track"><i></i></span>
      <b class="num">{Math.round(lane.qps).toLocaleString()}</b>
    </div>
  {/each}
</figure>

<style>
  .race {
    display: grid;
    gap: 14px;
    margin: 0;
    padding: 22px 22px 24px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--glass-fill);
  }
  figcaption {
    margin-bottom: 4px;
    color: var(--muted);
    font: 11.5px var(--mono);
  }
  .lane {
    display: grid;
    grid-template-columns: 64px 1fr 58px;
    align-items: center;
    gap: 12px;
    color: var(--text-2);
    font-size: 13.5px;
  }
  .lane.lead {
    color: var(--text);
    font-weight: 600;
  }
  .track {
    height: 10px;
    border-radius: 99px;
    background: var(--surface-2);
    overflow: hidden;
  }
  .track i {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: var(--axis);
    transform-origin: left;
    transform: scaleX(0);
    animation: grow 1.4s var(--ease) var(--d) forwards;
  }
  .lead .track i {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    box-shadow: 0 0 18px var(--accent-glow);
  }
  b {
    font: 600 13px var(--mono);
    text-align: right;
    opacity: 0;
    animation: show 0.5s var(--ease) calc(var(--d) + 1s) forwards;
  }
  @keyframes grow {
    to {
      transform: scaleX(var(--w));
    }
  }
  @keyframes show {
    to {
      opacity: 1;
    }
  }
</style>
