<script lang="ts">
  import { scaleLinear } from "d3-scale";
  import data from "../../../../bench/results/optimizations.json";

  const steps = data.steps;
  const max = Math.max(...steps.flatMap((s) => s.samples));
  const x = scaleLinear().domain([0, max * 1.04]).range([0, 100]);
  const base = steps[0].qps;
  let shown = $state(false);
  let root: HTMLElement | undefined = $state();

  $effect(() => {
    if (!root) return;
    const observer = new IntersectionObserver(([e]) => e.isIntersecting && ((shown = true), observer.disconnect()), { threshold: 0.3 });
    observer.observe(root);
    return () => observer.disconnect();
  });
</script>

<div class="opt" class:shown bind:this={root} role="img" aria-label="Queries per second after each optimization: {steps.map((s) => `${s.label} ${s.qps}`).join(', ')}">
  {#each steps as step, i (step.label)}
    {@const gain = step.qps / base - 1}
    <div class="row" style:--d="{i * 0.12}s">
      <span class="label">{step.label.replace(/^\+ /, "")}</span>
      <div class="track">
        <i class="bar" class:last={i === steps.length - 1} style:width="{x(step.qps)}%"></i>
        {#each step.samples as sample, j (j)}
          <b class="dot" style:left="{x(sample)}%"></b>
        {/each}
      </div>
      <span class="value num">
        {step.qps.toLocaleString()}
        {#if i > 0}<em class:up={gain > 0.03}>{gain >= 0 ? "+" : ""}{(gain * 100).toFixed(0)}%</em>{/if}
      </span>
    </div>
  {/each}
  <p class="note">
    Bars are the median of {data.rounds} interleaved runs; dots are the individual runs. Recall is {data.recall} at every step (same
    graph, reloaded from a snapshot).
  </p>
</div>

<style>
  .opt {
    display: grid;
    gap: 12px;
  }
  .row {
    display: grid;
    grid-template-columns: 160px 1fr 120px;
    align-items: center;
    gap: 16px;
  }
  .label {
    color: var(--text-2);
    font-size: 14px;
  }
  .track {
    position: relative;
    height: 30px;
    border-radius: 8px;
    background: var(--surface-2);
  }
  .bar {
    position: absolute;
    inset: 0 auto 0 0;
    border-radius: 8px;
    background: color-mix(in srgb, var(--accent) 38%, transparent);
    transform-origin: left;
    transform: scaleX(0);
    transition: transform 1.2s var(--ease) var(--d);
  }
  .bar.last {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
  }
  .dot {
    position: absolute;
    top: 50%;
    width: 6px;
    height: 6px;
    margin: -3px 0 0 -3px;
    border-radius: 50%;
    background: var(--text);
    opacity: 0;
    transition: opacity 0.4s calc(var(--d) + 1s);
  }
  .shown .bar {
    transform: none;
  }
  .shown .dot {
    opacity: 0.55;
  }
  .value {
    font: 600 14px var(--mono);
    text-align: right;
  }
  em {
    margin-left: 6px;
    color: var(--muted);
    font-style: normal;
    font-weight: 500;
  }
  em.up {
    color: var(--good);
  }
  .note {
    margin: 6px 0 0;
    color: var(--muted);
    font-size: 13px;
  }
  @media (max-width: 640px) {
    .row {
      grid-template-columns: 1fr auto;
      gap: 6px 12px;
    }
    .track {
      grid-column: 1 / -1;
      grid-row: 2;
    }
  }
</style>
