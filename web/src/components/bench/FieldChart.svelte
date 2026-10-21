<script lang="ts">
  // Seven libraries on four ann-benchmarks datasets: one bar per library, its queries per second at
  // the dataset's recall target. Oryx's two variants carry the accent; the others stay neutral.
  import { field } from "../../lib/bench";

  let selected = $state(0);
  const run = $derived(field[selected]);
  const max = $derived(Math.max(...run.entries.map((e) => e.qps ?? 0)));
  const oryx = $derived(run.entries.find((e) => e.id === "oryx")!);
  const fmt = (x: number) => Math.round(x).toLocaleString();

  // Arrow keys move between datasets, as in any radio group.
  function onkeydown(event: KeyboardEvent) {
    const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    selected = (selected + step + field.length) % field.length;
    (event.currentTarget as HTMLElement).querySelectorAll("button")[selected]?.focus();
  }
</script>

<div class="field">
  <div class="seg" role="radiogroup" aria-label="Dataset" tabindex="-1" {onkeydown}>
    {#each field as d, i (d.key)}
      <button role="radio" aria-checked={i === selected} tabindex={i === selected ? 0 : -1} class:on={i === selected} onclick={() => (selected = i)}>
        {d.name}
      </button>
    {/each}
  </div>
  <p class="meta">
    {run.n.toLocaleString()} × {run.dim}-D {run.what}, {run.metric}, {run.queries.toLocaleString()} queries ·
    <b>queries per second at recall@10 = {run.recall.toFixed(2)}</b>
  </p>

  <ol class="rows" aria-label="Queries per second on {run.name} at recall {run.recall}, fastest first">
    {#each run.entries as e (e.id)}
      <li class="row" class:oryx={e.oryx}>
        <span class="name">{e.name}</span>
        <span class="track">
          {#if e.qps !== null}
            <i class:sq8={e.id === "oryx-sq8"} style:width="{(e.qps / max) * 100}%"></i>
          {:else}
            <em>never reaches {run.recall.toFixed(2)}</em>
          {/if}
        </span>
        <b class="num">{e.qps !== null ? fmt(e.qps) : "—"}</b>
        <span class="tip" aria-hidden="true">
          <strong>{e.name}</strong>
          {#if e.qps !== null}
            <span>{fmt(e.qps)} QPS at recall {run.recall.toFixed(2)}</span>
            {#if !e.oryx}<span>{(e.qps / oryx.qps!).toFixed(2)}× Oryx (fp32)</span>{/if}
          {:else}
            <span>Best recall {e.best.toFixed(3)} in the sweep</span>
          {/if}
          <span>Built in {e.build.toFixed(0)} s on 24 threads</span>
        </span>
      </li>
    {/each}
  </ol>
</div>

<style>
  .seg {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 2px;
    padding: 3px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
  }
  .seg button {
    padding: 6px 12px;
    border: 0;
    border-radius: 7px;
    background: none;
    color: var(--text-2);
    font-size: 13px;
    cursor: pointer;
  }
  .seg button.on {
    background: var(--surface-3);
    color: var(--text);
    box-shadow: 0 0 0 1px var(--border-strong);
  }
  .meta {
    margin: 14px 0 18px;
    color: var(--text-2);
    font-size: 13.5px;
  }
  .meta b {
    color: var(--text);
    font-weight: 600;
  }
  .rows {
    display: grid;
    gap: 8px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .row {
    position: relative;
    display: grid;
    grid-template-columns: 110px 1fr 64px;
    align-items: center;
    gap: 12px;
    padding: 2px 0;
    border-radius: 8px;
    color: var(--text-2);
    font-size: 14px;
  }
  .row.oryx .name {
    color: var(--text);
    font-weight: 600;
  }
  .row b {
    color: var(--text);
    font: 600 13.5px var(--mono);
    text-align: right;
  }
  .track {
    display: flex;
    align-items: center;
    height: 24px;
    border-radius: 6px;
    background: var(--surface-2);
    overflow: hidden;
  }
  .track i {
    height: 100%;
    border-radius: 6px 4px 4px 6px;
    background: color-mix(in srgb, var(--muted) 75%, var(--surface-2));
    transition: width 0.5s var(--ease);
  }
  .oryx .track i {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
  }
  .oryx .track i.sq8 {
    background: linear-gradient(90deg, var(--accent-2), var(--accent-strong));
  }
  .track em {
    padding: 0 10px;
    color: var(--muted);
    font-size: 12px;
    font-style: normal;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .row:hover .track i {
    filter: brightness(1.2);
  }
  .tip {
    position: absolute;
    z-index: 2;
    bottom: calc(100% + 6px);
    left: 122px;
    display: grid;
    gap: 2px;
    padding: 8px 10px;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    background: var(--surface-solid);
    box-shadow: var(--shadow);
    color: var(--text-2);
    font-size: 12.5px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transform: translateY(4px);
    transition: opacity 0.15s, transform 0.15s;
  }
  .tip strong {
    color: var(--text);
  }
  .row:hover .tip {
    opacity: 1;
    transform: none;
  }
  @media (max-width: 640px) {
    .row {
      grid-template-columns: 88px 1fr 52px;
      gap: 8px;
      font-size: 13px;
    }
    .tip {
      left: 96px;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .track i,
    .tip {
      transition: none;
    }
  }
</style>
