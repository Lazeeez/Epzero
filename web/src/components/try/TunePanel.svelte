<script lang="ts">
  import { engine } from "../../lib/engine/engine.svelte";
  import { formatQps } from "../../lib/format";

  let { ef, onclose, onrebuilt }: { ef: number; onclose: () => void; onrebuilt: () => void } = $props();

  const choices = { M: [8, 16, 32, 48], efConstruction: [32, 64, 128, 256] } as const;
  let M = $state(engine.params.M);
  let efConstruction = $state(engine.params.efConstruction);

  interface Run {
    M: number;
    efConstruction: number;
    buildMs: number;
    memoryMiB: number;
    layers: number;
    recall: number | null;
    qps: number | null;
    distances: number | null;
    ef: number;
  }
  // Every configuration tried in this visit, so the trade-offs can be compared side by side.
  const history = $state<Run[]>([]);
  const current = $derived(
    [...history].reverse().find((r) => r.M === engine.params.M && r.efConstruction === engine.params.efConstruction),
  );
  const changed = $derived(M !== engine.params.M || efConstruction !== engine.params.efConstruction);
  let panel: HTMLElement | undefined = $state();

  async function measure() {
    const graph = engine.graph;
    if (!graph) return;
    const result = await engine.evaluate(ef, 300).catch(() => null);
    const run: Run = {
      M: engine.params.M,
      efConstruction: engine.params.efConstruction,
      buildMs: graph.buildMs,
      memoryMiB: graph.memoryBytes / 1048576,
      layers: graph.maxLevel + 1,
      recall: result?.recall ?? null,
      qps: result?.qps ?? null,
      distances: result?.distancesPerQuery ?? null,
      ef,
    };
    const same = history.findIndex((r) => r.M === run.M && r.efConstruction === run.efConstruction && r.ef === run.ef);
    if (same >= 0) history[same] = run;
    else history.push(run);
  }

  async function rebuild() {
    await engine.rebuild(M, efConstruction);
    onrebuilt();
  }

  // Measure the graph when the panel opens, after every rebuild (a new graph object), and
  // whenever ef changes.
  // Debounced: dragging the ef slider shouldn't queue a one-second measurement per tick.
  $effect(() => {
    void ef;
    void engine.graph;
    if (engine.status !== "ready" || engine.rebuilding) return;
    const timer = setTimeout(measure, 350);
    return () => clearTimeout(timer);
  });

  $effect(() => {
    panel?.focus();
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onclose();
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  });

  const best = $derived(history.length > 1 ? Math.max(...history.map((r) => r.qps ?? 0)) : 0);
</script>

<section class="tune card" bind:this={panel} tabindex="-1" aria-labelledby="tune-title">
  <header>
    <div>
      <h2 id="tune-title">Tune the index</h2>
      <p>Rebuild all {engine.count.toLocaleString()} photos with new graph parameters, then see what it did to recall and speed.</p>
    </div>
    <button class="close" onclick={onclose} aria-label="Close index settings">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" /></svg>
    </button>
  </header>

  <div class="params">
    <fieldset disabled={engine.rebuilding}>
      <legend>Links per node <code>M</code></legend>
      <div class="seg">
        {#each choices.M as value (value)}
          <button class:on={M === value} onclick={() => (M = value)} aria-pressed={M === value}>{value}</button>
        {/each}
      </div>
      <small>More links: better recall, more memory, slower build.</small>
    </fieldset>
    <fieldset disabled={engine.rebuilding}>
      <legend>Build beam <code>efConstruction</code></legend>
      <div class="seg">
        {#each choices.efConstruction as value (value)}
          <button class:on={efConstruction === value} onclick={() => (efConstruction = value)} aria-pressed={efConstruction === value}>{value}</button>
        {/each}
      </div>
      <small>A wider build search picks better neighbours, at build time.</small>
    </fieldset>
  </div>

  <div class="actions">
    <button class="btn btn-primary" onclick={rebuild} disabled={!changed || engine.rebuilding || engine.status !== "ready"}>
      {#if engine.rebuilding}
        <i class="spin"></i>Rebuilding… {Math.round(engine.rebuildProgress * 100)}%
      {:else}
        Rebuild index
      {/if}
    </button>
    {#if engine.rebuilding}
      <span class="bar"><i style:width="{engine.rebuildProgress * 100}%"></i></span>
    {:else if engine.evaluating}
      <span class="note">Measuring recall on 300 held-out queries…</span>
    {:else if current}
      <span class="note num">
        Now: M {current.M} · efC {current.efConstruction} · recall@10 <b>{current.recall?.toFixed(3) ?? "–"}</b> at ef {current.ef}
      </span>
    {/if}
  </div>

  {#if history.length}
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>M</th><th>efC</th><th>Build</th><th>Memory</th><th>Layers</th><th>Recall@10</th><th>QPS (1 thread)</th><th>Scored / query</th></tr>
        </thead>
        <tbody>
          {#each history as run (`${run.M}-${run.efConstruction}-${run.ef}`)}
            <tr class:current={run === current}>
              <td>{run.M}</td>
              <td>{run.efConstruction}</td>
              <td>{(run.buildMs / 1000).toFixed(1)} s</td>
              <td>{run.memoryMiB.toFixed(0)} MiB</td>
              <td>{run.layers}</td>
              <td>{run.recall?.toFixed(3) ?? "–"}</td>
              <td class:best={run.qps !== null && run.qps === best}>{run.qps ? formatQps(run.qps) : "–"}</td>
              <td>{run.distances ? Math.round(run.distances).toLocaleString() : "–"}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <p class="foot">Recall is measured against an exact scan, at the current beam width <code>ef = {ef}</code>. Change ef with the slider to see it move.</p>
  {/if}
</section>

<style>
  .tune {
    display: grid;
    gap: 16px;
    width: min(620px, calc(100vw - 32px));
    max-height: calc(100dvh - var(--nav-h) - 110px);
    overflow: auto;
    padding: 20px;
    background: var(--surface-solid);
    outline: none;
    animation: drop 0.3s var(--ease);
  }
  header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
  }
  h2 {
    font-size: 18px;
  }
  header p {
    margin: 4px 0 0;
    color: var(--text-2);
    font-size: 13px;
  }
  .close {
    display: grid;
    place-items: center;
    flex: none;
    width: 32px;
    height: 32px;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: var(--surface);
    color: var(--text-2);
    cursor: pointer;
  }
  .close svg {
    width: 15px;
    height: 15px;
    stroke: currentColor;
    stroke-width: 2;
    stroke-linecap: round;
  }
  .params {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  fieldset {
    display: grid;
    gap: 8px;
    min-width: 0;
    margin: 0;
    padding: 0;
    border: 0;
  }
  legend {
    margin-bottom: 8px;
    color: var(--text-2);
    font-size: 13px;
  }
  small {
    color: var(--muted);
    font-size: 12px;
  }
  .seg {
    display: flex;
    padding: 3px;
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .seg button {
    flex: 1;
    padding: 6px 0;
    border: 0;
    border-radius: 7px;
    background: transparent;
    color: var(--muted);
    font: 13px var(--mono);
    cursor: pointer;
  }
  .seg button.on {
    background: var(--surface-3);
    color: var(--text);
  }
  fieldset:disabled .seg {
    opacity: 0.5;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px;
  }
  .actions .btn {
    height: 40px;
  }
  .spin {
    width: 13px;
    height: 13px;
    border: 2px solid color-mix(in srgb, var(--page) 40%, transparent);
    border-top-color: var(--page);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .bar {
    flex: 1;
    min-width: 120px;
    height: 5px;
    border-radius: 99px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .bar i {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    transition: width 0.2s;
  }
  .note {
    color: var(--muted);
    font: 12.5px var(--mono);
  }
  .note b {
    color: var(--good);
  }
  .table-wrap {
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font: 12.5px var(--mono);
  }
  th,
  td {
    padding: 7px 9px;
    border-bottom: 1px solid var(--border);
    text-align: right;
    white-space: nowrap;
  }
  th {
    color: var(--muted);
    font-weight: 500;
  }
  th:nth-child(-n + 2),
  td:nth-child(-n + 2) {
    text-align: left;
  }
  tr.current td {
    background: var(--accent-wash);
    color: var(--accent-strong);
  }
  td.best {
    color: var(--good);
  }
  .foot {
    margin: 0;
    color: var(--muted);
    font-size: 12px;
  }
  @keyframes drop {
    from {
      opacity: 0;
      transform: translateY(-6px);
    }
  }
  @media (max-width: 560px) {
    .params {
      grid-template-columns: 1fr;
    }
  }
</style>
