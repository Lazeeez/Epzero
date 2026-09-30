<script lang="ts">
  import { formatQps, formatUs } from "../../lib/format";
  import type { EngineInfo, FromWorker, KernelRow, LabConfig, StageId, SweepRow } from "../../lib/lab/protocol";
  import { reloadForThreads } from "../../lib/isolation";
  import { UNSUPPORTED_BROWSER, wasmSimdSupported } from "../../lib/support";
  import RecallChart from "./RecallChart.svelte";

  const cores = Math.min(navigator.hardwareConcurrency || 1, 16);

  const choices = {
    n: [20000, 50000, 100000, 200000],
    dim: [64, 128, 256],
    M: [8, 16, 32],
    efConstruction: [64, 128, 256],
  } as const;
  const labels: Record<keyof typeof choices, string> = { n: "Vectors", dim: "Dimensions", M: "Links per node (M)", efConstruction: "efConstruction" };
  let settings = $state({ n: 100000, dim: 128, M: 16, efConstruction: 128 });

  const stageNames: [StageId, string][] = [
    ["generate", "Generate clustered vectors"],
    ["truth", "Exact ground truth"],
    ["build", "Build the HNSW graph"],
    ["baseline", "Exact-scan baseline"],
    ["sweep", "Recall / speed sweep"],
    ["kernel", "SIMD kernel check"],
  ];

  let root: HTMLElement | undefined = $state();
  let worker: Worker | null = null;
  let info = $state<EngineInfo | null>(null);
  let error = $state("");
  let running = $state(false);
  let config = $state<LabConfig | null>(null);
  let stages = $state<Partial<Record<StageId, { state: "running" | "done"; ms?: number; detail?: string }>>>({});
  let buildProgress = $state(0);
  let rows = $state<SweepRow[]>([]);
  let baseline = $state<number | null>(null);
  let summary = $state<{ buildMs: number; memoryBytes: number; layers: number } | null>(null);
  let kernels = $state<KernelRow[]>([]);

  // Start the engine worker when the lab scrolls near the viewport, not on page load.
  $effect(() => {
    if (!root) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        observer.disconnect();
        if (!wasmSimdSupported()) {
          error = UNSUPPORTED_BROWSER; // don't download an engine this browser can't run
          return;
        }
        worker = new Worker(new URL("../../lib/lab/lab.worker.ts", import.meta.url), { type: "module" });
        worker.onmessage = (event: MessageEvent<FromWorker>) => handle(event.data);
        worker.onerror = (event) => handle({ type: "error", message: event.message || "The engine worker failed to start." });
      },
      { rootMargin: "400px 0px" },
    );
    observer.observe(root);
    return () => {
      observer.disconnect();
      worker?.terminate();
      worker = null;
    };
  });

  function run() {
    if (!worker || !info || running) return;
    config = { ...settings, queries: 1000, clusters: 100, k: 10, efs: [10, 16, 24, 32, 48, 64, 96, 128, 192, 256] };
    stages = {};
    buildProgress = 0;
    rows = [];
    baseline = null;
    summary = null;
    kernels = [];
    error = "";
    running = true;
    worker.postMessage({ type: "run", config: $state.snapshot(config) });
  }

  function setStage(id: StageId, patch: { state?: "running" | "done"; ms?: number; detail?: string }) {
    stages = { ...stages, [id]: { ...(stages[id] ?? { state: "running" }), ...patch } };
  }

  function handle(message: FromWorker) {
    switch (message.type) {
      case "ready":
        info = message.info;
        break;
      case "stage":
        setStage(message.stage, { state: message.state, ms: message.ms });
        if (message.stage === "build" && message.state === "done") buildProgress = 1;
        break;
      case "progress":
        buildProgress = message.built / message.total;
        setStage("build", { detail: `${message.built.toLocaleString()} / ${message.total.toLocaleString()} · ${formatQps(message.insertsPerSec)} inserts/s` });
        break;
      case "summary":
        summary = { buildMs: message.buildMs, memoryBytes: message.memoryBytes, layers: message.layers };
        setStage("build", { detail: `${formatQps(config!.n / (message.buildMs / 1000))} inserts/s on ${info!.threads} ${info!.threads === 1 ? "thread" : "threads"}` });
        break;
      case "baseline":
        baseline = message.qps;
        setStage("baseline", { detail: `${formatQps(message.qps)} QPS at recall 1.000` });
        break;
      case "sweep":
        rows = [...rows, message.row];
        setStage("sweep", { detail: `ef ${message.row.ef} · recall ${message.row.recall.toFixed(3)}` });
        break;
      case "kernel":
        kernels = [...kernels, message.row];
        setStage("kernel", { detail: `${(message.row.scalarNs / message.row.simdNs).toFixed(1)}× faster at ${message.row.dim}-D` });
        break;
      case "done":
        running = false;
        break;
      case "error":
        error = message.message;
        running = false;
        break;
    }
  }

  // The operating point a production system would pick: the fastest setting with recall >= 0.95.
  const operating = $derived.by(() => {
    const good = rows.filter((r) => r.recall >= 0.95);
    if (good.length) return good.reduce((a, b) => (b.qps > a.qps ? b : a));
    return rows.reduce<SweepRow | undefined>((a, b) => (!a || b.recall > a.recall ? b : a), undefined);
  });
  const maxKernel = $derived(Math.max(1, ...kernels.map((k) => k.scalarNs / k.simdNs)));
  const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`);
</script>

<div class="lab" bind:this={root}>
  <div class="setup card">
    <ul class="env" aria-label="Runtime">
      {#if info}
        <li class="ok">SIMD: {info.simd === "wasm" ? "WebAssembly SIMD128" : info.simd}</li>
        {#if info.variant === "mt"}
          <li class="ok">{info.threads} threads</li>
        {:else}
          <!-- A warning, not an error: everything works, on one core instead of all of them. -->
          <li class="warn" title="Browsers only allow WebAssembly threads on cross-origin isolated pages; this site's service worker makes it so once it serves the page.">
            1 thread: multithreading is off for this page load
          </li>
          {#if cores > 1}
            <li class="action">
              <button onclick={reloadForThreads}>Reload to use all {cores} cores</button>
            </li>
          {/if}
        {/if}
      {:else if error}
        <li class="warn">{error}</li>
      {:else}
        <li>Loading the engine…</li>
      {/if}
    </ul>

    <div class="options">
      {#each Object.keys(choices) as key (key)}
        {@const k = key as keyof typeof choices}
        <fieldset disabled={running}>
          <legend>{labels[k]}</legend>
          <div class="seg">
            {#each choices[k] as value (value)}
              <label class:on={settings[k] === value}>
                <input type="radio" name={k} {value} checked={settings[k] === value} onchange={() => (settings[k] = value)} />
                {k === "n" ? `${value / 1000}k` : value}
              </label>
            {/each}
          </div>
        </fieldset>
      {/each}
    </div>

    <button class="btn btn-primary run" onclick={run} disabled={!info || running}>
      {#if running}<i class="spin"></i>Running…{:else}Run on this device{/if}
    </button>
  </div>

  {#if config}
    <div class="results">
      <ol class="stages card" aria-live="polite">
        {#each stageNames as [id, name] (id)}
          {@const stage = stages[id]}
          <li data-state={stage?.state ?? "pending"}>
            <span class="dot"></span>
            <span class="name">{name}</span>
            <span class="detail num">{stage?.detail ?? ""}</span>
            <span class="ms num">{stage?.ms !== undefined ? fmtMs(stage.ms) : ""}</span>
            {#if id === "build"}
              <span class="bar"><i style:width="{buildProgress * 100}%"></i></span>
            {/if}
          </li>
        {/each}
      </ol>

      {#if operating && summary}
        <div class="tiles">
          <div class="tile card hero">
            <span>Queries / s at recall {operating.recall >= 0.95 ? "≥ 0.95" : operating.recall.toFixed(2)}</span>
            <strong class="num">{formatQps(operating.qps)}</strong>
            <em class="num">ef {operating.ef} · recall {operating.recall.toFixed(3)} · p50 {formatUs(operating.p50Us)}</em>
          </div>
          <div class="tile card">
            <span>Faster than exact</span>
            <strong class="num">{baseline ? `${Math.round(operating.qps / baseline).toLocaleString()}×` : "–"}</strong>
            <em class="num">{operating.distancesPerQuery.toFixed(0)} of {config.n.toLocaleString()} vectors scored</em>
          </div>
          <div class="tile card">
            <span>Graph build</span>
            <strong class="num">{(summary.buildMs / 1000).toFixed(1)} s</strong>
            <em class="num">{formatQps(config.n / (summary.buildMs / 1000))} inserts/s</em>
          </div>
          <div class="tile card">
            <span>Index</span>
            <strong class="num">{(summary.memoryBytes / 1048576).toFixed(0)} MiB</strong>
            <em class="num">{summary.layers} layers · M = {config.M}</em>
          </div>
        </div>
      {/if}

      {#if rows.length}
        <div class="card pad">
          <RecallChart
            series={[{ name: "Oryx on this device", color: "var(--accent)", points: rows }]}
            {baseline}
            baselineLabel="Exact scan on this device"
            height={360}
            label="Recall versus queries per second measured in this browser"
          />
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>ef</th><th>Recall@10</th><th>QPS</th><th>p50</th><th>p99</th><th>Scored / query</th><th>vs exact</th></tr>
              </thead>
              <tbody>
                {#each rows as r (r.ef)}
                  <tr class:op={r === operating}>
                    <td>{r.ef}</td><td>{r.recall.toFixed(4)}</td><td>{Math.round(r.qps).toLocaleString()}</td><td>{formatUs(r.p50Us)}</td>
                    <td>{formatUs(r.p99Us)}</td><td>{Math.round(r.distancesPerQuery).toLocaleString()}</td>
                    <td>{baseline ? `${Math.round(r.qps / baseline).toLocaleString()}×` : "–"}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      {/if}

      {#if kernels.length}
        <div class="card pad kernels">
          <h3>SIMD distance kernel vs scalar</h3>
          {#each kernels as k (k.dim)}
            {@const speedup = k.scalarNs / k.simdNs}
            <div class="krow">
              <span class="num">{k.dim}-D</span>
              <span class="track"><i style:width="{(speedup / maxKernel) * 100}%"></i></span>
              <span class="num"><b>{speedup.toFixed(1)}×</b> {k.scalarNs.toFixed(1)} → {k.simdNs.toFixed(1)} ns</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .lab {
    display: grid;
    gap: 16px;
  }
  .setup {
    display: grid;
    min-width: 0;
    gap: 20px;
    padding: 24px;
  }
  .env {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .env li {
    padding: 5px 12px;
    border: 1px solid var(--border);
    border-radius: 99px;
    color: var(--muted);
    font: 12.5px var(--mono);
  }
  .env li.ok {
    border-color: color-mix(in srgb, var(--good) 45%, transparent);
    color: var(--good);
  }
  .env li.warn {
    border-color: color-mix(in srgb, var(--warn) 45%, transparent);
    color: var(--warn);
  }
  .env li.action {
    padding: 0;
    border: 0;
  }
  .env li.action button {
    padding: 5px 12px;
    border: 1px solid var(--accent);
    border-radius: 99px;
    background: var(--accent-wash);
    color: var(--accent-strong);
    font: 12.5px var(--mono);
    cursor: pointer;
  }
  .options {
    display: flex;
    flex-wrap: wrap;
    gap: 16px 28px;
  }
  fieldset {
    min-width: 0; /* fieldsets default to min-content width, which overflows phones */
    margin: 0;
    padding: 0;
    border: 0;
  }
  legend {
    margin-bottom: 8px;
    color: var(--text-2);
    font-size: 13px;
  }
  .seg {
    display: flex;
    flex-wrap: wrap;
    padding: 3px;
    border: 1px solid var(--border);
    border-radius: 11px;
  }
  .seg label {
    padding: 6px 12px;
    border-radius: 8px;
    color: var(--muted);
    font: 13px var(--mono);
    cursor: pointer;
  }
  .seg label.on {
    background: var(--surface-3);
    color: var(--text);
  }
  .seg input {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }
  .seg label:has(input:focus-visible) {
    outline: 2px solid var(--accent);
  }
  fieldset:disabled .seg {
    opacity: 0.5;
  }
  .run {
    justify-self: start;
  }
  .spin {
    width: 14px;
    height: 14px;
    border: 2px solid color-mix(in srgb, var(--page) 40%, transparent);
    border-top-color: var(--page);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .results {
    display: grid;
    gap: 16px;
  }
  .lab > *,
  .results > * {
    min-width: 0;
  }
  .stages {
    display: grid;
    gap: 2px;
    margin: 0;
    padding: 14px 20px;
    list-style: none;
  }
  .stages li {
    display: grid;
    grid-template-columns: 14px 220px 1fr auto;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
    color: var(--muted);
    font-size: 14px;
  }
  .stages li[data-state="running"],
  .stages li[data-state="done"] {
    color: var(--text);
  }
  .dot {
    width: 10px;
    height: 10px;
    border: 2px solid var(--border-strong);
    border-radius: 50%;
  }
  [data-state="running"] .dot {
    border-color: var(--accent);
    border-top-color: transparent;
    animation: spin 0.8s linear infinite;
  }
  [data-state="done"] .dot {
    border-color: var(--good);
    background: var(--good);
  }
  .detail {
    color: var(--text-2);
    font: 12.5px var(--mono);
  }
  .ms {
    font: 12.5px var(--mono);
  }
  .bar {
    grid-column: 2 / -1;
    height: 4px;
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
  .tiles {
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr 1fr;
    gap: 16px;
  }
  .tile {
    display: grid;
    gap: 4px;
    padding: 18px;
  }
  .tile span {
    color: var(--text-2);
    font-size: 13px;
  }
  .tile strong {
    font-size: 30px;
    letter-spacing: -0.03em;
  }
  .tile.hero strong {
    color: var(--accent-strong);
  }
  .tile em {
    color: var(--muted);
    font: normal 12px var(--mono);
  }
  .pad {
    padding: 24px;
  }
  .table-wrap {
    margin-top: 20px;
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font: 13px var(--mono);
  }
  th,
  td {
    padding: 8px 12px;
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
  tr.op td {
    background: var(--accent-wash);
    color: var(--accent-strong);
  }
  .kernels h3 {
    margin-bottom: 16px;
    font-size: 17px;
  }
  .krow {
    display: grid;
    grid-template-columns: 60px 1fr 200px;
    align-items: center;
    gap: 14px;
    padding: 5px 0;
    font: 13px var(--mono);
    color: var(--text-2);
  }
  .krow b {
    color: var(--text);
  }
  .track {
    height: 14px;
    border-radius: 99px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .track i {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    transition: width 0.6s var(--ease);
  }
  @media (max-width: 900px) {
    .tiles {
      grid-template-columns: 1fr 1fr;
    }
    .stages li {
      grid-template-columns: 14px 1fr auto;
    }
    .detail {
      grid-column: 2 / -1;
      grid-row: 2;
    }
    .krow {
      grid-template-columns: 50px 1fr;
    }
    .krow span:last-child {
      grid-column: 2;
    }
  }
</style>
