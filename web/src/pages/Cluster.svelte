<script lang="ts">
  // The cluster page: the real Go router, compiled to WebAssembly, against simulated data nodes.
  // Inject faults, watch searches keep their answers, and check the read/write history for
  // linearizability with Porcupine, all in this tab.
  import Topology from "../components/cluster/Topology.svelte";
  import Sparkline from "../components/cluster/Sparkline.svelte";
  import { clusterSim, SIM_OPTIONS } from "../lib/cluster/sim.svelte";
  import type { Second } from "../lib/cluster/protocol";
  import { reveal } from "../lib/reveal";
  import { spotlight } from "../lib/spotlight";

  $effect(() => {
    clusterSim.start();
    return () => clusterSim.stop();
  });

  const snap = $derived(clusterSim.snapshot);
  const running = $derived(clusterSim.status === "running" && !!snap);
  // Until the simulator is up, the diagram shows the nodes it will start, so nothing moves when it does.
  const placeholder = Array.from({ length: SIM_OPTIONS.nodes }, (_, i) => ({ name: `node-${i + 1}`, up: true, slow_ms: 0, shards: [] }));
  const nodes = $derived(running ? (snap?.nodes ?? []) : placeholder);
  const seconds = $derived(snap?.seconds ?? []);
  const recent = $derived(seconds.slice(-10));
  const events = $derived((snap?.cluster.events ?? []).slice(0, 14));
  const failovers = $derived((snap?.cluster.events ?? []).filter((e) => e.kind === "failover").length);

  function sum(rows: Second[], key: keyof Second) {
    return rows.reduce((total, s) => total + (s[key] as number), 0);
  }
  const searches = $derived(sum(recent, "searches"));
  const qps = $derived(recent.length ? searches / recent.length : 0);
  const exact = $derived.by(() => {
    const rows = recent.filter((s) => s.recall >= 0);
    const complete = rows.reduce((n, s) => n + s.searches - s.partial - s.failed, 0);
    if (!complete) return null;
    return rows.reduce((total, s) => total + s.recall * (s.searches - s.partial - s.failed), 0) / complete;
  });
  const partial = $derived(searches ? (sum(recent, "partial") + sum(recent, "failed")) / searches : 0);
  const p99 = $derived(recent.length ? Math.max(...recent.map((s) => s.p99)) : 0);
  const p50 = $derived(recent.length ? recent[recent.length - 1].p50 : 0);
  const writes = $derived(sum(recent, "writes"));
  const unreplicated = $derived(writes ? (sum(recent, "unreplicated") + sum(recent, "write_errors")) / writes : 0);

  // Chaos: a random fault every few seconds, healed before the next.
  let chaos = $state(false);
  $effect(() => {
    if (!chaos || clusterSim.status !== "running") return;
    let timer = 0;
    const kinds = ["kill", "isolate", "slow", "lossy"] as const;
    const step = (healNext: boolean) => {
      if (healNext) clusterSim.fault({ kind: "heal" });
      else {
        const node = nodes[Math.floor(Math.random() * nodes.length)]?.name;
        const kind = kinds[Math.floor(Math.random() * kinds.length)];
        if (node) {
          if (kind === "slow") clusterSim.fault({ kind, node, ms: 120 });
          else if (kind === "lossy") clusterSim.fault({ kind, node, loss: 0.3 });
          else clusterSim.fault({ kind, node });
        }
      }
      timer = window.setTimeout(() => step(!healNext), healNext ? 3000 : 4500);
    };
    step(false);
    return () => {
      clearTimeout(timer);
      clusterSim.fault({ kind: "heal" });
    };
  });

  function ago(iso: string) {
    const s = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
    return s < 60 ? `${s.toFixed(0)} s ago` : `${Math.floor(s / 60)} min ago`;
  }
  // Re-render "ago" labels with each snapshot.
  const now = $derived(snap ? Date.now() : 0);

  const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
</script>

<div class="cluster">
  <header class="head container">
    <p class="eyebrow">Cluster</p>
    <h1><span class="gradient-text">Break the cluster.</span><br />Watch it keep its answers.</h1>
    <p class="lede">
      The Epzero router runs here, the same Go code that fronts the real cluster, compiled to WebAssembly. Its data nodes and the network
      between them are simulated in memory, and they replicate like the real C++ nodes. Kill a node, cut it off or slow it down:
      searches keep returning exact answers, and a linearizability checker audits every read and write.
    </p>
  </header>

  <section class="container grid" aria-label="Simulated cluster">
    <div class="stage card">
      <Topology {nodes} links={running ? (snap?.links ?? []) : []} shards={running ? (snap?.cluster.shards ?? []) : []} ready={running} />
      {#if !running}
        <div class="boot" class:failed={clusterSim.status === "error"} role="status" aria-live="polite">
          {#if clusterSim.status === "error"}
            <strong>The simulator couldn't start: {clusterSim.error}</strong>
          {:else}
            <i class="spin" aria-hidden="true"></i>
            <span>
              {clusterSim.status === "starting"
                ? `Starting ${SIM_OPTIONS.nodes} nodes, loading ${SIM_OPTIONS.vectors.toLocaleString()} vectors`
                : "Loading the router (Go → WebAssembly)"}
            </span>
          {/if}
        </div>
      {/if}
    </div>

    <aside class="side">
      <div class="card panel" aria-label="Live metrics" use:spotlight>
        <div class="tiles">
          <div class="tile">
            <span>Exact answers</span>
            <b class="num" class:bad={exact !== null && exact < 0.999}>{exact === null ? "–" : pct(exact)}</b>
            <small>of the true top 10, last 10 s</small>
          </div>
          <div class="tile">
            <span>Incomplete</span>
            <b class="num" class:bad={partial > 0}>{pct(partial)}</b>
            <small>searches missing a shard</small>
          </div>
          <div class="tile">
            <span>Latency p99</span>
            <b class="num">{p99.toFixed(1)}<em>ms</em></b>
            <small>p50 {p50.toFixed(1)} ms · {qps.toFixed(0)} searches/s</small>
          </div>
          <div class="tile">
            <span>Failovers</span>
            <b class="num">{failovers}</b>
            <small>{pct(unreplicated, 0)} of writes on one copy</small>
          </div>
        </div>
        <Sparkline {seconds} />
      </div>

      <div class="card panel controls" use:spotlight>
        <div class="row">
          <div>
            <strong>Consistency</strong>
            <small>Which copies a search may read</small>
          </div>
          <div class="seg" role="radiogroup" aria-label="Consistency">
            {#each ["strong", "eventual"] as const as level (level)}
              <button
                role="radio"
                aria-checked={clusterSim.consistency === level}
                class:on={clusterSim.consistency === level}
                onclick={() => clusterSim.setConsistency(level)}
              >
                {level === "strong" ? "Strong" : "Eventual"}
              </button>
            {/each}
          </div>
        </div>
        <p class="hint">
          {clusterSim.consistency === "strong"
            ? "Primaries only: every search sees every acknowledged write. A shard waits for failover when its primary dies."
            : "Primary or in-sync replica, hedged when slow: lower tail latency, but around a fault a search can miss a write."}
        </p>
        <div class="row">
          <label class="switch">
            <input type="checkbox" role="switch" bind:checked={chaos} disabled={clusterSim.status !== "running"} />
            <span>Chaos monkey</span>
          </label>
          <button class="btn btn-ghost small" onclick={() => clusterSim.fault({ kind: "heal" })} disabled={clusterSim.status !== "running"}>
            Heal everything
          </button>
        </div>
        <button class="btn btn-primary verify" onclick={() => clusterSim.verify()} disabled={clusterSim.status !== "running" || clusterSim.checking}>
          {clusterSim.checking ? "Checking…" : "Check linearizability"}
        </button>
        {#if clusterSim.check}
          {@const c = clusterSim.check}
          <p class="verdict {c.result}" role="status">
            {#if c.result === "ok"}
              <b>✓ Linearizable.</b> {c.operations.toLocaleString()} reads and writes ({c.ambiguous.toLocaleString()} with unknown outcome) checked by Porcupine in {c.millis} ms.
            {:else if c.result === "illegal"}
              <b>✗ Not linearizable.</b> Among {c.operations.toLocaleString()} operations, some reads saw writes out of order.
              {clusterSim.consistency === "eventual" ? "That is the price of reading replicas: switch to Strong and check again." : ""}
            {:else}
              <b>Inconclusive.</b> The checker ran out of time on {c.operations.toLocaleString()} operations.
            {/if}
          </p>
        {/if}
      </div>
    </aside>
  </section>

  <section class="container lower">
    <div class="card panel log" aria-labelledby="events-title" use:spotlight>
      <h2 id="events-title">Router events</h2>
      {#if events.length}
        <ol>
          {#each events as e, i (e.time + i)}
            <li class={e.kind}>
              <span class="kind">{e.kind}</span>
              <span class="msg">{e.message}</span>
              <time datetime={e.time}>{now && ago(e.time)}</time>
            </li>
          {/each}
        </ol>
      {:else}
        <p class="empty">Nothing yet: kill a node to see the router fail its shards over.</p>
      {/if}
    </div>

    <div class="card panel how" use:reveal use:spotlight>
      <h2>What's real here</h2>
      <ul>
        <li>
          <b>The router is the production code.</b> Health checks, failover by epoch, hedged reads, write routing and the shard map are
          package <code>cluster/router</code>, unchanged, talking to nodes through an interface.
        </li>
        <li>
          <b>The nodes mirror <code>epzero-node</code>.</b> Primaries forward each write to an in-sync replica before acknowledging it; a
          lagging replica fetches what it missed; a returning ex-primary starts over from a snapshot.
        </li>
        <li>
          <b>The same simulation tests the router.</b> On a virtual clock, Porcupine checks every history under crashes, partitions and
          lossy links. It found two bugs, both now fixed: a retried write could be applied twice, and a replica could keep serving stale
          reads after a lost forward.
        </li>
      </ul>
    </div>
  </section>
</div>

<style>
  .cluster {
    padding: calc(var(--nav-h) + 48px) 0 64px;
  }
  .head h1 {
    margin: 0 0 16px;
    font-size: clamp(34px, 5vw, 56px);
    line-height: 1.05;
    letter-spacing: -0.03em;
  }
  .lede {
    max-width: 760px;
    margin: 0 0 32px;
    color: var(--text-2);
    font-size: 17px;
    line-height: 1.6;
  }
  .grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px;
    gap: 20px;
    align-items: start;
  }
  .stage {
    position: relative;
    padding: 20px;
  }
  .boot {
    position: absolute;
    top: 16px;
    left: 16px;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    border: 1px solid var(--border);
    border-radius: 99px;
    background: var(--surface-solid);
    color: var(--text-2);
    font-size: 13px;
  }
  .boot.failed {
    border-color: var(--result);
  }
  .spin {
    width: 14px;
    height: 14px;
    border: 2px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .side {
    display: grid;
    gap: 16px;
  }
  .panel {
    padding: 16px;
  }
  .tiles {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px 12px;
    margin-bottom: 12px;
  }
  .tile {
    display: grid;
    gap: 2px;
  }
  .tile span {
    color: var(--muted);
    font-size: 12px;
  }
  .tile b {
    font: 600 22px var(--mono);
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .tile b.bad {
    color: var(--warn);
  }
  .tile b em {
    margin-left: 3px;
    font: 500 12px var(--mono);
    font-style: normal;
    color: var(--muted);
  }
  .tile small {
    color: var(--muted);
    font-size: 11.5px;
  }
  .controls {
    display: grid;
    gap: 12px;
  }
  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .row strong {
    display: block;
    font-size: 14px;
  }
  .row small {
    color: var(--muted);
    font-size: 12px;
  }
  .seg {
    display: inline-flex;
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
  .hint {
    margin: 0;
    color: var(--text-2);
    font-size: 13px;
    line-height: 1.5;
  }
  .switch {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    cursor: pointer;
  }
  .switch input {
    width: 18px;
    height: 18px;
    accent-color: var(--accent);
  }
  .small {
    height: 34px;
    padding: 0 12px;
    font-size: 13px;
  }
  .verify {
    width: 100%;
  }
  .verdict {
    margin: 0;
    padding: 10px 12px;
    border-radius: 10px;
    font-size: 13px;
    line-height: 1.5;
    background: var(--surface);
    border: 1px solid var(--border);
  }
  .verdict.ok b {
    color: var(--good);
  }
  .verdict.illegal b {
    color: var(--result);
  }
  .lower {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 20px;
    margin-top: 20px;
  }
  h2 {
    margin: 0 0 12px;
    font-size: 16px;
  }
  .log ol {
    display: grid;
    gap: 6px;
    margin: 0;
    padding: 0;
    list-style: none;
    font-size: 13px;
  }
  .log li {
    display: grid;
    grid-template-columns: 82px 1fr auto;
    gap: 10px;
    align-items: baseline;
  }
  .kind {
    font: 600 11px var(--mono);
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .failover .kind,
  .node-down .kind,
  .unavailable .kind {
    color: var(--warn);
  }
  .node-up .kind,
  .repair .kind {
    color: var(--good);
  }
  .msg {
    color: var(--text-2);
  }
  time {
    color: var(--muted);
    font: 11px var(--mono);
    white-space: nowrap;
  }
  .empty {
    margin: 0;
    color: var(--muted);
    font-size: 13px;
  }
  .how ul {
    display: grid;
    gap: 12px;
    margin: 0;
    padding: 0;
    list-style: none;
    color: var(--text-2);
    font-size: 14px;
    line-height: 1.55;
  }
  .how b {
    color: var(--text);
  }
  .how code {
    font: 12.5px var(--mono);
  }
  @media (max-width: 980px) {
    .grid,
    .lower {
      grid-template-columns: minmax(0, 1fr); /* 1fr would let wide content push the page sideways */
    }
  }
  @media (max-width: 640px) {
    .cluster {
      padding-top: calc(var(--nav-h) + 28px);
    }
    .stage {
      padding: 56px 12px 12px; /* room for the loading pill above the nodes */
    }
    .log li {
      grid-template-columns: 1fr auto;
    }
    .log .kind {
      grid-column: 1 / -1;
    }
  }
</style>
