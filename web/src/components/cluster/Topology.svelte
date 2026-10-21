<script lang="ts">
  // The cluster as a diagram: the router above its nodes, a line to each node carrying the
  // router's traffic (dots, sampled from real message counts), and an arc from each shard's
  // primary to its replica. Each node card shows its shard copies and the faults to inject.
  import { untrack } from "svelte";
  import type { ShardMap, SimLink, SimNode } from "../../lib/cluster/protocol";
  import { clusterSim } from "../../lib/cluster/sim.svelte";

  let { nodes, links, shards, ready = true }: { nodes: SimNode[]; links: SimLink[]; shards: ShardMap[]; ready?: boolean } = $props();

  const W = 1000;
  const H = 440;
  const ROUTER = { x: W / 2, y: 58 };
  const NODE_Y = 300;
  const xOf = (i: number) => (W / nodes.length) * (i + 0.5);
  const index = $derived(new Map(nodes.map((n, i) => [n.name, i])));
  const link = (from: string, to: string) => links.find((l) => l.from === from && l.to === to);

  // The router's view of each node: down, cut off, lossy or slow.
  function nodeState(n: SimNode) {
    const out = link("router", n.name);
    if (!n.up) return "down";
    if (out?.cut) return "cut";
    if (out && out.loss > 0) return "lossy";
    if (n.slow_ms > 0) return "slow";
    return "up";
  }

  // Traffic dots: each snapshot, new messages on a router link start a few dots along it.
  interface Dot {
    id: number;
    i: number;
    back: boolean;
    born: number;
  }
  let dots = $state<Dot[]>([]);
  let seq = 0;
  let previous = new Map<string, number>();
  const reduced = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  $effect(() => {
    const now = performance.now();
    const fresh: Dot[] = [];
    nodes.forEach((n, i) => {
      for (const back of [false, true]) {
        const l = back ? link(n.name, "router") : link("router", n.name);
        if (!l) continue;
        const key = `${l.from}>${l.to}`;
        const delta = l.sent - (previous.get(key) ?? l.sent);
        previous.set(key, l.sent);
        if (reduced || l.cut || !n.up) continue;
        for (let k = 0; k < Math.min(3, Math.ceil(delta / 12)); k++) fresh.push({ id: seq++, i, back, born: now + k * 80 });
      }
    });
    dots = [...untrack(() => dots).filter((d) => now - d.born < 700), ...fresh];
  });

  const ROLE = { primary: "P", replica: "R", none: "–" } as const;
</script>

<div class="frame">
  <div class="topology">
    <svg viewBox="0 0 {W} {H}" aria-hidden="true">
      <defs>
        <linearGradient id="wire" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="var(--accent)" stop-opacity="0.65" />
          <stop offset="1" stop-color="var(--accent)" stop-opacity="0.15" />
        </linearGradient>
      </defs>
      <!-- Replication: each shard's primary to its replica, below the nodes. -->
      {#each shards as s (s.id)}
        {@const a = index.get(s.primary)}
        {@const b = index.get(s.replica)}
        {#if a !== undefined && b !== undefined}
          {@const x1 = xOf(a)}
          {@const x2 = xOf(b)}
          {@const dip = 48 + 26 * Math.abs(a - b)}
          <path
            class="repl"
            class:lagging={!s.replica_in_sync}
            d="M{x1} {NODE_Y + 64} C {x1} {NODE_Y + 64 + dip}, {x2} {NODE_Y + 64 + dip}, {x2} {NODE_Y + 64}"
          />
        {/if}
      {/each}
      {#each nodes as n, i (n.name)}
        <line class="wire {nodeState(n)}" x1={ROUTER.x} y1={ROUTER.y + 26} x2={xOf(i)} y2={NODE_Y - 70} />
      {/each}
      {#each dots as d (d.id)}
        {@const x = xOf(d.i)}
        <circle
          class="dot"
          class:back={d.back}
          r="3.2"
          style:--x0="{d.back ? x : ROUTER.x}px"
          style:--y0="{d.back ? NODE_Y - 70 : ROUTER.y + 26}px"
          style:--x1="{d.back ? ROUTER.x : x}px"
          style:--y1="{d.back ? ROUTER.y + 26 : NODE_Y - 70}px"
          style:animation-delay="{Math.max(0, d.born - performance.now())}ms"
        />
      {/each}
    </svg>

    <div class="router" style:left="{(ROUTER.x / W) * 100}%" style:top="{(ROUTER.y / H) * 100}%">
      <strong>Router</strong>
      <span>Go · WebAssembly</span>
    </div>

    {#each nodes as n, i (n.name)}
      {@const st = nodeState(n)}
      <div class="node card {st}" style:left="{(xOf(i) / W) * 100}%" style:top="{(NODE_Y / H) * 100}%" style:--slot="{100 / nodes.length}%">
        <header>
          <strong>{n.name}</strong>
          <span class="pill" class:pending={!ready}>{!ready ? "Starting" : st === "up" ? "Up" : st === "down" ? "Down" : st === "cut" ? "Cut off" : st === "lossy" ? "Lossy" : `Slow ${n.slow_ms} ms`}</span>
        </header>
        <ul class="copies" aria-label="Shard copies on {n.name}">
          {#each n.shards ?? [] as c (c.shard)}
            <li class={c.role} title="Shard {c.shard}: {c.role}, epoch {c.epoch}, {c.vectors.toLocaleString()} vectors, log at {c.lsn}{c.role === 'replica' ? (c.in_sync ? ', in sync' : ', catching up') : ''}">
              <b>S{c.shard}</b><i>{ROLE[c.role]}</i><span class="num">e{c.epoch}</span>
            </li>
          {/each}
        </ul>
        <div class="actions">
          {#if n.up}
            <button onclick={() => clusterSim.fault({ kind: "kill", node: n.name })} disabled={!ready}>Kill</button>
          {:else}
            <button class="go" onclick={() => clusterSim.fault({ kind: "restart", node: n.name })}>Restart</button>
          {/if}
          <button onclick={() => clusterSim.fault({ kind: "isolate", node: n.name })} disabled={!ready || !n.up || st === "cut"}>Isolate</button>
          <button onclick={() => clusterSim.fault({ kind: "slow", node: n.name, ms: n.slow_ms ? 0 : 120 })} disabled={!ready || !n.up} aria-pressed={n.slow_ms > 0}>
            Slow
          </button>
        </div>
      </div>
    {/each}
  </div>
</div>

<style>
  /* The layout follows the room the diagram has, not the window: the page's side panel takes a
     share of the width, so a wide window can still leave too little for four nodes in a row. */
  .frame {
    container: topology / inline-size;
  }
  .topology {
    position: relative;
    aspect-ratio: 1000 / 440;
  }
  svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    overflow: visible;
  }
  .wire {
    stroke: url(#wire);
    stroke-width: 2;
    transition: stroke 0.3s;
  }
  .wire.down,
  .wire.cut {
    stroke: var(--result);
    stroke-dasharray: 6 7;
    stroke-opacity: 0.7;
  }
  .wire.lossy {
    stroke: var(--warn);
    stroke-dasharray: 2 5;
  }
  .repl {
    fill: none;
    stroke: var(--accent-2);
    stroke-width: 1.6;
    stroke-opacity: 0.55;
  }
  .repl.lagging {
    stroke: var(--warn);
    stroke-dasharray: 4 5;
  }
  .dot {
    fill: var(--accent-strong);
    opacity: 0;
    animation: travel 0.55s var(--ease) forwards;
  }
  .dot.back {
    fill: var(--good);
  }
  @keyframes travel {
    0% {
      opacity: 0;
      cx: var(--x0);
      cy: var(--y0);
    }
    15% {
      opacity: 1;
    }
    85% {
      opacity: 1;
    }
    100% {
      opacity: 0;
      cx: var(--x1);
      cy: var(--y1);
    }
  }
  .router,
  .node {
    position: absolute;
    transform: translate(-50%, -50%);
  }
  .router {
    display: grid;
    justify-items: center;
    gap: 2px;
    padding: 10px 18px;
    border: 1px solid var(--accent);
    border-radius: 12px;
    background: var(--surface-solid);
    box-shadow: 0 0 0 4px var(--accent-wash), 0 10px 30px -10px var(--accent-glow);
  }
  .router strong {
    font-size: 14px;
  }
  .router span {
    color: var(--muted);
    font: 11px var(--mono);
  }
  .node {
    box-sizing: border-box;
    width: min(calc(var(--slot) - 12px), 210px);
    padding: 10px;
    background: var(--surface-solid);
    transition: border-color 0.3s, opacity 0.3s;
  }
  .node.down {
    border-color: var(--result);
  }
  .node.down .copies {
    opacity: 0.4;
  }
  .node.cut,
  .node.lossy,
  .node.slow {
    border-color: var(--warn);
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    margin-bottom: 8px;
  }
  header strong {
    font: 600 13px var(--mono);
  }
  .pill {
    padding: 2px 8px;
    border-radius: 99px;
    background: var(--accent-wash);
    color: var(--good);
    font-size: 11.5px;
    white-space: nowrap;
  }
  .pill.pending {
    background: transparent;
    color: var(--muted);
    box-shadow: inset 0 0 0 1px var(--border-strong);
  }
  .down .pill {
    color: var(--result);
    background: transparent;
    box-shadow: inset 0 0 0 1px var(--result);
  }
  .cut .pill,
  .lossy .pill,
  .slow .pill {
    color: var(--warn);
    background: transparent;
    box-shadow: inset 0 0 0 1px var(--warn);
  }
  .copies {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    min-height: 22px; /* one row of chips, there before they arrive */
    margin: 0 0 8px;
    padding: 0;
    list-style: none;
  }
  .copies li {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font: 11px var(--mono);
    color: var(--text-2);
  }
  .copies li.primary {
    border-color: var(--accent);
    color: var(--text);
  }
  .copies i {
    font-style: normal;
    font-weight: 700;
    color: var(--accent-strong);
  }
  .copies li.replica i {
    color: var(--accent-2);
  }
  .copies span {
    color: var(--muted);
  }
  .actions {
    display: flex;
    gap: 4px;
  }
  .actions button {
    flex: 1;
    min-height: 30px;
    padding: 0 4px;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    background: var(--surface);
    color: var(--text);
    font-size: 12px;
    cursor: pointer;
  }
  .actions button:hover:not(:disabled) {
    border-color: var(--text-2);
  }
  .actions button.go {
    border-color: var(--good);
    color: var(--good);
  }
  .actions button[aria-pressed="true"] {
    border-color: var(--warn);
    color: var(--warn);
  }
  .actions button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  /* Under 740px a row of four would squeeze each node below its buttons' width (and the old
     min-height forced the diagram past the panel's edge): the nodes go two by two instead. */
  @container topology (max-width: 739px) {
    .topology {
      aspect-ratio: auto;
      min-height: 0;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    svg {
      display: none;
    }
    .router {
      position: static;
      grid-column: 1 / -1;
      transform: none;
      justify-self: center;
    }
    .node {
      position: static;
      width: auto;
      transform: none;
    }
  }
</style>
