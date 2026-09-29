<script lang="ts">
  import { engine, type SearchOutcome } from "../../lib/engine/engine.svelte";
  import { sceneController } from "../../lib/scene/scene.svelte";

  let { outcome }: { outcome: SearchOutcome } = $props();

  // Instrument readings after every event of the search, computed once per search so the HUD
  // follows the replay (or the debugger's scrubbing) at no cost.
  const readings = $derived.by(() => {
    const t = outcome.trace;
    const n = t.to.length;
    const hops = new Uint16Array(n);
    const best = new Float32Array(n);
    const heading = new Float32Array(n);
    const xy = engine.layout;
    let moves = 0;
    let closest = Infinity;
    let bearing = 0;
    for (let i = 0; i < n; i++) {
      if (t.accepted[i]) {
        moves++;
        if (xy) {
          const dx = xy[t.to[i] * 2] - xy[t.from[i] * 2];
          const dy = xy[t.to[i] * 2 + 1] - xy[t.from[i] * 2 + 1];
          if (dx || dy) bearing = ((Math.atan2(dx, -dy) * 180) / Math.PI + 360) % 360;
        }
      }
      closest = Math.min(closest, t.distance[i]);
      hops[i] = moves;
      best[i] = closest;
      heading[i] = bearing;
    }
    return { hops, best, heading, n };
  });

  const event = $derived(Math.min(sceneController.playhead.event, readings.n - 1));
  const started = $derived(event >= 0);
  const done = $derived(event >= readings.n - 1);
  const layer = $derived(started ? outcome.trace.layer[event] : outcome.trace.entryLayer);
  const hops = $derived(started ? readings.hops[event] : 0);
  const heading = $derived(started ? readings.heading[event] : 0);
  // How close the best photo found so far is to the final answer (cosine similarities; text and
  // image queries sit at different absolute levels, so the reading is relative to the answer).
  const final = $derived(Math.max(1e-6, 1 - outcome.trace.resultDistances[0]));
  const signal = $derived(started ? Math.min(1, Math.max(0, 1 - readings.best[event]) / final) : 0);
  const scanned = $derived(started ? event + 1 : 0);
  const cardinal = (deg: number) => ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round(deg / 45) % 8];
</script>

<aside class="hud" aria-label="Navigator instruments" class:done>
  <div class="compass" aria-hidden="true">
    <svg viewBox="-50 -50 100 100">
      <circle r="44" class="ring" />
      {#each Array.from({ length: 24 }, (_, i) => i * 15) as tick (tick)}
        <line
          y1={tick % 90 === 0 ? -44 : -44}
          y2={tick % 90 === 0 ? -35 : -40}
          transform="rotate({tick})"
          class:major={tick % 90 === 0}
        />
      {/each}
      <text y="-24">N</text>
      <g class="needle" style:transform="rotate({heading}deg)">
        <path d="M0 -30 L5 0 L0 6 L-5 0 Z" />
      </g>
      <circle r="3" class="hub" />
    </svg>
  </div>
  <dl class="num">
    <div><dt>Heading</dt><dd>{Math.round(heading).toString().padStart(3, "0")}° {cardinal(heading)}</dd></div>
    <div><dt>Layer</dt><dd>{layer}<small>{layer > 0 ? " express" : " ground"}</small></dd></div>
    <div><dt>Hops</dt><dd>{hops}</dd></div>
    <div><dt>Scanned</dt><dd>{scanned.toLocaleString()}</dd></div>
  </dl>
  <div class="signal">
    <span class="label">{done ? "Target locked" : "Closing in"}</span>
    <span class="bar"><i style:width="{Math.min(100, signal * 100)}%"></i></span>
    <span class="value num">{(signal * 100).toFixed(1)}%</span>
  </div>
</aside>

<style>
  .hud {
    position: fixed;
    top: calc(var(--nav-h) + 90px);
    right: 88px; /* beside the view controls column */
    z-index: 5;
    display: grid;
    gap: 12px;
    width: 196px;
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: color-mix(in srgb, var(--surface-solid) 55%, transparent);
    backdrop-filter: blur(14px);
    font: 12px var(--mono);
    pointer-events: none !important;
  }
  .compass {
    justify-self: center;
    width: 104px;
    height: 104px;
  }
  svg {
    width: 100%;
    height: 100%;
    overflow: visible;
  }
  .ring {
    fill: none;
    stroke: var(--border-strong);
  }
  line {
    stroke: var(--muted);
    stroke-width: 1;
  }
  line.major {
    stroke: var(--text-2);
    stroke-width: 1.6;
  }
  text {
    fill: var(--text-2);
    font: 600 9px var(--mono);
    text-anchor: middle;
    dominant-baseline: middle;
  }
  .needle {
    transition: transform 0.35s var(--ease);
  }
  .needle path {
    fill: var(--result);
    filter: drop-shadow(0 0 4px var(--result));
  }
  .hub {
    fill: var(--text);
  }
  dl {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 10px;
    margin: 0;
  }
  dt {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  dd {
    margin: 0;
    color: var(--text);
    font-size: 13px;
    font-weight: 600;
  }
  small {
    color: var(--muted);
    font-size: 10px;
    font-weight: 400;
  }
  .signal {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 5px 8px;
    align-items: center;
  }
  .signal .label {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .done .signal .label {
    color: var(--good);
  }
  .bar {
    grid-column: 1 / -1;
    grid-row: 2;
    height: 5px;
    border-radius: 99px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .bar i {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--good));
    transition: width 0.2s;
  }
  .value {
    color: var(--text);
    font-weight: 600;
  }
  /* The HUD needs the room beside the panel; small screens keep the debugger instead. */
  @media (max-width: 1180px), (max-height: 640px) {
    .hud {
      display: none;
    }
  }
</style>
