<script lang="ts">
  import { engine, type SearchOutcome } from "../../lib/engine/engine.svelte";
  import { agreement, touched } from "../../lib/engine/trace-analysis";
  import { token } from "../../lib/theme";

  let { outcome, onclose }: { outcome: SearchOutcome; onclose: () => void } = $props();

  const EXACT_MS = 5000; // the exact scan's replay length; HNSW gets the same slowdown factor
  const trace = $derived(outcome.trace);
  const visited = $derived(touched(trace));
  const ratio = $derived(trace.micros / Math.max(trace.exactMicros, 1));
  const hnswMs = $derived(Math.max(EXACT_MS * ratio, 120));
  const slowdown = $derived(EXACT_MS / (trace.exactMicros / 1000));
  const hits = $derived(agreement(trace));

  let exactCanvas: HTMLCanvasElement | undefined = $state();
  let hnswCanvas: HTMLCanvasElement | undefined = $state();
  let elapsed = $state(0);
  let frame = 0;
  let dialog: HTMLDivElement | undefined = $state();

  const exactDone = $derived(elapsed >= EXACT_MS);
  const hnswDone = $derived(elapsed >= hnswMs);
  const exactCount = $derived(Math.min(engine.count, Math.floor((elapsed / EXACT_MS) * engine.count)));
  const hnswCount = $derived(Math.min(visited.length, Math.floor((elapsed / hnswMs) * visited.length)));
  const clock = (fraction: number, micros: number) => {
    const us = Math.min(fraction, 1) * micros;
    return us >= 1000 ? `${(us / 1000).toFixed(2)} ms` : `${us.toFixed(0)} µs`;
  };

  // Each panel: a pre-rendered map of every photo, with lit dots added incrementally per frame.
  interface Panel {
    ctx: CanvasRenderingContext2D;
    size: number;
    drawn: number;
  }

  function setup(canvas: HTMLCanvasElement): Panel {
    const size = canvas.clientWidth;
    const dpr = Math.min(devicePixelRatio, 2);
    canvas.width = canvas.height = size * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = token("--surface-solid");
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = token("--axis");
    const xy = engine.layout!;
    for (let i = 0; i < engine.count; i++) {
      const [x, y] = project(xy, i, size);
      ctx.fillRect(x, y, 1.2, 1.2);
    }
    return { ctx, size, drawn: 0 };
  }

  function project(xy: Float32Array, i: number, size: number): [number, number] {
    const pad = 10;
    return [pad + ((xy[i * 2] + 1) / 2) * (size - 2 * pad), pad + ((xy[i * 2 + 1] + 1) / 2) * (size - 2 * pad)];
  }

  function light(panel: Panel, ids: ArrayLike<number>, upTo: number, color: string, dot: number) {
    const xy = engine.layout!;
    panel.ctx.fillStyle = color;
    for (let i = panel.drawn; i < upTo; i++) {
      const [x, y] = project(xy, ids[i], panel.size);
      panel.ctx.fillRect(x - dot / 2, y - dot / 2, dot, dot);
    }
    panel.drawn = Math.max(panel.drawn, upTo);
  }

  function ring(panel: Panel, ids: ArrayLike<number>, color: string) {
    const xy = engine.layout!;
    panel.ctx.strokeStyle = color;
    panel.ctx.lineWidth = 1.6;
    for (let i = 0; i < ids.length; i++) {
      const [x, y] = project(xy, ids[i], panel.size);
      panel.ctx.beginPath();
      panel.ctx.arc(x, y, 5, 0, Math.PI * 2);
      panel.ctx.stroke();
    }
  }

  function start() {
    if (!exactCanvas || !hnswCanvas || !engine.layout) return;
    cancelAnimationFrame(frame);
    const exact = setup(exactCanvas);
    const hnsw = setup(hnswCanvas);
    const order = Uint32Array.from({ length: engine.count }, (_, i) => i);
    const scanColor = token("--text-2");
    const hnswColor = token("--accent");
    const resultColor = token("--result");
    let t0 = 0;
    let exactRinged = false;
    let hnswRinged = false;
    const tick = (now: number) => {
      t0 ||= now;
      elapsed = now - t0;
      light(exact, order, Math.min(engine.count, Math.floor((elapsed / EXACT_MS) * engine.count)), scanColor, 1.4);
      light(hnsw, visited, Math.min(visited.length, Math.floor((elapsed / hnswMs) * visited.length)), hnswColor, 2.4);
      if (!hnswRinged && elapsed >= hnswMs) {
        ring(hnsw, trace.results, resultColor);
        hnswRinged = true;
      }
      if (!exactRinged && elapsed >= EXACT_MS) {
        ring(exact, trace.exact, resultColor);
        exactRinged = true;
      }
      if (elapsed < EXACT_MS + 50) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
  }

  $effect(() => {
    void outcome;
    if (exactCanvas && hnswCanvas) start();
    return () => cancelAnimationFrame(frame);
  });

  $effect(() => {
    dialog?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onclose();
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  });
</script>

<div class="backdrop" onclick={onclose} role="presentation"></div>
<div class="race card" role="dialog" aria-modal="true" aria-labelledby="race-title" tabindex="-1" bind:this={dialog}>
  <header>
    <div>
      <h2 id="race-title">HNSW vs. an exact scan, same query</h2>
      <p>Each dot is one of the {engine.count.toLocaleString()} photos, placed by visual similarity. Replayed ~{slowdown.toFixed(0)}× slower than real time, with the true ratio between the two.</p>
    </div>
    <button class="close" onclick={onclose} aria-label="Close">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" /></svg>
    </button>
  </header>

  <div class="panels">
    <figure>
      <canvas bind:this={exactCanvas} aria-label="Exact scan progress"></canvas>
      <figcaption>
        <span class="name">Exact scan</span>
        <span class="clock num">{clock(elapsed / EXACT_MS, trace.exactMicros)}</span>
        <span class="sub num">{exactCount.toLocaleString()} / {engine.count.toLocaleString()} compared {exactDone ? "· done" : ""}</span>
      </figcaption>
    </figure>
    <figure>
      <canvas bind:this={hnswCanvas} aria-label="HNSW search progress"></canvas>
      <figcaption>
        <span class="name accent">Oryx HNSW</span>
        <span class="clock num">{clock(elapsed / hnswMs, trace.micros)}</span>
        <span class="sub num">
          {hnswCount.toLocaleString()} compared ({((visited.length / engine.count) * 100).toFixed(1)}%) {hnswDone ? "· done" : ""}
        </span>
      </figcaption>
    </figure>
  </div>

  <footer>
    <p class:show={exactDone}>
      Both return the top {trace.results.length}: <strong class="num">{hits}/{trace.results.length}</strong> identical, with HNSW
      <strong class="num">{(trace.exactMicros / Math.max(trace.micros, 1)).toFixed(0)}×</strong> faster.
    </p>
    <button class="btn btn-ghost" onclick={start}>Replay</button>
  </footer>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    z-index: 80;
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(6px);
    animation: fade 0.25s var(--ease);
  }
  .race {
    position: fixed;
    top: 50%;
    left: 50%;
    z-index: 81;
    display: grid;
    gap: 18px;
    width: min(1020px, 100% - 32px);
    max-height: calc(100dvh - 32px);
    overflow: auto;
    padding: 24px;
    background: var(--surface-solid);
    transform: translate(-50%, -50%);
    animation: pop 0.35s var(--ease);
  }
  header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
  }
  h2 {
    font-size: 20px;
  }
  header p {
    max-width: 640px;
    margin: 6px 0 0;
    color: var(--text-2);
    font-size: 13.5px;
  }
  .close {
    display: grid;
    place-items: center;
    flex: none;
    width: 34px;
    height: 34px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    color: var(--text-2);
    cursor: pointer;
  }
  .close svg {
    width: 16px;
    height: 16px;
    stroke: currentColor;
    stroke-width: 2;
    stroke-linecap: round;
  }
  .panels {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  figure {
    margin: 0;
  }
  canvas {
    display: block;
    width: 100%;
    aspect-ratio: 1;
    border: 1px solid var(--border);
    border-radius: 14px;
  }
  figcaption {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 2px 12px;
    padding: 10px 2px 0;
  }
  .name {
    font-weight: 600;
  }
  .name.accent {
    color: var(--accent-strong);
  }
  .clock {
    font: 600 20px var(--mono);
    text-align: right;
  }
  .sub {
    grid-column: 1 / -1;
    color: var(--muted);
    font-size: 12.5px;
  }
  footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }
  footer p {
    margin: 0;
    color: var(--text-2);
    opacity: 0;
    transition: opacity 0.4s;
  }
  footer p.show {
    opacity: 1;
  }
  footer strong {
    color: var(--good);
  }
  @keyframes fade {
    from {
      opacity: 0;
    }
  }
  @keyframes pop {
    from {
      opacity: 0;
      transform: translate(-50%, -48%) scale(0.98);
    }
  }
  @media (max-width: 640px) {
    .panels {
      grid-template-columns: 1fr;
    }
    .race {
      padding: 16px;
    }
  }
</style>
