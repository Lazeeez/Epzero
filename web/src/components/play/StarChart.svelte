<script lang="ts">
  // The game's star chart: every photo as a faint star at its 2-D map position (UMAP of the CLIP
  // vectors, so nearby stars are similar photos), with the target, the player's route, the current
  // neighbourhood and, after the round, the route Epzero's own search took.
  import { engine } from "../../lib/engine/engine.svelte";
  import type { Trace } from "../../lib/engine/protocol";
  import { onThemeChange, token } from "../../lib/theme";

  let {
    target,
    goals,
    path,
    neighbors,
    highlight = null,
    epzero = null,
  }: {
    target: number;
    goals: Set<number>;
    path: number[];
    neighbors: ArrayLike<number>;
    highlight?: number | null;
    epzero?: Trace | null;
  } = $props();

  let canvas: HTMLCanvasElement | undefined = $state();
  let size = $state({ w: 0, h: 0 });
  let themeTick = $state(0);
  let background: HTMLCanvasElement | null = null;
  // Theme colors, read once per theme (getComputedStyle every frame would be wasteful).
  let colors = { accent: "", route: "", good: "", text: "" };
  let pulse = 0;
  let frame = 0;

  $effect(() => onThemeChange(() => themeTick++));

  $effect(() => {
    if (!canvas) return;
    const observer = new ResizeObserver(([entry]) => {
      size = { w: entry.contentRect.width, h: entry.contentRect.height };
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  });

  const pad = 14;
  function place(id: number, w: number, h: number): [number, number] {
    const xy = engine.layout!;
    const side = Math.min(w, h) - pad * 2;
    const ox = (w - side) / 2;
    const oy = (h - side) / 2;
    return [ox + ((xy[id * 2] + 1) / 2) * side, oy + ((xy[id * 2 + 1] + 1) / 2) * side];
  }

  // The star field is static: render it once per size/theme into an offscreen canvas.
  function renderBackground(w: number, h: number, dpr: number) {
    const bg = document.createElement("canvas");
    bg.width = Math.round(w * dpr);
    bg.height = Math.round(h * dpr);
    const ctx = bg.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const dim = token("--muted");
    const bright = token("--text-2");
    for (let i = 0; i < engine.count; i++) {
      const [x, y] = place(i, w, h);
      // Deterministic twinkle: a few brighter stars among the dust.
      const hash = (Math.imul(i, 2654435761) >>> 0) / 4294967296;
      ctx.globalAlpha = hash > 0.97 ? 0.9 : 0.35 + hash * 0.25;
      ctx.fillStyle = hash > 0.97 ? bright : dim;
      const r = hash > 0.97 ? 1.4 : 0.8;
      ctx.fillRect(x - r / 2, y - r / 2, r, r);
    }
    return bg;
  }

  $effect(() => {
    void themeTick;
    const { w, h } = size;
    if (!canvas || !w || !h || !engine.layout) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    background = renderBackground(w, h, dpr);
    colors = { accent: token("--accent"), route: token("--result"), good: token("--good"), text: token("--text") };
    draw();
  });

  // Redraw on any game state change; keep animating the pulse on the current star.
  $effect(() => {
    void path.length;
    void neighbors;
    void highlight;
    void epzero;
    void goals;
    draw();
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) return;
    const tick = () => {
      pulse = (pulse + 0.012) % 1;
      draw();
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  });

  function draw() {
    if (!canvas || !background || !size.w) return;
    const ctx = canvas.getContext("2d")!;
    const { w, h } = size;
    const dpr = canvas.width / w;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(background, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { accent, route, good, text } = colors;

    // Epzero's route (revealed after the round): every accepted hop, all layers flattened.
    if (epzero) {
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.6;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      for (let i = 0; i < epzero.to.length; i++) {
        if (!epzero.accepted[i]) continue;
        const [ax, ay] = place(epzero.from[i], w, h);
        const [bx, by] = place(epzero.to[i], w, h);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
      ctx.stroke();
      const [ex, ey] = place(epzero.entryLabel, w, h);
      ctx.fillStyle = accent;
      ctx.beginPath();
      ctx.arc(ex, ey, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // The goal region: the target's true nearest neighbours.
    ctx.globalAlpha = 0.8;
    ctx.strokeStyle = good;
    ctx.lineWidth = 1;
    for (const id of goals) {
      if (id === target) continue;
      const [x, y] = place(id, w, h);
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.stroke();
    }

    // The current neighbourhood: where each option would take you.
    const current = path[path.length - 1];
    const [cx, cy] = place(current, w, h);
    ctx.globalAlpha = 0.35;
    ctx.strokeStyle = text;
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (let i = 0; i < neighbors.length; i++) {
      const [x, y] = place(neighbors[i], w, h);
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
    if (highlight !== null) {
      const [x, y] = place(highlight, w, h);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Your route.
    ctx.globalAlpha = 1;
    ctx.strokeStyle = route;
    ctx.lineWidth = 2.2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    path.forEach((id, i) => {
      const [x, y] = place(id, w, h);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = route;
    for (const id of path) {
      const [x, y] = place(id, w, h);
      ctx.beginPath();
      ctx.arc(x, y, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }

    // You are here (pulsing), and the target (a four-point star).
    ctx.strokeStyle = route;
    ctx.globalAlpha = 1 - pulse;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, 5 + pulse * 16, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
    const [tx, ty] = place(target, w, h);
    star(ctx, tx, ty, 9, good);
  }

  function star(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string) {
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.fillStyle = color;
    ctx.beginPath();
    for (let i = 0; i < 8; i++) {
      const angle = (i * Math.PI) / 4 - Math.PI / 2;
      const radius = i % 2 === 0 ? r : r * 0.32;
      ctx.lineTo(x + Math.cos(angle) * radius, y + Math.sin(angle) * radius);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
</script>

<canvas bind:this={canvas} aria-label="Star chart of the collection: the target, your route and your current options"></canvas>

<style>
  canvas {
    display: block;
    width: 100%;
    height: auto; /* square by aspect ratio: 100% height would take the whole card and push the legend out */
    aspect-ratio: 1;
  }
</style>
