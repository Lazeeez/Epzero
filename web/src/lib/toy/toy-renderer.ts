// Canvas renderer for the landing page's HNSW explainer. It morphs between a flat, top-down map
// of the points and the layered picture from the HNSW paper (every layer a tilted plane stacked
// above the one below), and replays a real search trace over either. Everything a frame needs is
// an affine projection per layer, so the morph, drawing and hit-testing all share one code path.
import type { Trace } from "../engine/protocol";
import { onThemeChange, token } from "../theme";
import type { ToyGraph } from "./toy";

export interface ToyStage {
  /** 0 = flat map, 1 = stacked layers */
  tilt: number;
  /** Opacity of the base layer's links. */
  links: number;
  /** A node whose neighbourhood is highlighted (the "every point links to a few neighbours" step). */
  focus: number | null;
  /** How much of the trace may play: none, the upper-layer descent, or everything. */
  play: "none" | "descent" | "all";
  /** Clicks place a query. */
  interactive: boolean;
}

interface Affine {
  ax: number;
  bx: number;
  cx: number;
  ay: number;
  by: number;
  cy: number;
}

interface Colors {
  page: string;
  point: string;
  upper: string[];
  link: string;
  plane: string;
  planeEdge: string;
  path: string;
  visit: string;
  good: string;
  warn: string;
  text: string;
}

const LAYER_TOKENS = ["--series-1", "--accent-2", "--series-3", "--series-4"];

export class ToyRenderer {
  private ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private dpr = 1;
  private colors!: Colors;
  private frame = 0;
  private last = 0;
  private reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Animated values chase their targets.
  private tilt = 0;
  private links = 0;
  private stage: ToyStage = { tilt: 0, links: 0, focus: null, play: "none", interactive: false };

  private trace: Trace | null = null;
  private query: [number, number] | null = null;
  private descentEvents = 0;
  /** Events revealed so far (fractional while animating). */
  private playhead = 0;
  private fast = false;
  private ghost: [number, number] | null = null;
  private pulse = 0;

  private queryListener: ((x: number, y: number) => void) | null = null;
  private progressListener: ((revealed: number, total: number) => void) | null = null;
  private cleanups: (() => void)[] = [];

  constructor(
    private canvas: HTMLCanvasElement,
    private graph: ToyGraph,
  ) {
    this.ctx = canvas.getContext("2d")!;
    this.readColors();
    const resize = new ResizeObserver(() => this.resize());
    resize.observe(canvas);
    this.cleanups.push(() => resize.disconnect());
    this.cleanups.push(onThemeChange(() => (this.readColors(), this.invalidate())));

    const onMove = (event: PointerEvent) => {
      if (!this.stage.interactive) return;
      this.ghost = this.unproject(event);
      this.invalidate();
    };
    const onLeave = () => {
      this.ghost = null;
      this.invalidate();
    };
    const onClick = (event: PointerEvent) => {
      if (!this.stage.interactive) return;
      const point = this.unproject(event);
      if (point) this.queryListener?.(point[0], point[1]);
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("click", onClick as EventListener);
    this.cleanups.push(() => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("click", onClick as EventListener);
    });
    this.resize();
  }

  destroy() {
    cancelAnimationFrame(this.frame);
    for (const cleanup of this.cleanups) cleanup();
  }

  onQuery(listener: (x: number, y: number) => void) {
    this.queryListener = listener;
  }

  onProgress(listener: (revealed: number, total: number) => void) {
    this.progressListener = listener;
  }

  setStage(stage: ToyStage) {
    this.stage = stage;
    if (this.reducedMotion) {
      this.tilt = stage.tilt;
      this.links = stage.links;
    }
    this.invalidate();
  }

  /** Shows a search. `restart` replays it from the first event; otherwise playback continues. */
  setTrace(query: [number, number], trace: Trace, { restart = true, fast = false } = {}) {
    this.query = query;
    this.trace = trace;
    this.fast = fast;
    let descent = 0;
    while (descent < trace.layer.length && trace.layer[descent] > 0) descent++;
    this.descentEvents = descent;
    if (restart) this.playhead = 0;
    if (this.reducedMotion) this.playhead = this.cap();
    this.invalidate();
  }

  private cap(): number {
    if (!this.trace) return 0;
    if (this.stage.play === "none") return 0;
    if (this.stage.play === "descent") return this.descentEvents;
    return this.trace.layer.length;
  }

  /** Milliseconds per event: the descent is slow enough to follow, the beam search a sweep. */
  private eventMs(index: number): number {
    const trace = this.trace!;
    const beam = trace.layer.length - this.descentEvents;
    if (index < this.descentEvents) return this.fast ? 90 : 260;
    return Math.min(this.fast ? 14 : 45, (this.fast ? 1400 : 3200) / Math.max(beam, 1));
  }

  invalidate() {
    if (this.frame) return;
    this.last = 0;
    this.frame = requestAnimationFrame((now) => this.tick(now));
  }

  private tick(now: number) {
    this.frame = 0;
    // Real elapsed time (capped after a stall, e.g. a background tab) so slow devices keep pace.
    const dt = this.last ? Math.min(now - this.last, 250) : 16;
    this.last = now;
    let animating = false;

    const approach = (value: number, target: number, rate: number) => {
      const next = target + (value - target) * Math.exp(-dt * rate);
      if (Math.abs(next - target) < 0.002) return target;
      animating = true;
      return next;
    };
    this.tilt = approach(this.tilt, this.stage.tilt, 0.0042);
    this.links = approach(this.links, this.stage.links, 0.005);

    const cap = this.cap();
    if (this.trace && this.playhead < cap) {
      let budget = dt;
      while (budget > 0 && this.playhead < cap) {
        const index = Math.floor(this.playhead);
        const ms = this.eventMs(index);
        const step = Math.min(budget / ms, index + 1 - this.playhead);
        this.playhead += step;
        budget -= step * ms;
        if (this.playhead >= index + 1 - 1e-9) this.playhead = index + 1;
      }
      this.playhead = Math.min(this.playhead, cap);
      animating = true;
    } else if (this.playhead > cap) {
      this.playhead = cap;
    }
    if (this.stage.focus !== null) {
      this.pulse = (this.pulse + dt / 1400) % 1;
      animating = true;
    }

    this.draw();
    this.progressListener?.(Math.floor(this.playhead), this.trace?.layer.length ?? 0);
    if (animating) this.frame = requestAnimationFrame((t) => this.tick(t));
  }

  private readColors() {
    this.colors = {
      page: token("--page"),
      point: token("--text-2"),
      upper: LAYER_TOKENS.map(token),
      link: token("--muted"),
      plane: token("--accent"),
      planeEdge: token("--border-strong"),
      path: token("--result"),
      visit: token("--accent-strong"),
      good: token("--good"),
      warn: token("--warn"),
      text: token("--text"),
    };
  }

  private resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.dpr = Math.min(devicePixelRatio || 1, 2);
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = Math.round(rect.width * this.dpr);
    this.canvas.height = Math.round(rect.height * this.dpr);
    this.invalidate();
  }

  /** Screen transform for points on `layer` at the current morph. */
  private affine(layer: number): Affine {
    const w = this.width;
    const h = this.height;
    const top = this.graph.maxLevel;
    const t = easeInOut(this.tilt);
    // Flat: the unit square, centred.
    const side = Math.min(w, h) - 32;
    const fx = (w - side) / 2;
    const fy = (h - side) / 2;
    // Stacked: each layer a rhombus (x - y across, x + y down), raised by `gap` per layer.
    // Stack height is 2b (one rhombus) + top * gap; keep it inside 90% of the canvas.
    const a = Math.min(w * 0.46, (h * 0.9) / (0.42 * (2 + top)));
    const b = a * 0.42;
    const gap = b;
    const cxIso = w / 2;
    const cyIso = h / 2 - b + (top / 2 - layer) * gap;
    return {
      ax: lerp(side, a, t),
      bx: lerp(0, -a, t),
      cx: lerp(fx, cxIso, t),
      ay: lerp(0, b, t),
      by: lerp(side, b, t),
      cy: lerp(fy, cyIso, t),
    };
  }

  private unproject(event: PointerEvent): [number, number] | null {
    const rect = this.canvas.getBoundingClientRect();
    const X = event.clientX - rect.left;
    const Y = event.clientY - rect.top;
    const m = this.affine(0);
    const det = m.ax * m.by - m.bx * m.ay;
    if (Math.abs(det) < 1e-6) return null;
    const x = ((X - m.cx) * m.by - m.bx * (Y - m.cy)) / det;
    const y = (m.ax * (Y - m.cy) - m.ay * (X - m.cx)) / det;
    if (x < -0.02 || x > 1.02 || y < -0.02 || y > 1.02) return null;
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
  }

  private draw() {
    const { ctx, graph, colors } = this;
    const pts = graph.points;
    const top = graph.maxLevel;
    const t = easeInOut(this.tilt);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    const layers = Array.from({ length: top + 1 }, (_, l) => this.affine(l));
    const px = (m: Affine, i: number) => m.ax * pts[i * 2] + m.bx * pts[i * 2 + 1] + m.cx;
    const py = (m: Affine, i: number) => m.ay * pts[i * 2] + m.by * pts[i * 2 + 1] + m.cy;

    const trace = this.trace;
    const revealed = trace ? Math.floor(this.playhead) : 0;

    for (let layer = 0; layer <= top; layer++) {
      const m = layers[layer];
      const presence = layer === 0 ? 1 : t;
      if (presence < 0.01) continue;

      // The layer's plane.
      if (t > 0.01) {
        ctx.globalAlpha = t;
        ctx.beginPath();
        const corner = (x: number, y: number) => [m.ax * x + m.bx * y + m.cx, m.ay * x + m.by * y + m.cy] as const;
        const pad = 0.03;
        ctx.moveTo(...corner(-pad, -pad));
        ctx.lineTo(...corner(1 + pad, -pad));
        ctx.lineTo(...corner(1 + pad, 1 + pad));
        ctx.lineTo(...corner(-pad, 1 + pad));
        ctx.closePath();
        const fill = ctx.createLinearGradient(m.cx, m.cy, m.cx, m.cy + m.by + m.ay);
        fill.addColorStop(0, withAlpha(colors.plane, 0.02));
        fill.addColorStop(1, withAlpha(colors.plane, 0.07));
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = colors.planeEdge;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // The layer's links.
      const linkAlpha = layer === 0 ? this.links * (trace && revealed > 0 ? 0.5 : 1) : t;
      if (linkAlpha > 0.01) {
        const edges = graph.edges[layer];
        ctx.beginPath();
        for (let e = 0; e < edges.length; e += 2) {
          ctx.moveTo(px(m, edges[e]), py(m, edges[e]));
          ctx.lineTo(px(m, edges[e + 1]), py(m, edges[e + 1]));
        }
        ctx.strokeStyle = layer === 0 ? colors.link : colors.upper[(layer - 1) % colors.upper.length];
        ctx.globalAlpha = linkAlpha * (layer === 0 ? 0.32 : 0.55);
        ctx.lineWidth = layer === 0 ? 0.7 : 1.1;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // The layer's nodes.
      ctx.globalAlpha = presence;
      ctx.fillStyle = layer === 0 ? colors.point : colors.upper[(layer - 1) % colors.upper.length];
      const r = layer === 0 ? 1.7 : 2.8 + layer * 0.3;
      ctx.beginPath();
      for (let i = 0; i < graph.levels.length; i++) {
        if (graph.levels[i] < layer) continue;
        const x = px(m, i);
        const y = py(m, i);
        ctx.moveTo(x + r, y);
        ctx.arc(x, y, r, 0, Math.PI * 2);
      }
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    this.drawFocus(layers[0], px, py);
    if (trace && this.query) this.drawTrace(trace, revealed, layers, px, py, t);
    if (this.ghost && this.stage.interactive) {
      const m = layers[0];
      const x = m.ax * this.ghost[0] + m.bx * this.ghost[1] + m.cx;
      const y = m.ay * this.ghost[0] + m.by * this.ghost[1] + m.cy;
      ctx.strokeStyle = colors.path;
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
  }

  private drawFocus(m: Affine, px: (m: Affine, i: number) => number, py: (m: Affine, i: number) => number) {
    const focus = this.stage.focus;
    if (focus === null) return;
    const { ctx, colors } = this;
    const edges = this.graph.edges[0];
    const fx = px(m, focus);
    const fy = py(m, focus);
    ctx.strokeStyle = colors.path;
    ctx.fillStyle = colors.path;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let e = 0; e < edges.length; e += 2) {
      const other = edges[e] === focus ? edges[e + 1] : edges[e + 1] === focus ? edges[e] : -1;
      if (other < 0) continue;
      ctx.moveTo(fx, fy);
      ctx.lineTo(px(m, other), py(m, other));
      ctx.moveTo(px(m, other) + 3, py(m, other));
      ctx.arc(px(m, other), py(m, other), 3, 0, Math.PI * 2);
    }
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(fx, fy, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1 - this.pulse;
    ctx.beginPath();
    ctx.arc(fx, fy, 5 + this.pulse * 18, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  private drawTrace(
    trace: Trace,
    revealed: number,
    layers: Affine[],
    px: (m: Affine, i: number) => number,
    py: (m: Affine, i: number) => number,
    t: number,
  ) {
    const { ctx, colors } = this;
    const [qx, qy] = this.query!;
    const qScreen = (layer: number) => {
      const m = layers[layer];
      return [m.ax * qx + m.bx * qy + m.cx, m.ay * qx + m.by * qy + m.cy] as const;
    };

    // The query, pinned on every layer it passes through, joined by a plumb line.
    const top = Math.min(trace.entryLayer, layers.length - 1);
    if (t > 0.05) {
      ctx.globalAlpha = 0.45 * t;
      ctx.strokeStyle = colors.path;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(...qScreen(top));
      ctx.lineTo(...qScreen(0));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (let layer = 0; layer <= top; layer++) {
      if (layer > 0 && t < 0.05) break;
      const [x, y] = qScreen(layer);
      ctx.globalAlpha = layer === 0 ? 1 : t;
      drawQueryMark(ctx, x, y, colors.path, colors.page);
    }
    ctx.globalAlpha = 1;

    // Entry point.
    const entry = trace.entryLabel;
    const entryM = layers[top];
    ctx.strokeStyle = colors.text;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(px(entryM, entry), py(entryM, entry), 7, 0, Math.PI * 2);
    ctx.stroke();

    // Rejected candidates: faint; visited nodes: lit; accepted moves: the bright path.
    const layerOf = (i: number) => Math.min(trace.layer[i], layers.length - 1);
    ctx.lineWidth = 1;
    ctx.strokeStyle = colors.path;
    ctx.globalAlpha = 0.22;
    ctx.beginPath();
    for (let i = 0; i < revealed; i++) {
      if (trace.accepted[i]) continue;
      const m = layers[layerOf(i)];
      ctx.moveTo(px(m, trace.from[i]), py(m, trace.from[i]));
      ctx.lineTo(px(m, trace.to[i]), py(m, trace.to[i]));
    }
    ctx.stroke();

    ctx.globalAlpha = 0.9;
    ctx.fillStyle = colors.visit;
    ctx.beginPath();
    for (let i = 0; i < revealed; i++) {
      const m = layers[layerOf(i)];
      const x = px(m, trace.to[i]);
      const y = py(m, trace.to[i]);
      ctx.moveTo(x + 2.6, y);
      ctx.arc(x, y, 2.6, 0, Math.PI * 2);
    }
    ctx.fill();

    ctx.globalAlpha = 1;
    ctx.strokeStyle = colors.path;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let i = 0; i < revealed; i++) {
      if (!trace.accepted[i]) continue;
      const m = layers[layerOf(i)];
      ctx.moveTo(px(m, trace.from[i]), py(m, trace.from[i]));
      ctx.lineTo(px(m, trace.to[i]), py(m, trace.to[i]));
    }
    ctx.stroke();

    // Where the search drops from one layer to the next, following the same node down.
    if (t > 0.05) {
      ctx.setLineDash([4, 4]);
      ctx.globalAlpha = t;
      ctx.beginPath();
      for (let i = 1; i < revealed; i++) {
        const above = layerOf(i - 1);
        const here = layerOf(i);
        if (here >= above) continue;
        const node = trace.from[i];
        ctx.moveTo(px(layers[above], node), py(layers[above], node));
        ctx.lineTo(px(layers[here], node), py(layers[here], node));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // The newest event, ringed.
    if (revealed > 0 && revealed < trace.layer.length) {
      const i = revealed - 1;
      const m = layers[layerOf(i)];
      ctx.strokeStyle = colors.text;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px(m, trace.to[i]), py(m, trace.to[i]), 6, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Finished: the answer, and any true neighbour it missed.
    if (revealed >= trace.layer.length) {
      const m = layers[0];
      const found = new Set(trace.results);
      ctx.lineWidth = 2;
      ctx.strokeStyle = colors.good;
      ctx.beginPath();
      for (const id of trace.results) {
        ctx.moveTo(px(m, id) + 6, py(m, id));
        ctx.arc(px(m, id), py(m, id), 6, 0, Math.PI * 2);
      }
      ctx.stroke();
      ctx.strokeStyle = colors.warn;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (const id of trace.exact) {
        if (found.has(id)) continue;
        ctx.moveTo(px(m, id) + 6, py(m, id));
        ctx.arc(px(m, id), py(m, id), 6, 0, Math.PI * 2);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.lineCap = "butt";
  }
}

function drawQueryMark(ctx: CanvasRenderingContext2D, x: number, y: number, color: string, page: string) {
  ctx.fillStyle = page;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y - 7);
  ctx.lineTo(x + 7, y);
  ctx.lineTo(x, y + 7);
  ctx.lineTo(x - 7, y);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 2.2, 0, Math.PI * 2);
  ctx.fill();
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** `color` (hex or rgb[a]) with a new alpha. */
function withAlpha(color: string, alpha: number): string {
  const c = color.trim();
  if (c.startsWith("#")) {
    const hex = c.length === 4 ? [...c.slice(1)].map((d) => d + d).join("") : c.slice(1, 7);
    const n = parseInt(hex, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  }
  const parts = c.match(/[\d.]+/g) ?? ["0", "0", "0"];
  return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
}
