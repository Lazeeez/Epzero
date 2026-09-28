import type { SweepRow } from "./protocol";

const SVG = "http://www.w3.org/2000/svg";
const RECALL_TICKS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999];
const MAX_RECALL = 0.9995; // 1.0 is infinitely far right on a "nines" axis

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  parent?.appendChild(node);
  return node;
}

// Positions recall by -log10(1 - recall), so 0.9, 0.99, 0.999 are evenly spaced and the high-recall
// region is readable.
const nines = (recall: number) => -Math.log10(1 - Math.min(recall, MAX_RECALL));

function nice125(value: number, up: boolean): number {
  const decade = 10 ** Math.floor(Math.log10(value));
  const steps = [1, 2, 5, 10].map((m) => m * decade);
  return up ? steps.find((s) => s >= value)! : [...steps].reverse().find((s) => s <= value)!;
}

export const formatQps = (q: number) => (q >= 10_000 ? `${(q / 1000).toFixed(1)}k` : Math.round(q).toLocaleString());
export const formatUs = (us: number) => (us >= 1000 ? `${(us / 1000).toFixed(2)} ms` : `${us.toFixed(0)} µs`);

export class RecallChart {
  private rows: SweepRow[] = [];
  private baseline: number | null = null;
  private readonly tooltip: HTMLDivElement;

  constructor(private readonly container: HTMLElement) {
    this.tooltip = document.createElement("div");
    this.tooltip.className = "tooltip";
    this.tooltip.hidden = true;
    new ResizeObserver(() => this.render()).observe(container);
  }

  update(rows: SweepRow[], baseline: number | null) {
    this.rows = [...rows].sort((a, b) => a.recall - b.recall);
    this.baseline = baseline;
    this.render();
  }

  private render() {
    const width = Math.max(this.container.clientWidth, 280);
    if (this.rows.length === 0 || width === 0) return;
    const compact = width < 560;
    const height = compact ? 300 : 380;
    const m = { top: 16, right: compact ? 12 : 20, bottom: 48, left: 54 };
    const pw = width - m.left - m.right;
    const ph = height - m.top - m.bottom;

    const recalls = this.rows.map((r) => r.recall);
    const qps = this.rows.map((r) => r.qps);
    if (this.baseline) qps.push(this.baseline);
    const xMin = [...RECALL_TICKS].reverse().find((t) => t <= Math.min(...recalls)) ?? 0.5;
    const xMax = Math.max(RECALL_TICKS.find((t) => t >= Math.max(...recalls)) ?? MAX_RECALL, 0.99);
    const yMin = nice125(Math.min(...qps) * 0.9, false);
    const yMax = nice125(Math.max(...qps) * 1.1, true);
    const sx = (r: number) => m.left + ((nines(r) - nines(xMin)) / (nines(xMax) - nines(xMin))) * pw;
    const sy = (q: number) =>
      m.top + ph - ((Math.log10(q) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * ph;

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `Queries per second versus recall for ${this.rows.length} ef settings`,
    });

    // Grid and axes.
    const grid = el("g", { class: "grid" }, svg);
    for (let decade = 10 ** Math.floor(Math.log10(yMin)); decade <= yMax; decade *= 10) {
      for (const mult of [1, 2, 5]) {
        const q = decade * mult;
        if (q < yMin || q > yMax) continue;
        el("line", { x1: m.left, x2: m.left + pw, y1: sy(q), y2: sy(q) }, grid);
        el("text", { x: m.left - 8, y: sy(q) + 4, "text-anchor": "end", class: "tick" }, svg).textContent =
          q >= 1000 ? `${q / 1000}k` : String(q);
      }
    }
    for (const t of RECALL_TICKS) {
      if (t < xMin || t > xMax) continue;
      if (compact && ![0.5, 0.9, 0.99, 0.999].includes(t)) continue; // keep labels from colliding
      el("line", { x1: sx(t), x2: sx(t), y1: m.top, y2: m.top + ph }, grid);
      el("text", { x: sx(t), y: m.top + ph + 18, "text-anchor": "middle", class: "tick" }, svg).textContent = String(t);
    }
    el("line", { x1: m.left, x2: m.left + pw, y1: m.top + ph, y2: m.top + ph, class: "baseline-axis" }, svg);
    el("text", { x: m.left + pw / 2, y: height - 8, "text-anchor": "middle", class: "axis-title" }, svg).textContent =
      "Recall@10 →";
    el("text", {
      x: 14,
      y: m.top + ph / 2,
      "text-anchor": "middle",
      class: "axis-title",
      transform: `rotate(-90 14 ${m.top + ph / 2})`,
    }, svg).textContent = "Queries / second (log) →";

    // Exact search reference: what you would get without an index.
    if (this.baseline && this.baseline >= yMin) {
      const y = sy(this.baseline);
      el("line", { x1: m.left, x2: m.left + pw, y1: y, y2: y, class: "reference" }, svg);
      el("text", { x: m.left + pw - 4, y: y - 7, "text-anchor": "end", class: "reference-label" }, svg).textContent =
        `Exact search: ${formatQps(this.baseline)} QPS`;
    }

    // Series: area wash, line, points.
    const pts = this.rows.map((r) => [sx(r.recall), sy(r.qps)] as const);
    const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    if (pts.length > 1) {
      el("path", { d: `${line} L${pts.at(-1)![0]},${m.top + ph} L${pts[0][0]},${m.top + ph} Z`, class: "series-area" }, svg);
      el("path", { d: line, class: "series-line" }, svg);
    }
    const crosshair = el("line", { y1: m.top, y2: m.top + ph, class: "crosshair", visibility: "hidden" }, svg);
    const circles = pts.map(([x, y]) => el("circle", { cx: x, cy: y, r: 4.5, class: "point" }, svg));

    // Hover: snap to the nearest point along x; the hit area is the whole plot.
    const hit = el("rect", { x: m.left, y: m.top, width: pw, height: ph, fill: "transparent" }, svg);
    const show = (index: number) => {
      const row = this.rows[index];
      circles.forEach((c, i) => c.classList.toggle("active", i === index));
      crosshair.setAttribute("x1", String(pts[index][0]));
      crosshair.setAttribute("x2", String(pts[index][0]));
      crosshair.setAttribute("visibility", "visible");
      const speedup = this.baseline ? `${Math.round(row.qps / this.baseline).toLocaleString()}×` : "–";
      this.tooltip.innerHTML =
        `<b>ef = ${row.ef}</b><dl>` +
        `<dt>Recall@10</dt><dd>${row.recall.toFixed(4)}</dd>` +
        `<dt>Throughput</dt><dd>${Math.round(row.qps).toLocaleString()} QPS</dd>` +
        `<dt>p50 / p99</dt><dd>${formatUs(row.p50Us)} / ${formatUs(row.p99Us)}</dd>` +
        `<dt>Distances / query</dt><dd>${Math.round(row.distancesPerQuery).toLocaleString()}</dd>` +
        `<dt>vs. exact search</dt><dd>${speedup}</dd></dl>`;
      this.tooltip.hidden = false;
      const scale = this.container.clientWidth / width;
      this.tooltip.style.left = `${Math.min(Math.max(pts[index][0] * scale, 100), this.container.clientWidth - 100)}px`;
      this.tooltip.style.top = `${pts[index][1] * scale}px`;
    };
    const hide = () => {
      circles.forEach((c) => c.classList.remove("active"));
      crosshair.setAttribute("visibility", "hidden");
      this.tooltip.hidden = true;
    };
    hit.addEventListener("pointermove", (event) => {
      const box = svg.getBoundingClientRect();
      const x = ((event.clientX - box.left) / box.width) * width;
      let best = 0;
      pts.forEach(([px], i) => {
        if (Math.abs(px - x) < Math.abs(pts[best][0] - x)) best = i;
      });
      show(best);
    });
    hit.addEventListener("pointerleave", hide);

    this.container.replaceChildren(svg, this.tooltip);
  }
}
