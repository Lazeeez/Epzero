<script lang="ts" module>
  export interface SweepPoint {
    ef: number;
    recall: number;
    qps: number;
  }
  export interface Series {
    name: string;
    /** A CSS color, usually a design token like var(--accent). */
    color: string;
    points: SweepPoint[];
  }
</script>

<script lang="ts">
  import { scaleLog } from "d3-scale";
  import { curveMonotoneX, line } from "d3-shape";
  import { formatQps, formatTick } from "../../lib/format";

  let {
    series,
    baseline = null,
    baselineLabel = "Exact scan",
    height = 420,
    label,
  }: { series: Series[]; baseline?: number | null; baselineLabel?: string; height?: number; label: string } = $props();

  let width = $state(0);
  let hidden = $state<Record<string, boolean>>({});
  let hoverEf = $state<number | null>(null);
  let drawn = $state(false);
  let svg: SVGSVGElement | undefined = $state();

  const compact = $derived(width < 560);
  const m = $derived({ top: 16, right: compact ? 12 : 24, bottom: 46, left: compact ? 48 : 62 });
  const h = $derived(compact ? Math.min(height, 330) : height);
  const visible = $derived(series.filter((s) => !hidden[s.name] && s.points.length));
  const all = $derived(visible.flatMap((s) => s.points));

  // Recall on a log scale of the miss rate (1 - recall), as ANN benchmarks plot it: the high-recall
  // end, where production systems operate, gets most of the width instead of a sliver.
  const miss = $derived.by(() => {
    const worst = Math.max(...all.map((p) => 1 - p.recall), 0.05);
    const best = Math.min(...all.map((p) => 1 - p.recall), 0.01);
    return scaleLog()
      .domain([worst * 1.15, best * 0.8])
      .range([m.left, Math.max(m.left + 10, width - m.right)]);
  });
  const x = (recall: number) => miss(Math.max(1 - recall, 1e-6));
  const y = $derived.by(() => {
    const values = all.map((p) => p.qps).concat(baseline ? [baseline] : []);
    const lo = Math.min(...values, Infinity);
    const hi = Math.max(...values, -Infinity);
    return scaleLog()
      .domain(Number.isFinite(lo) ? [lo * 0.8, hi * 1.2] : [1, 10])
      .range([h - m.bottom, m.top]);
  });
  const RECALL_TICKS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995];
  const xTicks = $derived.by(() => {
    const [lo, hi] = miss.domain();
    const ticks = RECALL_TICKS.filter((r) => 1 - r <= lo && 1 - r >= hi);
    return compact ? ticks.filter((_, i) => i % 2 === 0) : ticks;
  });
  const yTicks = $derived(
    y.ticks().filter((t) => {
      const lead = t / Math.pow(10, Math.floor(Math.log10(t) + 1e-9));
      return compact ? lead === 1 || lead === 3 : lead === 1 || lead === 2 || lead === 5;
    }),
  );
  const path = $derived(
    line<SweepPoint>()
      .x((p) => x(p.recall))
      .y((p) => y(p.qps))
      .curve(curveMonotoneX),
  );

  // Draw the lines in once the chart scrolls into view.
  $effect(() => {
    if (!svg) return;
    const observer = new IntersectionObserver(([e]) => e.isIntersecting && ((drawn = true), observer.disconnect()), { threshold: 0.3 });
    observer.observe(svg);
    return () => observer.disconnect();
  });

  function onMove(event: PointerEvent) {
    const rect = svg!.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    let best: SweepPoint | null = null;
    let bestD = Infinity;
    for (const p of all) {
      const d = (x(p.recall) - px) ** 2 + (y(p.qps) - py) ** 2;
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    hoverEf = best && bestD < 70 ** 2 ? best.ef : null;
  }

  const tooltip = $derived.by(() => {
    if (hoverEf === null) return null;
    const rows = visible
      .map((s) => ({ s, p: s.points.find((p) => p.ef === hoverEf) }))
      .filter((r): r is { s: Series; p: SweepPoint } => !!r.p);
    if (!rows.length) return null;
    const anchor = rows[0].p;
    return { rows, left: x(anchor.recall), top: y(anchor.qps) };
  });
  const sorted = (points: SweepPoint[]) => [...points].sort((a, b) => a.recall - b.recall);
</script>

<div class="chart" bind:clientWidth={width}>
  <div class="legend" role="group" aria-label="Series">
    {#each series as s (s.name)}
      <button
        class:off={hidden[s.name]}
        style:--c={s.color}
        onclick={() => (hidden = { ...hidden, [s.name]: !hidden[s.name] })}
        aria-pressed={!hidden[s.name]}
      >
        <i></i>{s.name}
      </button>
    {/each}
    {#if baseline}
      <span class="base"><i></i>{baselineLabel}</span>
    {/if}
  </div>

  <div class="plot">
    {#if width > 0}
      <svg
        bind:this={svg}
        {width}
        height={h}
        role="img"
        aria-label={label}
        onpointermove={onMove}
        onpointerleave={() => (hoverEf = null)}
      >
        <g class="grid">
          {#each yTicks as t (t)}
            <line x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} />
            <text x={m.left - 10} y={y(t)} dy="0.32em" text-anchor="end">{formatTick(t)}</text>
          {/each}
          {#each xTicks as t (t)}
            <line class="v" x1={x(t)} x2={x(t)} y1={m.top} y2={h - m.bottom} />
            <text x={x(t)} y={h - m.bottom + 20} text-anchor="middle">{t}</text>
          {/each}
        </g>
        <text class="axis-label" x={(m.left + width - m.right) / 2} y={h - 6} text-anchor="middle">Recall@10 (log scale of misses) →</text>
        <text class="axis-label" transform="translate(12 {(m.top + h - m.bottom) / 2}) rotate(-90)" text-anchor="middle">
          Queries per second (log) →
        </text>

        {#if baseline}
          <line class="baseline" x1={m.left} x2={width - m.right} y1={y(baseline)} y2={y(baseline)} />
        {/if}

        {#each visible as s (s.name)}
          <path
            d={path(sorted(s.points))}
            class="series"
            style:stroke={s.color}
            pathLength="1"
            stroke-dasharray="1"
            stroke-dashoffset={drawn ? 0 : 1}
          />
          {#each s.points as p (p.ef)}
            <circle
              cx={x(p.recall)}
              cy={y(p.qps)}
              r={hoverEf === p.ef ? 5.5 : 3.2}
              style:fill={s.color}
              class:shown={drawn}
            />
          {/each}
        {/each}

        {#if tooltip}
          <line class="cross" x1={tooltip.left} x2={tooltip.left} y1={m.top} y2={h - m.bottom} />
        {/if}
      </svg>

      {#if tooltip}
        <div
          class="tooltip card"
          style:left="{Math.min(Math.max(tooltip.left, 110), width - 110)}px"
          style:top="{Math.max(tooltip.top - 16, 8)}px"
        >
          <strong class="num">ef = {hoverEf}</strong>
          {#each tooltip.rows as { s, p } (s.name)}
            <span class="row num" style:--c={s.color}>
              <i></i>{s.name}<b>{formatQps(p.qps)}</b><em>{p.recall.toFixed(4)}</em>
            </span>
          {/each}
          {#if tooltip.rows.length > 1}
            <span class="ratio num">
              {tooltip.rows[0].s.name}: {(tooltip.rows[0].p.qps / tooltip.rows[tooltip.rows.length - 1].p.qps).toFixed(2)}× {tooltip.rows[
                tooltip.rows.length - 1
              ].s.name}
            </span>
          {/if}
        </div>
      {/if}
    {/if}
  </div>
</div>

<style>
  .chart {
    position: relative;
    /* The measured width drives the SVG's width; without containment the SVG would hold its grid
       cell open, so the chart could grow on resize but never shrink (e.g. rotating a tablet). */
    contain: inline-size;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
  }
  .legend button,
  .base {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 12px;
    border: 1px solid var(--border);
    border-radius: 99px;
    background: var(--surface);
    color: var(--text);
    font-size: 13px;
    cursor: pointer;
    transition: opacity 0.2s;
  }
  .legend button.off {
    opacity: 0.4;
  }
  .legend i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--c);
  }
  .base {
    cursor: default;
    color: var(--text-2);
  }
  .base i {
    width: 16px;
    height: 0;
    border-radius: 0;
    border-top: 2px dashed var(--muted);
    background: none;
  }
  .plot {
    position: relative;
  }
  svg {
    display: block;
    overflow: visible;
    touch-action: pan-y;
  }
  .grid line {
    stroke: var(--grid);
  }
  .grid line.v {
    stroke-dasharray: 2 4;
  }
  .grid text,
  .axis-label {
    fill: var(--muted);
    font: 11.5px var(--mono);
  }
  .axis-label {
    fill: var(--text-2);
    font-size: 12px;
  }
  .series {
    fill: none;
    stroke-width: 2.4;
    stroke-linecap: round;
    transition: stroke-dashoffset 1.6s var(--ease);
  }
  circle {
    stroke: var(--page);
    stroke-width: 1.5;
    opacity: 0;
    transition: opacity 0.6s 1s, r 0.15s;
  }
  circle.shown {
    opacity: 1;
  }
  .baseline {
    stroke: var(--muted);
    stroke-width: 1.5;
    stroke-dasharray: 5 5;
  }
  .cross {
    stroke: var(--border-strong);
    pointer-events: none;
  }
  .tooltip {
    position: absolute;
    z-index: 5;
    display: grid;
    gap: 5px;
    min-width: 210px;
    padding: 10px 12px;
    background: var(--surface-solid);
    font-size: 12.5px;
    pointer-events: none;
    transform: translate(-50%, -100%);
  }
  .tooltip strong {
    font: 600 12px var(--mono);
    color: var(--text-2);
  }
  .row {
    display: grid;
    grid-template-columns: 10px 1fr auto auto;
    align-items: center;
    gap: 8px;
  }
  .row i {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--c);
  }
  .row b {
    font: 600 12.5px var(--mono);
  }
  .row em {
    color: var(--muted);
    font: normal 11.5px var(--mono);
  }
  .ratio {
    margin-top: 2px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
    color: var(--accent-strong);
    font: 600 12px var(--mono);
  }
</style>
