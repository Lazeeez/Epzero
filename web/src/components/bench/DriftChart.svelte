<script lang="ts" module>
  export interface DriftBatch {
    policy: string;
    phase: string;
    pass: number;
    ef: number;
    recall: number;
    distances: number;
    estimate?: number;
    lower?: number;
    upper?: number;
  }
  export interface DriftSeries {
    policy: string;
    name: string;
    color: string;
  }
</script>

<script lang="ts">
  import { scaleLinear } from "d3-scale";
  import { line, area } from "d3-shape";

  // Recall per batch of queries as the workload drifts, one line per policy, with the target and
  // the guard's own confidence interval. One axis: recall.
  let {
    batches,
    series,
    target,
    label,
  }: { batches: DriftBatch[]; series: DriftSeries[]; target: number; label: string } = $props();

  let width = $state(0);
  let hidden = $state<Record<string, boolean>>({});
  let hover = $state<number | null>(null);
  let svg: SVGSVGElement | undefined = $state();

  const compact = $derived(width < 560);
  const m = $derived({ top: 34, right: compact ? 14 : 96, bottom: 34, left: compact ? 44 : 52 });
  const h = $derived(compact ? 300 : 360);
  const byPolicy = $derived(
    Object.fromEntries(series.map((s) => [s.policy, batches.filter((b) => b.policy === s.policy)])) as Record<string, DriftBatch[]>,
  );
  const steps = $derived(byPolicy[series[0].policy]?.length ?? 0);
  // Phase spans along the stream, from the first policy's batches.
  const phases = $derived.by(() => {
    const out: { name: string; start: number; end: number }[] = [];
    (byPolicy[series[0].policy] ?? []).forEach((b, i) => {
      const last = out.at(-1);
      if (last && last.name === b.phase) last.end = i;
      else out.push({ name: b.phase, start: i, end: i });
    });
    return out;
  });
  const visible = $derived(series.filter((s) => !hidden[s.policy]));
  const lowest = $derived(Math.min(target, ...visible.flatMap((s) => byPolicy[s.policy].map((b) => b.recall))));

  const x = $derived(scaleLinear().domain([-0.5, steps - 0.5]).range([m.left, Math.max(m.left + 10, width - m.right)]));
  const y = $derived(
    scaleLinear()
      .domain([Math.floor((lowest - 0.01) * 100) / 100, 1])
      .range([h - m.bottom, m.top]),
  );
  const yTicks = $derived(y.ticks(compact ? 4 : 6));
  const path = $derived(
    line<DriftBatch>()
      .x((_, i) => x(i))
      .y((b) => y(b.recall)),
  );
  const band = $derived(
    area<DriftBatch>()
      .defined((b) => b.lower !== undefined && b.lower > 0)
      .x((_, i) => x(i))
      .y0((b) => y(b.lower!))
      .y1((b) => y(b.upper!)),
  );
  const guard = $derived(series.find((s) => s.policy === "guard"));
  // The guard's interval can reach well below the axis early in a window: clip it to the plot.
  const uid = $props.id();
  const clip = `drift-${uid}`;
  // End-of-line labels, nudged apart where lines finish close together.
  const endLabels = $derived.by(() => {
    const labels = visible
      .map((s) => ({ name: s.name, y: y(byPolicy[s.policy][byPolicy[s.policy].length - 1].recall) }))
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 15);
    return labels;
  });

  function onMove(event: PointerEvent) {
    const rect = svg!.getBoundingClientRect();
    const i = Math.round(x.invert(event.clientX - rect.left));
    hover = i >= 0 && i < steps ? i : null;
  }
  const phaseName = (phase: string) =>
    ({ steady: "Steady", harder: "Harder queries", churn: "Churn", filtered: "Filtered (5%)" })[phase] ?? phase;
</script>

<div class="chart" bind:clientWidth={width}>
  <div class="legend" role="group" aria-label="Policies">
    {#each series as s (s.policy)}
      <button
        class:off={hidden[s.policy]}
        style:--c={s.color}
        onclick={() => (hidden = { ...hidden, [s.policy]: !hidden[s.policy] })}
        aria-pressed={!hidden[s.policy]}
      >
        <i></i>{s.name}
      </button>
    {/each}
    <span class="key target-key"><i></i>Target {target}</span>
    {#if guard && !hidden.guard}<span class="key band-key" style:--c={guard.color}><i></i>Guard's 95% interval</span>{/if}
  </div>

  <div class="plot">
    {#if width > 0}
      <svg bind:this={svg} {width} height={h} role="img" aria-label={label} onpointermove={onMove} onpointerleave={() => (hover = null)}>
        {#each phases as p, i (p.name)}
          <rect
            class="phase"
            class:alt={i % 2 === 1}
            x={x(p.start - 0.5)}
            y={m.top}
            width={x(p.end + 0.5) - x(p.start - 0.5)}
            height={h - m.top - m.bottom}
          />
          <text class="phase-name" x={(x(p.start - 0.5) + x(p.end + 0.5)) / 2} y={m.top - 12} text-anchor="middle">
            {compact ? phaseName(p.name).split(" ")[0] : phaseName(p.name)}
          </text>
        {/each}
        <g class="grid">
          {#each yTicks as t (t)}
            <line x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} />
            <text x={m.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{t.toFixed(2)}</text>
          {/each}
        </g>
        <text class="axis-label" x={(m.left + width - m.right) / 2} y={h - 8} text-anchor="middle">
          Batches of queries, in order →
        </text>

        <defs>
          <clipPath id={clip}>
            <rect x={m.left} y={m.top} width={Math.max(0, width - m.left - m.right)} height={h - m.top - m.bottom} />
          </clipPath>
        </defs>
        {#if guard && !hidden.guard}
          <path class="band" d={band(byPolicy.guard) ?? ""} style:fill={guard.color} clip-path="url(#{clip})" />
        {/if}
        <line class="target" x1={m.left} x2={width - m.right} y1={y(target)} y2={y(target)} />

        {#each visible as s (s.policy)}
          {@const points = byPolicy[s.policy]}
          <path class="series" d={path(points)} style:stroke={s.color} />
          {#each points as b, i (i)}
            <circle cx={x(i)} cy={y(b.recall)} r={hover === i ? 5.5 : 4} style:fill={s.color} />
          {/each}
        {/each}
        {#if !compact}
          {#each endLabels as l (l.name)}
            <text class="end-label" x={x(steps - 1) + 10} y={l.y} dy="0.32em">{l.name}</text>
          {/each}
        {/if}

        {#if hover !== null}
          <line class="cross" x1={x(hover)} x2={x(hover)} y1={m.top} y2={h - m.bottom} />
        {/if}
      </svg>

      {#if hover !== null}
        {@const first = byPolicy[series[0].policy][hover]}
        <div class="tooltip card" style:left="{Math.min(Math.max(x(hover), 130), width - 130)}px" style:top="{m.top}px">
          <strong class="num">{phaseName(first.phase)} · pass {first.pass + 1}</strong>
          {#each visible as s (s.policy)}
            {@const b = byPolicy[s.policy][hover]}
            <span class="row num" style:--c={s.color}>
              <i></i>{s.name}<b class:short={b.recall < target}>{b.recall.toFixed(3)}{b.recall < target ? " ↓" : ""}</b><em>ef {b.ef}</em>
            </span>
          {/each}
          {#if !hidden.guard && byPolicy.guard?.[hover]?.estimate !== undefined && (byPolicy.guard[hover].lower ?? 0) > 0}
            {@const g = byPolicy.guard[hover]}
            <span class="estimate num">Guard measured {g.estimate!.toFixed(3)} ({g.lower!.toFixed(3)}–{g.upper!.toFixed(3)})</span>
          {/if}
        </div>
      {/if}
    {/if}
  </div>

  <details>
    <summary>Show the data (↓ marks recall below the target)</summary>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Phase</th><th>Pass</th>{#each series as s (s.policy)}<th>{s.name}: recall</th><th>ef</th><th>distances</th>{/each}</tr>
        </thead>
        <tbody>
          {#each byPolicy[series[0].policy] as row, i (i)}
            <tr>
              <td>{phaseName(row.phase)}</td><td>{row.pass + 1}</td>
              {#each series as s (s.policy)}
                {@const b = byPolicy[s.policy][i]}
                <td class:short={b.recall < target}>{b.recall.toFixed(4)}{b.recall < target ? " ↓" : ""}</td><td>{b.ef}</td><td>{Math.round(b.distances).toLocaleString()}</td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </details>
</div>

<style>
  .chart {
    position: relative;
    contain: inline-size;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
  }
  .legend button,
  .key {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 12px;
    border: 1px solid var(--border);
    border-radius: 99px;
    background: var(--surface);
    color: var(--text);
    font-size: 13px;
  }
  .legend button {
    cursor: pointer;
  }
  .legend button.off {
    color: var(--muted);
    text-decoration: line-through;
  }
  .legend button.off i {
    opacity: 0.35;
  }
  .legend i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--c);
  }
  .key {
    color: var(--text-2);
  }
  .target-key i {
    width: 16px;
    height: 0;
    border-radius: 0;
    border-top: 2px dashed var(--text-2);
    background: none;
  }
  .band-key i {
    width: 14px;
    height: 10px;
    border-radius: 3px;
    background: var(--c);
    opacity: 0.2;
  }
  .plot {
    position: relative;
  }
  svg {
    display: block;
    overflow: visible;
    touch-action: pan-y;
  }
  .phase {
    fill: transparent;
  }
  .phase.alt {
    fill: var(--surface);
  }
  .phase-name {
    fill: var(--text-2);
    font: 600 11.5px var(--mono);
    letter-spacing: 0.02em;
  }
  .grid line {
    stroke: var(--grid);
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
  .target {
    stroke: var(--text-2);
    stroke-width: 1.5;
    stroke-dasharray: 5 5;
  }
  .band {
    opacity: 0.16;
  }
  .series {
    fill: none;
    stroke-width: 2;
    stroke-linejoin: round;
  }
  circle {
    stroke: var(--surface-solid);
    stroke-width: 2;
    transition: r 0.15s;
  }
  .end-label {
    fill: var(--text-2);
    font-size: 12px;
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
    min-width: 240px;
    padding: 10px 12px;
    background: var(--surface-solid);
    font-size: 12.5px;
    pointer-events: none;
    transform: translate(-50%, 0);
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
    color: var(--text);
  }
  .row em {
    color: var(--muted);
    font: normal 11.5px var(--mono);
  }
  .short {
    color: var(--warn) !important;
  }
  .estimate {
    margin-top: 2px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
    color: var(--text-2);
    font-size: 12px;
  }
  details {
    margin-top: 14px;
  }
  summary {
    cursor: pointer;
    color: var(--text-2);
    font-size: 13px;
  }
  .table-wrap {
    overflow-x: auto;
    margin-top: 10px;
  }
  table {
    border-collapse: collapse;
    font: 12px var(--mono);
    white-space: nowrap;
  }
  th,
  td {
    padding: 5px 10px;
    border-bottom: 1px solid var(--border);
    text-align: right;
  }
  th:first-child,
  td:first-child {
    text-align: left;
  }
  th {
    color: var(--muted);
    font-weight: 600;
  }
</style>
