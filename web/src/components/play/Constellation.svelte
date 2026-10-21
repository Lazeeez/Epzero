<script lang="ts" module>
  // The Play header's graphic: a small proximity graph and a greedy route hopping across it to a
  // target, drawn hop by hop on a loop. Built once from a fixed seed; decorative.
  const W = 420;
  const H = 320;
  function rng(seed: number) {
    return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  }
  const rand = rng(7);
  const points: { x: number; y: number }[] = [];
  while (points.length < 38) {
    const p = { x: 20 + rand() * (W - 40), y: 20 + rand() * (H - 40) };
    if (points.every((q) => Math.hypot(p.x - q.x, p.y - q.y) > 34)) points.push(p);
  }
  const dist = (a: number, b: number) => Math.hypot(points[a].x - points[b].x, points[a].y - points[b].y);
  const neighbours = points.map((_, i) =>
    points
      .map((_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => dist(i, a) - dist(i, b))
      .slice(0, 4),
  );
  const edges = new Set<string>();
  neighbours.forEach((ns, i) => ns.forEach((j) => edges.add(i < j ? `${i}-${j}` : `${j}-${i}`)));
  const edgeList = [...edges].map((e) => e.split("-").map(Number) as [number, number]);
  // Greedy walk from the leftmost point toward the rightmost, as a search would.
  const start = points.reduce((best, p, i) => (p.x < points[best].x ? i : best), 0);
  const target = points.reduce((best, p, i) => (p.x > points[best].x ? i : best), 0);
  const route = [start];
  for (let at = start; at !== target && route.length < 20; ) {
    const next = neighbours[at].reduce((best, j) => (dist(j, target) < dist(best, target) ? j : best), at);
    if (next === at) break;
    route.push((at = next));
  }
</script>

<svg class="constellation" viewBox="0 0 {W} {H}" aria-hidden="true">
  {#each edgeList as [a, b] (`${a}-${b}`)}
    <line class="edge" x1={points[a].x} y1={points[a].y} x2={points[b].x} y2={points[b].y} />
  {/each}
  {#each route.slice(1) as to, i (to)}
    {@const from = route[i]}
    <line class="hop" x1={points[from].x} y1={points[from].y} x2={points[to].x} y2={points[to].y} pathLength="1" style:--i={i} />
  {/each}
  {#each points as p, i (i)}
    <circle cx={p.x} cy={p.y} r={i === target ? 6 : 2.6} class:target={i === target} class:start={i === start} class:onroute={route.includes(i)} style:--i={route.indexOf(i)} />
  {/each}
</svg>

<style>
  .constellation {
    display: block;
    width: 100%;
    height: auto;
    overflow: visible;
  }
  .edge {
    stroke: var(--border-strong);
    stroke-width: 1;
  }
  circle {
    fill: var(--muted);
  }
  circle.onroute {
    animation: light 7s calc(var(--i) * 0.42s) infinite;
  }
  circle.start {
    fill: var(--accent-strong);
  }
  circle.target {
    fill: var(--result);
    animation: beacon 2.2s ease-in-out infinite;
    transform-box: fill-box;
    transform-origin: center;
  }
  .hop {
    stroke: var(--accent);
    stroke-width: 2.4;
    stroke-linecap: round;
    stroke-dasharray: 1;
    stroke-dashoffset: 1;
    animation: hop 7s calc(var(--i) * 0.42s + 0.42s) infinite;
  }
  @keyframes hop {
    0% {
      stroke-dashoffset: 1;
      opacity: 1;
    }
    6% {
      stroke-dashoffset: 0;
    }
    80% {
      stroke-dashoffset: 0;
      opacity: 1;
    }
    92%,
    100% {
      stroke-dashoffset: 0;
      opacity: 0;
    }
  }
  @keyframes light {
    0%,
    92%,
    100% {
      fill: var(--muted);
    }
    6%,
    80% {
      fill: var(--accent-strong);
    }
  }
  @keyframes beacon {
    50% {
      transform: scale(1.35);
      opacity: 0.75;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .hop,
    circle.onroute,
    circle.target {
      animation: none;
    }
    .hop {
      stroke-dashoffset: 0;
    }
    circle.onroute {
      fill: var(--accent-strong);
    }
  }
</style>
