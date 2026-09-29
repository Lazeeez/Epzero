<script lang="ts">
  import StarChart from "../components/play/StarChart.svelte";
  import Thumb from "../components/Thumb.svelte";
  import { engine } from "../lib/engine/engine.svelte";
  import type { Trace } from "../lib/engine/protocol";
  import { router } from "../lib/router.svelte";

  const MAX_HOPS = 25;
  const BEST_KEY = "oryx-navigator-best";

  type Phase = "intro" | "loading" | "playing" | "won" | "lost";
  let phase = $state<Phase>("intro");
  let target = $state(0);
  let goals = $state.raw(new Set<number>());
  let path = $state<number[]>([]);
  let similarity = $state(0);
  let startSimilarity = $state(0);
  let finalSimilarity = $state(1);
  let neighbors = $state.raw<{ id: number; similarity: number }[]>([]);
  let lastDelta = $state<number | null>(null);
  let hints = $state(0);
  let hinted = $state<number | null>(null);
  let highlight = $state<number | null>(null);
  let oryx = $state.raw<Trace | null>(null);
  let seen = $state(0);
  let error = $state("");
  let best = $state<number | null>(readBest());
  let moving = false; // one hop at a time, even on a fast double-click

  const hops = $derived(path.length - 1);
  const current = $derived(path[path.length - 1]);
  // Oryx's route, counted like yours: greedy moves on the express (upper) layers, then the photos
  // its beam search expanded on the base layer. (Its "accepted" events also include candidates
  // merely admitted to the beam, which aren't moves.)
  const oryxRoute = $derived.by(() => {
    if (!oryx) return { express: 0, expanded: 0 };
    let express = 0;
    const expanded = new Set<number>();
    for (let i = 0; i < oryx.to.length; i++) {
      if (oryx.layer[i] > 0) express += oryx.accepted[i];
      else expanded.add(oryx.from[i]);
    }
    return { express, expanded: expanded.size };
  });
  // Progress from where you started to the target's nearest neighbour.
  const progress = $derived(Math.max(0, Math.min(1, (similarity - startSimilarity) / Math.max(finalSimilarity - startSimilarity, 1e-6))));

  $effect(() => {
    engine.start();
  });

  function readBest(): number | null {
    try {
      const v = localStorage.getItem(BEST_KEY);
      return v ? Number(v) : null;
    } catch {
      return null;
    }
  }

  async function newRound() {
    error = "";
    phase = "loading";
    try {
      const t = engine.randomId();
      // The target's true neighbourhood (exact scan) is the goal; Oryx's own search is the benchmark.
      const outcome = await engine.search({ kind: "id", id: t }, 64, 10);
      // Start far away: the least similar of a few random photos.
      let start = engine.randomId();
      let startSim = await engine.similarity(start, t);
      for (let i = 0; i < 12; i++) {
        const candidate = engine.randomId();
        const sim = await engine.similarity(candidate, t);
        if (sim < startSim && !outcome.trace.exact.includes(candidate)) {
          start = candidate;
          startSim = sim;
        }
      }
      target = t;
      goals = new Set(outcome.trace.exact);
      oryx = null;
      pendingOryx = outcome.trace;
      path = [start];
      similarity = startSim;
      startSimilarity = startSim;
      finalSimilarity = await engine.similarity(outcome.trace.exact[1] ?? t, t);
      lastDelta = null;
      hints = 0;
      hinted = null;
      seen = 1;
      await loadNeighbors(start);
      phase = "playing";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      phase = "intro";
    }
  }
  let pendingOryx: Trace | null = null;

  async function loadNeighbors(id: number) {
    const hood = await engine.neighbors(id, target);
    neighbors = Array.from(hood.ids, (n, i) => ({ id: n, similarity: hood.similarities[i] }));
    seen += neighbors.length;
    hinted = null;
  }

  async function hop(id: number, sim: number) {
    if (phase !== "playing" || moving) return;
    moving = true;
    try {
      await move(id, sim);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      moving = false;
    }
  }

  async function move(id: number, sim: number) {
    lastDelta = sim - similarity;
    similarity = sim;
    path = [...path, id];
    highlight = null;
    if (goals.has(id)) return finish(true);
    if (hops >= MAX_HOPS) return finish(false);
    await loadNeighbors(id);
  }

  async function back() {
    if (path.length < 2 || phase !== "playing" || moving) return;
    // Going back is allowed, but it still costs a hop.
    const previous = path[path.length - 2];
    await hop(previous, await engine.similarity(previous, target));
  }

  function hint() {
    if (!neighbors.length) return;
    hints++;
    hinted = neighbors.reduce((a, b) => (b.similarity > a.similarity ? b : a)).id;
  }

  function finish(won: boolean) {
    phase = won ? "won" : "lost";
    oryx = pendingOryx;
    if (won && (best === null || hops < best)) {
      best = hops;
      try {
        localStorage.setItem(BEST_KEY, String(hops));
      } catch {
        /* storage blocked: the best score just isn't remembered */
      }
    }
  }

  const fmt = (x: number) => `${(x * 100).toFixed(0)}%`;
  const time = (us: number) => (us >= 1000 ? `${(us / 1000).toFixed(1)} ms` : `${Math.round(us)} µs`);
</script>

<div class="play container">
  <header class="head">
    <p class="eyebrow">Navigator challenge</p>
    <h1><span class="gradient-text">Can you out-navigate</span> Oryx?</h1>
    <p class="lede">
      You're dropped on a random photo among {engine.count ? engine.count.toLocaleString() : "29,825"}. Reach the target by hopping
      between neighbouring photos: the same links Oryx's search walks. Oryx gets express lanes; you get a star chart.
    </p>
  </header>

  {#if phase === "intro" || phase === "loading"}
    <div class="intro card">
      <ol>
        <li><b>Pick a neighbour</b> that looks more like the target. Each photo links to up to 32 others.</li>
        <li><b>Watch the chart:</b> nearby stars are similar photos. Hover a neighbour to see where it leads.</li>
        <li><b>Reach any of the target's 10 nearest photos</b> within {MAX_HOPS} hops.</li>
      </ol>
      <div class="start">
        {#if engine.status === "ready"}
          <button class="btn btn-primary" onclick={newRound} disabled={phase === "loading"}>
            {phase === "loading" ? "Charting a course…" : "Start navigating"}
          </button>
          {#if best !== null}<span class="best num">Your best: {best} hops</span>{/if}
        {:else if engine.status === "error"}
          <p class="warn">{engine.error}</p>
          <button class="btn btn-primary" onclick={() => engine.retry()}>Try again</button>
        {:else}
          <span class="loading num">{engine.stage || "Starting the engine"} · {Math.round(engine.progress * 100)}%</span>
          <span class="bar"><i style:width="{engine.progress * 100}%"></i></span>
        {/if}
      </div>
      {#if error}<p class="warn" role="alert">{error}</p>{/if}
    </div>
  {:else}
    <div class="game">
      <section class="side">
        <div class="target card">
          <span class="tag">Target</span>
          <Thumb id={target} size="n" alt={engine.caption(target)} />
          <p>{engine.caption(target)}</p>
        </div>
        <div class="chart card">
          <StarChart {target} {goals} {path} neighbors={neighbors.map((n) => n.id)} {highlight} {oryx} />
          <ul class="legend">
            <li><i class="dot you"></i>Your route</li>
            <li><i class="dot goal"></i>Target and its neighbours</li>
            {#if oryx}<li><i class="dot oryx"></i>Oryx's route</li>{/if}
          </ul>
        </div>
      </section>

      <section class="main">
        <div class="status card" aria-live="polite">
          <div class="here">
            <Thumb id={current} size="q" alt={engine.caption(current)} />
            <div>
              <span class="tag">You are here · hop {hops}/{MAX_HOPS}</span>
              <strong>{engine.caption(current)}</strong>
              {#if lastDelta !== null}
                <span class="delta num" class:up={lastDelta > 0} class:down={lastDelta < 0}>
                  {lastDelta > 0 ? "Warmer" : lastDelta < 0 ? "Colder" : "Same"} ({lastDelta > 0 ? "+" : ""}{(lastDelta * 100).toFixed(1)})
                </span>
              {/if}
            </div>
          </div>
          <div class="progress">
            <span class="num">Closing in {fmt(progress)}</span>
            <span class="bar"><i style:width="{progress * 100}%"></i></span>
          </div>
        </div>

        {#if phase === "playing"}
          <div class="toolbar">
            <span class="hint-text">Choose the next photo</span>
            <button class="btn btn-ghost" onclick={back} disabled={path.length < 2}>Back</button>
            <button class="btn btn-ghost" onclick={hint}>Ask Oryx{hints ? ` (${hints})` : ""}</button>
            <button class="btn btn-ghost" onclick={() => finish(false)}>Give up</button>
          </div>
          {#if !neighbors.length}<p class="warn">This photo has no links left to follow. Go back or give up.</p>{/if}
          {#if error}<p class="warn" role="alert">{error}</p>{/if}
          <ul class="options">
            {#each neighbors as n (n.id)}
              <li>
                <button
                  class:hinted={hinted === n.id}
                  class:visited={path.includes(n.id)}
                  onclick={() => hop(n.id, n.similarity)}
                  onpointerenter={() => (highlight = n.id)}
                  onpointerleave={() => highlight === n.id && (highlight = null)}
                  onfocus={() => (highlight = n.id)}
                  aria-label="Hop to {engine.caption(n.id)}"
                >
                  <Thumb id={n.id} size="q" alt="" />
                </button>
              </li>
            {/each}
          </ul>
        {:else}
          <div class="result card" class:won={phase === "won"}>
            <h2>{phase === "won" ? "You found it." : "Lost in space."}</h2>
            <dl class="num">
              <div><dt>You</dt><dd>{hops} hops</dd><span>{seen.toLocaleString()} photos looked at{hints ? ` · ${hints} hints` : ""}</span></div>
              <div class="vs">vs</div>
              <div>
                <dt>Oryx</dt>
                <dd>{oryxRoute.express} + {oryxRoute.expanded}</dd>
                <span>
                  express hops + base-layer expansions · {oryx?.distanceComputations.toLocaleString()} photos compared in
                  {oryx ? time(oryx.micros) : "–"}
                </span>
              </div>
            </dl>
            <p>
              Oryx starts on its sparse top layer, where one hop crosses the whole map, and only drops to the dense layer you were
              walking once it's close. There it widens into a beam search to make sure nothing nearer was missed. The blue lines on
              the chart are its route.
            </p>
            <div class="again">
              <button class="btn btn-primary" onclick={newRound}>Play again</button>
              <a class="btn btn-ghost" href={router.href("home", "#how")}>How the layers work</a>
              {#if best !== null}<span class="best num">Best: {best} hops</span>{/if}
            </div>
          </div>
        {/if}
      </section>
    </div>
  {/if}
</div>

<style>
  .play {
    padding: calc(var(--nav-h) + 48px) 0 64px;
  }
  .head h1 {
    font-size: clamp(34px, 5vw, 60px);
    letter-spacing: -0.045em;
  }
  .lede {
    max-width: 700px;
    margin: 16px 0 0;
    color: var(--text-2);
    font-size: 16.5px;
  }
  .intro {
    display: grid;
    gap: 20px;
    max-width: 720px;
    margin-top: 36px;
    padding: 26px;
  }
  .intro ol {
    display: grid;
    gap: 10px;
    margin: 0;
    padding-left: 20px;
    color: var(--text-2);
  }
  .intro b {
    color: var(--text);
  }
  .start {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px;
  }
  .best,
  .loading {
    color: var(--muted);
    font: 13px var(--mono);
  }
  .warn {
    margin: 0;
    color: var(--warn);
  }
  .bar {
    flex: 1;
    min-width: 120px;
    height: 6px;
    border-radius: 99px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .bar i {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--good));
    transition: width 0.4s var(--ease);
  }
  .game {
    display: grid;
    grid-template-columns: minmax(0, 380px) minmax(0, 1fr);
    gap: 16px;
    margin-top: 32px;
  }
  .side,
  .main {
    display: grid;
    align-content: start;
    gap: 16px;
    min-width: 0;
  }
  .card {
    background: color-mix(in srgb, var(--surface-solid) 80%, transparent);
  }
  .tag {
    color: var(--accent-strong);
    font: 600 11px var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .target {
    display: grid;
    gap: 10px;
    padding: 14px;
  }
  .target :global(.thumb) {
    aspect-ratio: 4 / 3;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 0 2px var(--good);
  }
  .target :global(img),
  .here :global(img),
  .options :global(img) {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .target p {
    margin: 0;
    font-weight: 550;
  }
  .chart {
    padding: 8px;
  }
  .chart :global(canvas) {
    border-radius: 10px;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    margin: 6px 6px 4px;
    padding: 0;
    list-style: none;
    color: var(--muted);
    font-size: 12px;
  }
  .legend li {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
  }
  .dot.you {
    background: var(--result);
  }
  .dot.goal {
    background: var(--good);
  }
  .dot.oryx {
    background: var(--accent);
  }
  .status {
    display: grid;
    gap: 12px;
    padding: 14px;
  }
  .here {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .here :global(.thumb) {
    flex: none;
    width: 74px;
    height: 74px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 0 2px var(--result);
  }
  .here div {
    display: grid;
    gap: 3px;
    min-width: 0;
  }
  .here strong {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .delta {
    font: 12.5px var(--mono);
    color: var(--muted);
  }
  .delta.up {
    color: var(--good);
  }
  .delta.down {
    color: var(--warn);
  }
  .progress {
    display: grid;
    gap: 6px;
    color: var(--text-2);
    font: 12.5px var(--mono);
  }
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }
  .toolbar .btn {
    height: 36px;
    padding: 0 14px;
    font-size: 13.5px;
  }
  .hint-text {
    margin-right: auto;
    color: var(--text-2);
    font-size: 14px;
  }
  .options {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(92px, 1fr));
    gap: 8px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .options button {
    display: block;
    width: 100%;
    aspect-ratio: 1;
    padding: 0;
    border: 0;
    border-radius: 12px;
    overflow: hidden;
    background: var(--surface-2);
    cursor: pointer;
    transition: transform 0.2s var(--ease), box-shadow 0.2s;
  }
  .options button :global(.thumb) {
    width: 100%;
    height: 100%;
  }
  .options button:hover,
  .options button:focus-visible {
    transform: translateY(-2px);
    box-shadow: 0 0 0 2px var(--text);
  }
  .options button.visited {
    opacity: 0.45;
  }
  .options button.hinted {
    box-shadow: 0 0 0 3px var(--accent), 0 0 24px var(--accent-glow);
  }
  .result {
    display: grid;
    gap: 16px;
    padding: 24px;
  }
  .result h2 {
    font-size: 26px;
    color: var(--warn);
  }
  .result.won h2 {
    color: var(--good);
  }
  .result dl {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
    align-items: start; /* both sides' labels and numbers line up, whatever the caption length */
    gap: 12px;
    margin: 0;
  }
  .result dl > div:not(.vs) {
    display: grid;
    gap: 2px;
  }
  .vs {
    align-self: center;
    color: var(--muted);
  }
  dt {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  dd {
    margin: 0;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.03em;
  }
  .result dl span {
    color: var(--muted);
    font-size: 12px;
  }
  .result p {
    margin: 0;
    color: var(--text-2);
  }
  .again {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
  }
  @media (max-width: 900px) {
    .game {
      grid-template-columns: 1fr;
    }
    .side {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }
  }
  @media (max-width: 560px) {
    .options {
      grid-template-columns: repeat(auto-fill, minmax(76px, 1fr));
    }
  }
</style>
