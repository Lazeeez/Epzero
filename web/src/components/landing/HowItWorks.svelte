<script lang="ts">
  import type { Trace } from "../../lib/engine/protocol";
  import { reveal } from "../../lib/reveal";
  import { UNSUPPORTED_BROWSER, wasmSimdSupported } from "../../lib/support";
  import { buildToy, type ToyGraph } from "../../lib/toy/toy";
  import type { ToyRenderer, ToyStage } from "../../lib/toy/toy-renderer";

  const K = 5;
  const DEMO_QUERY: [number, number] = [0.8, 0.24];

  const steps: { title: string; stage: ToyStage }[] = [
    { title: "Every item becomes a point", stage: { tilt: 0, links: 0, focus: null, play: "none", interactive: false } },
    { title: "Link each point to a few neighbours", stage: { tilt: 0, links: 1, focus: 0, play: "none", interactive: false } },
    { title: "Add express lanes", stage: { tilt: 1, links: 0.6, focus: null, play: "none", interactive: false } },
    { title: "Descend greedily", stage: { tilt: 1, links: 0.6, focus: null, play: "descent", interactive: false } },
    { title: "Widen the search at the bottom", stage: { tilt: 1, links: 0.6, focus: null, play: "all", interactive: false } },
    { title: "Your turn", stage: { tilt: 1, links: 0.6, focus: null, play: "all", interactive: true } },
  ];

  let active = $state(0);
  let canvas: HTMLCanvasElement | undefined = $state();
  let section: HTMLElement | undefined = $state();
  let graph = $state.raw<ToyGraph | null>(null);
  let renderer: ToyRenderer | null = null;
  let failed = $state(false);

  let ef = $state(12);
  let flat = $state(false);
  let query = $state<[number, number]>(DEMO_QUERY);
  let trace = $state.raw<Trace | null>(null);
  let revealed = $state(0);

  const count = $derived(graph ? graph.levels.length : 640);
  const perLayer = $derived.by(() => {
    if (!graph) return [];
    const out = new Array(graph.maxLevel + 1).fill(0);
    for (const level of graph.levels) for (let l = 0; l <= level; l++) out[l]++;
    return out;
  });
  const done = $derived(trace !== null && revealed >= trace.layer.length);
  const hits = $derived.by(() => {
    if (!trace) return 0;
    const truth = new Set(trace.exact);
    return Array.from(trace.results).filter((id) => truth.has(id)).length;
  });
  const currentLayer = $derived(trace && revealed > 0 ? trace.layer[revealed - 1] : (trace?.entryLayer ?? 0));

  // Build the toy index (real engine, a few ms) once the section is near the viewport.
  $effect(() => {
    if (!section) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        observer.disconnect();
        buildToy().then((g) => (graph = g), () => (failed = true));
      },
      { rootMargin: "600px 0px" },
    );
    observer.observe(section);
    return () => observer.disconnect();
  });

  $effect(() => {
    if (!canvas || !graph) return;
    let disposed = false;
    let instance: ToyRenderer | null = null;
    import("../../lib/toy/toy-renderer").then(({ ToyRenderer }) => {
      if (disposed) return;
      instance = new ToyRenderer(canvas!, graph!);
      instance.onProgress((n) => {
        if (n !== revealed) revealed = n;
      });
      instance.onQuery((x, y) => {
        query = [x, y];
        run(true);
      });
      renderer = instance;
      applyStage();
      run(false);
    });
    return () => {
      disposed = true;
      instance?.destroy();
      renderer = null;
    };
  });

  function focusNode(g: ToyGraph): number {
    let best = 0;
    let bestDistance = Infinity;
    for (let i = 0; i < g.levels.length; i++) {
      const dx = g.points[i * 2] - 0.42;
      const dy = g.points[i * 2 + 1] - 0.5;
      const d = dx * dx + dy * dy;
      if (g.levels[i] === 0 && d < bestDistance) {
        best = i;
        bestDistance = d;
      }
    }
    return best;
  }

  function applyStage() {
    if (!renderer || !graph) return;
    const base = steps[active].stage;
    renderer.setStage({
      ...base,
      focus: base.focus === null ? null : focusNode(graph),
      tilt: base.interactive && flat ? 0 : base.tilt,
    });
  }

  function run(fast: boolean) {
    if (!graph || !renderer) return;
    trace = graph.index.trace(new Float32Array(query), K, Math.max(ef, K));
    renderer.setTrace(query, trace, { restart: true, fast });
  }

  // Scroll position picks the step; the renderer morphs to it and continues any playback.
  $effect(() => {
    void active;
    void flat;
    applyStage();
  });

  let stepEls: HTMLElement[] = $state([]);
  // The active step is the last one whose middle has passed a trigger line: mid-screen on desktop,
  // below the pinned figure on phones. Computed from positions on every scroll (not from crossing
  // events), so jumping straight to a step, e.g. via a link or Page Down, still lands on it.
  $effect(() => {
    if (!section) return;
    let frame = 0;
    const update = () => {
      frame = 0;
      const box = section!.getBoundingClientRect();
      if (box.bottom < 0 || box.top > innerHeight) return;
      const line = innerHeight * (matchMedia("(max-width: 900px)").matches ? 0.8 : 0.62);
      let next = 0;
      stepEls.forEach((el, i) => {
        const r = el?.getBoundingClientRect();
        if (r && r.top + r.height / 2 < line) next = i;
      });
      if (next !== active) active = next;
    };
    const schedule = () => (frame ||= requestAnimationFrame(update));
    addEventListener("scroll", schedule, { passive: true });
    addEventListener("resize", schedule);
    schedule();
    return () => {
      cancelAnimationFrame(frame);
      removeEventListener("scroll", schedule);
      removeEventListener("resize", schedule);
    };
  });

  // Returning to the demo steps replays the demo query from the start.
  let lastActive = 0;
  $effect(() => {
    const step = active;
    if (step === 3 && lastActive < 3 && renderer) {
      query = DEMO_QUERY;
      ef = 12;
      run(false);
    }
    lastActive = step;
  });

  // Keyboard alternative to clicking the map: arrows move the query (shift for bigger steps).
  function onCanvasKey(event: KeyboardEvent) {
    if (active !== 5) return;
    const step = event.shiftKey ? 0.08 : 0.03;
    const moves: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    query = [Math.min(1, Math.max(0, query[0] + move[0])), Math.min(1, Math.max(0, query[1] + move[1]))];
    run(true);
  }

  function onEf(value: number) {
    ef = value;
    run(true);
  }
</script>

<section class="how section" id="how" bind:this={section} aria-labelledby="how-title">
  <div class="container">
    <header class="intro" use:reveal>
      <p class="eyebrow">How it works</p>
      <h2 id="how-title" class="section-title">Finding the needle without searching the haystack.</h2>
      <p class="section-lede">
        Epzero implements HNSW, the graph index behind most production vector databases. Scroll to build one, then watch a real
        search run through it. This isn't an animation of the algorithm: it's the Epzero engine, in WebAssembly, indexing
        {count} points and recording every decision it makes.
      </p>
    </header>

    <div class="scrolly">
      <div class="steps">
        <article class="step" class:active={active === 0} bind:this={stepEls[0]}>
          <span class="index num">01</span>
          <h3>{steps[0].title}</h3>
          <p>
            An embedding model turns each photo, sentence or product into a vector, so similar things land close together. Here
            they're 2-D so you can see them; the photos in the playground are 512-D.
          </p>
          <p>
            Search means: given a query, find the nearest points. Comparing against every point is exact but linear, and at a
            billion vectors that's a billion distance computations per query.
          </p>
        </article>

        <article class="step" class:active={active === 1} bind:this={stepEls[1]}>
          <span class="index num">02</span>
          <h3>{steps[1].title}</h3>
          <p>
            Connect every point to a handful of close neighbours (here <code>M = 6</code>). Now you can navigate: from anywhere,
            hop to whichever neighbour is closer to the query, and repeat.
          </p>
          <p>
            Epzero picks links with the paper's diversity heuristic: a neighbour is skipped if an already-linked one is closer to
            it, which keeps bridges between clusters so greedy routing doesn't get stuck.
          </p>
        </article>

        <article class="step" class:active={active === 2} bind:this={stepEls[2]}>
          <span class="index num">03</span>
          <h3>{steps[2].title}</h3>
          <p>
            Hops along short links are slow to cross the map. So each point also gets a random height, with each level
            <code>M</code>× rarer than the one below, like a skip list. Upper layers are sparse, with long links.
          </p>
          <p class="layers num">
            {#each perLayer as n, layer (layer)}
              <span><i style:--c="var(--layer-{layer})"></i>Layer {layer}: {n} points</span>
            {/each}
          </p>
        </article>

        <article class="step" class:active={active === 3} bind:this={stepEls[3]}>
          <span class="index num">04</span>
          <h3>{steps[3].title}</h3>
          <p>
            A search starts at the single entry point on the top layer. It moves to any neighbour closer to the query
            <span class="query-mark" role="img" aria-label="(the diamond)"></span>, and when none is, it drops a layer and continues from
            the same point.
          </p>
          <p>A few long hops cover most of the distance, touching only a few points.</p>
          <p class="name-note">
            That path is where the name comes from: <strong>e</strong>ntry <strong>p</strong>oint to layer <strong>zero</strong>.
          </p>
        </article>

        <article class="step" class:active={active === 4} bind:this={stepEls[4]}>
          <span class="index num">05</span>
          <h3>{steps[4].title}</h3>
          <p>
            On the dense bottom layer, greedy isn't enough. Epzero keeps a beam of the <code>ef</code> best candidates found so
            far, expands the closest unexpanded one, and stops when nothing left can beat the beam.
          </p>
          <p>
            <span class="swatch good"></span>Green rings are the answer. <span class="swatch warn"></span>A dashed amber ring is
            a true neighbour it missed. That's the recall trade-off: a wider beam finds more, at the cost of more work.
          </p>
        </article>

        <article class="step last" class:active={active === 5} bind:this={stepEls[5]}>
          <span class="index num">06</span>
          <h3>{steps[5].title}</h3>
          <p>
            Click anywhere on the map (or focus it and use the arrow keys) to search from there. Narrow the beam until it starts
            missing neighbours.
          </p>
          <div class="controls">
            <label>
              <span>Beam width <code>ef</code></span>
              <input
                type="range"
                min={K}
                max="64"
                value={ef}
                oninput={(e) => onEf(Number((e.currentTarget as HTMLInputElement).value))}
                disabled={!graph}
              />
              <output class="num">{ef}</output>
            </label>
            <div class="toggle" role="group" aria-label="View">
              <button class:on={!flat} onclick={() => (flat = false)}>Layers</button>
              <button class:on={flat} onclick={() => (flat = true)}>Flat</button>
            </div>
          </div>
        </article>
      </div>

      <figure class="figure">
        <div class="frame card">
          <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
          <canvas
            bind:this={canvas}
            class:interactive={active === 5}
            tabindex={active === 5 ? 0 : -1}
            onkeydown={onCanvasKey}
            aria-label={active === 5
              ? "Search map: use the arrow keys to move the query and search from there"
              : `An HNSW graph of ${count} points and a search through it`}
          ></canvas>
          {#if !graph}
            <div class="placeholder">
              {failed ? (wasmSimdSupported() ? "The demo couldn't load WebAssembly." : UNSUPPORTED_BROWSER) : "Building the index…"}
            </div>
          {/if}
          <div class="readout num" aria-live="polite">
            {#if trace && active >= 3}
              <span>Layer <b>{currentLayer}</b></span>
              <span><b>{revealed}</b> distance computations</span>
              <span class:dim={!done}>
                {#if done}
                  <b class:good={hits === K} class:warn={hits < K}>{hits}/{K}</b> true neighbours found · compared
                  <b>{((revealed / count) * 100).toFixed(0)}%</b> of points
                {:else}
                  searching…
                {/if}
              </span>
            {:else}
              <span>{count} points</span>
              {#if graph}<span>{graph.maxLevel + 1} layers</span><span><code>M = 6</code></span>{/if}
            {/if}
          </div>
        </div>
      </figure>
    </div>
  </div>
</section>

<style>
  .how {
    --layer-0: var(--text-2);
    --layer-1: var(--series-1);
    --layer-2: var(--accent-2);
    --layer-3: var(--series-3);
    --layer-4: var(--series-4);
  }
  .scrolly {
    display: grid;
    grid-template-columns: minmax(0, 420px) minmax(0, 1fr);
    gap: clamp(32px, 5vw, 72px);
    margin-top: 72px;
  }
  .steps {
    display: grid;
  }
  .step {
    min-height: 88vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 24px 0;
  }
  /* Steps other than the current one recede by color, not transparency: the muted color keeps
     WCAG AA contrast (>= 5:1 in both themes), so they stay readable ahead of time. */
  .step :global(*) {
    transition: color 0.5s var(--ease);
  }
  .step:not(.active) h3,
  .step:not(.active) p,
  .step:not(.active) :global(code),
  .step:not(.active) :global(strong) {
    color: var(--muted);
  }
  .step.last {
    min-height: 70vh;
  }
  .index {
    color: var(--accent-strong);
    font: 600 13px var(--mono);
  }
  h3 {
    margin: 10px 0 14px;
    font-size: clamp(24px, 2.6vw, 32px);
  }
  .step p {
    margin: 0 0 12px;
    color: var(--text-2);
    line-height: 1.7;
  }
  /* Where "Epzero" comes from, under the step that shows it. */
  .step p.name-note {
    padding-left: 12px;
    border-left: 2px solid var(--result);
    font-size: 14px;
  }
  .name-note strong {
    color: var(--text);
  }
  .layers {
    display: grid;
    gap: 6px;
    font: 13px var(--mono);
  }
  .layers span {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .layers i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--c);
  }
  .query-mark {
    display: inline-block;
    width: 10px;
    height: 10px;
    margin: 0 2px;
    border: 2px solid var(--result);
    transform: rotate(45deg);
  }
  .swatch {
    display: inline-block;
    width: 11px;
    height: 11px;
    margin-right: 6px;
    border: 2px solid;
    border-radius: 50%;
    vertical-align: -1px;
  }
  .swatch.good {
    border-color: var(--good);
  }
  .swatch.warn {
    border-color: var(--warn);
    border-style: dashed;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px;
    margin-top: 8px;
  }
  .controls label {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text-2);
    font-size: 14px;
  }
  .controls input {
    width: 150px;
    accent-color: var(--accent);
  }
  .controls output {
    min-width: 24px;
    font: 14px var(--mono);
    color: var(--text);
  }
  .toggle {
    display: flex;
    padding: 3px;
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .toggle button {
    padding: 5px 12px;
    border: 0;
    border-radius: 7px;
    background: transparent;
    color: var(--muted);
    font-size: 13px;
    cursor: pointer;
  }
  .toggle button.on {
    background: var(--surface-3);
    color: var(--text);
  }
  .figure {
    position: sticky;
    top: calc(var(--nav-h) + max(16px, (100vh - var(--nav-h) - min(76vh, 680px)) / 2));
    height: min(76vh, 680px);
    margin: 0;
  }
  .frame {
    position: relative;
    height: 100%;
    overflow: hidden;
    background: color-mix(in srgb, var(--surface-solid) 70%, transparent);
  }
  canvas {
    display: block;
    width: 100%;
    height: calc(100% - 44px);
  }
  canvas:focus-visible {
    outline-offset: -3px; /* inside the frame, which clips overflow */
  }
  canvas.interactive {
    cursor: crosshair;
  }
  .placeholder {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    padding: 24px;
    color: var(--muted);
    font: 13px var(--mono);
    line-height: 1.6;
    text-align: center;
  }
  .readout {
    position: absolute;
    inset: auto 0 0 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 20px;
    min-height: 44px;
    padding: 10px 18px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font: 12.5px var(--mono);
  }
  .readout b {
    color: var(--text);
    font-weight: 600;
  }
  .readout b.good {
    color: var(--good);
  }
  .readout b.warn {
    color: var(--warn);
  }
  .dim {
    color: var(--muted); /* already the muted color: no opacity on top, to keep contrast */
  }
  /* Phones: the figure pins to the top and the steps scroll up beneath it as cards. A flex column
     (not a grid) so the sticky figure's containing block is the whole section. */
  @media (max-width: 900px) {
    .scrolly {
      display: flex;
      flex-direction: column;
      gap: 0;
      margin-top: 40px;
    }
    .figure {
      order: -1;
      z-index: 2;
      top: calc(var(--nav-h) + 8px);
      height: 46vh;
    }
    .frame {
      background: var(--surface-solid);
    }
    .steps {
      gap: 42vh;
      padding: 50vh 0 30vh;
    }
    .step,
    .step.last {
      min-height: 0;
      padding: 20px;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--overlay-fill);
    }
  }
</style>
