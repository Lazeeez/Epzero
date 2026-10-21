<script lang="ts">
  import { engine, type SearchOutcome } from "../../lib/engine/engine.svelte";
  import { agreement } from "../../lib/engine/trace-analysis";
  import { router } from "../../lib/router.svelte";
  import { sceneController } from "../../lib/scene/scene.svelte";
  import Thumb from "../Thumb.svelte";

  let live = $state.raw<SearchOutcome | null>(null);
  let queries = $state(0);
  let visible = $state(true);
  let hero: HTMLElement | undefined = $state();

  // Live demo: a real query every few seconds, on the real index, while the hero is on screen.
  $effect(() => {
    if (engine.status !== "ready" || !visible) return;
    let timer = 0;
    let cancelled = false;
    const tick = async () => {
      try {
        const outcome = await engine.search({ kind: "id", id: engine.randomId() }, 64, 12);
        if (cancelled) return;
        live = outcome;
        queries++;
        sceneController.scene?.play(outcome.trace, { flyTo: false, durationScale: 0.7 });
      } catch {
        /* the next tick retries */
      }
      if (!cancelled) timer = window.setTimeout(tick, 5200);
    };
    timer = window.setTimeout(tick, queries ? 1500 : 400);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  });

  // The 3D backdrop fades as the story scrolls past the hero; the demo pauses off screen.
  $effect(() => {
    const root = document.documentElement;
    const onScroll = () => {
      const fade = 1 - scrollY / (innerHeight * 0.85);
      root.style.setProperty("--hero-scene-opacity", String(Math.max(0.06, Math.min(1, fade))));
    };
    onScroll();
    addEventListener("scroll", onScroll, { passive: true });
    // Below the hero the backdrop is a faint still frame: stop rendering it (and the demo queries).
    const observer = new IntersectionObserver(
      ([entry]) => {
        visible = entry.isIntersecting;
        sceneController.setVisible(visible);
      },
      { threshold: 0 },
    );
    if (hero) observer.observe(hero);
    return () => {
      removeEventListener("scroll", onScroll);
      observer.disconnect();
      sceneController.setVisible(true);
      root.style.removeProperty("--hero-scene-opacity");
    };
  });

  const time = (us: number) => (us >= 1000 ? `${(us / 1000).toFixed(1)} ms` : `${Math.round(us)} µs`);
</script>

<section class="hero" bind:this={hero} aria-labelledby="hero-title">
  <div class="shade" aria-hidden="true"></div>
  <div class="container content">
    <p class="eyebrow"><span class="pulse" aria-hidden="true"></span> Running live in this tab</p>
    <h1 id="hero-title"><span class="gradient-text">A vector database,</span><br />built from scratch.</h1>
    <p class="lede">
      Epzero is an HNSW search engine I wrote in C++20: hand-tuned SIMD, lock-free reads, 8-bit quantization, a crash-safe
      write-ahead log and replication. Behind this text, it's indexing <strong>{(engine.count || 29825).toLocaleString()} photos</strong>
      on your CPU, compiled to WebAssembly.
    </p>
    <div class="actions">
      <a class="btn btn-primary" href={router.href("try")}>Search it yourself <span aria-hidden="true">→</span></a>
      <a class="btn btn-ghost" href="#how">How it works</a>
    </div>

    <div class="telemetry num" aria-live="polite">
      {#if engine.status === "ready" && engine.graph}
        {#if engine.graph.prebuilt}
          <span><i class="ok"></i>Index ready in {Math.max(1, Math.round(engine.buildMs))} ms · prebuilt graph</span>
        {:else}
          <span><i class="ok"></i>Index built in {(engine.buildMs / 1000).toFixed(1)} s on {engine.graph.threads} {engine.graph.threads === 1 ? "thread" : "threads"}</span>
        {/if}
        <span>{engine.graph.maxLevel + 1} layers</span>
        <span>WebAssembly SIMD128</span>
      {:else if engine.status === "error"}
        <span class="failed">The in-browser engine couldn't start: {engine.error}</span>
        {#if !engine.unsupported}<button class="retry" onclick={() => engine.retry()}>Try again</button>{/if}
      {:else}
        <span class="progress"><i class="spin"></i>{engine.stage || "Starting the engine"}</span>
        <span class="bar"><b style:width="{(engine.progress * 100).toFixed(1)}%"></b></span>
      {/if}
    </div>
  </div>

  {#if live}
    {@const trace = live.trace}
    {#key queries}
      <aside class="live card" aria-label="Live query">
        <header>
          <span class="tag">Live query</span>
          <span class="num">{time(trace.micros)}</span>
        </header>
        <div class="strip">
          <span class="query"><Thumb id={trace.results[0]} size="q" /></span>
          <span class="arrow" aria-hidden="true">→</span>
          {#each Array.from(trace.results).slice(1, 5) as id (id)}
            <Thumb {id} size="q" />
          {/each}
        </div>
        <p class="num">
          Scanned {((trace.distanceComputations / engine.count) * 100).toFixed(1)}% of the photos ·
          {agreement(trace)}/{trace.results.length} match an exact scan · {Math.round(trace.exactMicros / Math.max(trace.micros, 1))}× faster
        </p>
      </aside>
    {/key}
  {/if}

  <a class="scroll" href="#problem" aria-label="Scroll to the story">
    <span></span>
  </a>
</section>

<style>
  .hero {
    position: relative;
    display: grid;
    align-items: center;
    min-height: 100svh;
    padding: calc(var(--nav-h) + 40px) 0 80px;
  }
  .shade {
    position: absolute;
    inset: 0;
    background:
      radial-gradient(ellipse 70% 60% at 18% 45%, color-mix(in srgb, var(--page) 88%, transparent), transparent 70%),
      linear-gradient(180deg, transparent 70%, var(--page));
    pointer-events: none;
  }
  .content {
    position: relative;
  }
  .pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--good);
    box-shadow: 0 0 0 0 color-mix(in srgb, var(--good) 60%, transparent);
    animation: ping 2s infinite;
  }
  h1 {
    max-width: 820px;
    font-size: clamp(44px, 7.6vw, 96px);
    font-weight: 700;
    letter-spacing: -0.045em;
    line-height: 0.98;
  }
  .lede {
    max-width: 610px;
    margin: 26px 0 0;
    color: var(--text-2);
    font-size: clamp(16px, 1.6vw, 19px);
    line-height: 1.6;
  }
  .lede strong {
    color: var(--text);
    font-weight: 600;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 34px;
  }
  .actions .btn {
    height: 50px;
    padding: 0 24px;
    font-size: 15.5px;
  }
  .telemetry {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 22px;
    margin-top: 40px;
    color: var(--muted);
    font: 12.5px var(--mono);
  }
  .telemetry span {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }
  .failed {
    color: var(--warn);
  }
  .retry {
    padding: 4px 12px;
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    background: var(--surface);
    color: var(--text);
    font: inherit;
    cursor: pointer;
  }
  .ok {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--good);
  }
  .spin {
    width: 12px;
    height: 12px;
    border: 2px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .bar {
    width: 160px;
    height: 4px;
    border-radius: 99px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .bar b {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    transition: width 0.25s var(--ease);
  }
  .live {
    position: absolute;
    right: max(24px, calc((100% - var(--content)) / 2));
    bottom: 96px;
    width: 330px;
    padding: 14px;
    background: var(--overlay-fill);
    animation: live-in 0.6s var(--ease);
  }
  .live header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    font: 12px var(--mono);
    color: var(--text-2);
  }
  .tag {
    color: var(--accent-strong);
  }
  .strip {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .strip :global(.thumb) {
    width: 48px;
    height: 48px;
    border-radius: 9px;
    overflow: hidden;
  }
  .strip :global(img) {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .query :global(.thumb) {
    box-shadow: 0 0 0 2px var(--result);
  }
  .arrow {
    color: var(--muted);
  }
  .live p {
    margin: 10px 0 0;
    color: var(--muted);
    font-size: 11.5px;
    line-height: 1.5;
  }
  .scroll {
    position: absolute;
    left: 50%;
    bottom: 28px;
    width: 26px;
    height: 42px;
    border: 1.5px solid var(--border-strong);
    border-radius: 99px;
    transform: translateX(-50%);
  }
  .scroll span {
    position: absolute;
    left: 50%;
    top: 8px;
    width: 3px;
    height: 8px;
    border-radius: 99px;
    background: var(--text-2);
    transform: translateX(-50%);
    animation: wheel 1.8s var(--ease) infinite;
  }
  @keyframes ping {
    70% {
      box-shadow: 0 0 0 8px transparent;
    }
    100% {
      box-shadow: 0 0 0 0 transparent;
    }
  }
  @keyframes wheel {
    0% {
      opacity: 0;
      transform: translate(-50%, 0);
    }
    30% {
      opacity: 1;
    }
    100% {
      opacity: 0;
      transform: translate(-50%, 14px);
    }
  }
  @keyframes live-in {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
  }
  /* Phones: the loading line and the ready line wrap differently. Reserving both rows keeps the
     (vertically centered) hero from shifting when the index comes up. */
  @media (max-width: 640px) {
    .telemetry {
      min-height: 48px;
      align-content: flex-start;
    }
  }
  @media (max-width: 900px) {
    .live {
      display: none;
    }
    /* On narrow screens the text sits over the graph: veil it more. */
    .shade {
      background: linear-gradient(180deg, color-mix(in srgb, var(--page) 45%, transparent), color-mix(in srgb, var(--page) 78%, transparent) 55%, var(--page));
    }
  }
</style>
