<script lang="ts">
  import { imageEncoder, textEncoder } from "../lib/engine/encoders.svelte";
  import { engine, type SearchOutcome } from "../lib/engine/engine.svelte";
  import type { Query } from "../lib/engine/protocol";
  import { sceneController } from "../lib/scene/scene.svelte";
  import Thumb from "../components/Thumb.svelte";
  import Race from "../components/try/Race.svelte";
  import ResultsGrid from "../components/try/ResultsGrid.svelte";
  import SearchBox from "../components/try/SearchBox.svelte";
  import SearchStats from "../components/try/SearchStats.svelte";
  import Timeline from "../components/try/Timeline.svelte";
  import TunePanel from "../components/try/TunePanel.svelte";
  import NavHud from "../components/try/NavHud.svelte";
  import ViewControls from "../components/try/ViewControls.svelte";
  import type { BlendBase } from "../components/try/BlendBox.svelte";

  let ef = $state(64);
  let outcome = $state.raw<SearchOutcome | null>(null);
  let encodeMs = $state<number | null>(null);
  let raceOpen = $state(false);
  let tuneOpen = $state(false);
  // Hide the search panel to explore the graph with the whole screen.
  let panelHidden = $state(false);
  let panelInset = $state(0); // phones: how much of the bottom the panel covers
  // The photo the Blend tab mixes words into: the latest photo query, or else the top result.
  let blendBase = $state.raw<BlendBase | null>(null);
  let panel: HTMLElement | undefined = $state();

  const params = new URLSearchParams(location.search);
  const initialText = params.get("q") ?? "";
  const initialPhoto = params.has("photo") ? Number(params.get("photo")) : null;
  const initialBlend = params.has("plus") ? { words: params.get("plus")!, mix: Math.min(1, Math.max(0, Number(params.get("mix") ?? 0.5) || 0.5)) } : null;

  let searchError = $state("");

  /** Runs a search and replays it in 3D. `live` (slider-driven) keeps the camera still and replays fast. */
  async function search(query: Query, { live = false } = {}) {
    let result: SearchOutcome;
    try {
      result = await engine.search(query, ef);
    } catch (error) {
      searchError = error instanceof Error ? error.message : String(error);
      return;
    }
    searchError = "";
    outcome = result;
    encodeMs =
      query.kind === "id" || query.source === "blend" ? null : query.source === "text" ? textEncoder.lastMs : imageEncoder.lastMs;
    if (sceneController.scene) sceneController.scene.play(result.trace, { flyTo: !live, durationScale: live ? 0.35 : 1 });
    else unplayed = result; // the 3D scene is still loading: replay it once it's there
    if (query.kind === "id") blendBase = { id: query.id };
    else if (query.source === "photo" || query.source === "camera") blendBase = { vector: query.vector, preview: query.preview };
    else if (query.source === "text") blendBase = { id: result.trace.results[0] };
    // Shareable URL for text, photo and blend queries (image uploads have no stable identity).
    const url = new URL(location.href);
    url.search = "";
    if (query.kind === "id") url.searchParams.set("photo", String(query.id));
    else if (query.source === "text") url.searchParams.set("q", query.text);
    else if (query.source === "blend" && query.blend?.id !== undefined) {
      url.searchParams.set("photo", String(query.blend.id));
      url.searchParams.set("plus", query.blend.words);
      url.searchParams.set("mix", query.blend.mix.toFixed(2));
    }
    history.replaceState(history.state, "", url);
  }

  let unplayed: SearchOutcome | null = null;
  $effect(() => {
    if (!sceneController.ready || !unplayed) return;
    sceneController.scene?.play(unplayed.trace);
    unplayed = null;
  });

  const searchId = (id: number) => search({ kind: "id", id });
  const surprise = () => searchId(engine.randomId());

  // Clicking a node in the 3D view searches for photos like it.
  $effect(() => {
    sceneController.onNodeClick((id) => searchId(id));
    return () => sceneController.onNodeClick(null);
  });

  // First query once the index is ready: from the URL if there is one, else a random photo.
  let started = false;
  $effect(() => {
    if (engine.status !== "ready" || started) return;
    started = true;
    if (initialPhoto !== null && initialPhoto >= 0 && initialPhoto < engine.count) searchId(initialPhoto); // a blend link continues in the Blend tab
    else if (initialText)
      textEncoder.encode(initialText).then(
        (vector) => search({ kind: "vector", source: "text", text: initialText, vector }),
        () => surprise(),
      );
    else if (engine.last && engine.last.query.kind === "vector") search(engine.last.query); // back from another page
    else surprise();
  });

  // Re-run the current query live when ef changes.
  let efTimer = 0;
  function onEf(value: number) {
    ef = value;
    clearTimeout(efTimer);
    efTimer = window.setTimeout(() => outcome && search(outcome.query), 150);
  }

  // Keep the graph centered in the space the panel leaves free.
  $effect(() => {
    if (!panel || !sceneController.ready) return; // ready is reactive; the scene loads after first paint
    const hidden = panelHidden;
    const update = () => {
      const box = panel!.getBoundingClientRect();
      panelInset = hidden || innerWidth > 900 ? 0 : Math.max(0, innerHeight - box.top);
      if (hidden) sceneController.scene?.setInsets(0, 0);
      else if (innerWidth > 900) sceneController.scene?.setInsets(box.right, 0);
      else sceneController.scene?.setInsets(0, panelInset);
    };
    const observer = new ResizeObserver(update);
    observer.observe(panel);
    addEventListener("resize", update);
    update();
    return () => {
      observer.disconnect();
      removeEventListener("resize", update);
      sceneController.scene?.setInsets(0, 0);
    };
  });

  const hover = $derived(sceneController.hover);
  const selected = $derived(sceneController.selected);

  function searchSelected() {
    const pick = sceneController.selected;
    if (!pick) return;
    sceneController.selected = null;
    if (panelHidden && innerWidth <= 900) panelHidden = false; // show the results
    searchId(pick.id);
  }
</script>

<div class="try">
  <aside class="panel card" class:collapsed={panelHidden} bind:this={panel} aria-label="Search" inert={panelHidden}>
    <header>
      <p class="eyebrow">Playground</p>
      <h1>Search {engine.count ? engine.count.toLocaleString() : "30,000"} photos, locally.</h1>
      <p class="lede">
        Oryx's C++ engine runs in this tab as WebAssembly. Search by description, photo or camera, then step through exactly how
        the graph found the answer.
      </p>
    </header>

    <SearchBox
      onsearch={search}
      onsurprise={surprise}
      disabled={engine.status !== "ready" || engine.rebuilding}
      {initialText}
      {blendBase}
      {initialBlend}
    />

    {#if searchError}<p class="search-error" role="alert">Search failed: {searchError}</p>{/if}

    {#if outcome}
      <SearchStats {outcome} {encodeMs} />
      <ResultsGrid {outcome} onpick={searchId} />
    {/if}
  </aside>

  <div class="knobs card" role="group" aria-label="Search settings">
    <label class="ef">
      <span><span class="wide-only">Beam width&nbsp;</span><code>ef</code></span>
      <input type="range" min="16" max="256" step="16" value={ef} oninput={(e) => onEf(Number((e.currentTarget as HTMLInputElement).value))} />
      <output class="num">{ef}</output>
    </label>
    <div class="buttons">
      <button class="btn btn-ghost" onclick={() => (raceOpen = true)} disabled={!outcome}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 3 4 14h7l-1 7 9-11h-7z" /></svg>
        Race vs. exact
      </button>
      <button
        class="btn btn-ghost icon-only"
        class:on={tuneOpen}
        onclick={() => (tuneOpen = !tuneOpen)}
        disabled={engine.status !== "ready"}
        aria-label="Index settings"
        aria-expanded={tuneOpen}
        title="Tune and rebuild the index"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h10M18 7h2M4 17h4M12 17h8" /><circle cx="16" cy="7" r="2.2" /><circle cx="10" cy="17" r="2.2" /></svg>
      </button>
    </div>
  </div>

  <div class="view-anchor" style:--panel-inset="{panelInset}px">
    <ViewControls {panelHidden} ontogglepanel={() => (panelHidden = !panelHidden)} />
  </div>

  {#if tuneOpen}
    <div class="tune-anchor">
      <TunePanel {ef} onclose={() => (tuneOpen = false)} onrebuilt={() => outcome && search(outcome.query)} />
    </div>
  {/if}

  {#if outcome}
    <NavHud {outcome} />
    <div class="dock" class:wide={panelHidden}>
      <Timeline {outcome} />
    </div>
  {/if}

  {#if engine.status !== "ready"}
    <div class="loading card" role="status" aria-live="polite">
      {#if engine.status === "error"}
        <strong>Something went wrong</strong>
        <p>{engine.error}</p>
        <button class="btn btn-primary retry" onclick={() => engine.retry()}>Try again</button>
      {:else}
        <strong>{engine.stage || "Starting the engine"}</strong>
        <div class="meter"><i style:width="{(engine.progress * 100).toFixed(1)}%"></i></div>
        <p>C++ compiled to WebAssembly, building on every core of your CPU. Nodes appear as they are inserted.</p>
      {/if}
    </div>
  {/if}

  {#if hover}
    <div class="hovercard card" style:left="{Math.min(Math.max(hover.x, 130), innerWidth - 130)}px" style:top="{Math.max(hover.y, 220)}px">
      <Thumb id={hover.id} size="n" alt={engine.caption(hover.id)} />
      <div>
        <strong>{engine.caption(hover.id)}</strong>
        <span>© {engine.author(hover.id)} · CC BY 2.0 · click to search</span>
      </div>
    </div>
  {/if}

  {#if selected}
    <div
      class="selectcard card"
      role="dialog"
      aria-label="Selected photo"
      style:left="{Math.min(Math.max(selected.x, 140), innerWidth - 140)}px"
      style:top="{Math.min(Math.max(selected.y, 420), innerHeight - 20)}px"
    >
      <Thumb id={selected.id} size="n" alt={engine.caption(selected.id)} />
      <div class="info">
        <strong>{engine.caption(selected.id)}</strong>
        <span>© {engine.author(selected.id)} · CC BY 2.0 · photo #{selected.id}</span>
      </div>
      <div class="actions">
        <button class="btn btn-primary" onclick={searchSelected}>Find similar</button>
        <button class="btn btn-ghost" onclick={() => selected && sceneController.scene?.focusNode(selected.id)}>Zoom here</button>
        <button
          class="btn btn-ghost x"
          onclick={() => ((sceneController.selected = null), sceneController.scene?.clearMarker())}
          aria-label="Close">×</button
        >
      </div>
    </div>
  {/if}

  {#if raceOpen && outcome}
    <Race {outcome} onclose={() => (raceOpen = false)} />
  {/if}
</div>

<style>
  .try {
    height: 100dvh;
    padding-top: var(--nav-h);
    pointer-events: none; /* the 3D scene underneath stays interactive */
  }
  .try > :global(*) {
    pointer-events: auto;
  }
  .panel {
    position: fixed;
    top: calc(var(--nav-h) + 16px);
    left: 24px;
    bottom: calc(24px + env(safe-area-inset-bottom, 0px));
    width: 400px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 20px;
    background: var(--glass);
    overflow-y: auto;
    scrollbar-width: thin;
  }
  h1 {
    font-size: 24px;
    letter-spacing: -0.03em;
  }
  .panel .eyebrow {
    margin-bottom: 10px;
  }
  .lede {
    margin: 8px 0 0;
    color: var(--text-2);
    font-size: 13.5px;
  }
  .knobs {
    position: fixed;
    top: calc(var(--nav-h) + 16px);
    right: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 12px 10px 16px;
    background: var(--glass);
  }
  .ef {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text-2);
    font-size: 13px;
    white-space: nowrap;
  }
  .ef input {
    width: 130px;
    accent-color: var(--accent);
  }
  .ef output {
    min-width: 28px;
    color: var(--text);
    font: 13px var(--mono);
  }
  .buttons {
    display: flex;
    gap: 8px;
  }
  .buttons .btn {
    height: 36px;
    padding: 0 12px;
    font-size: 13.5px;
  }
  .buttons svg {
    width: 15px;
    height: 15px;
    fill: none;
    stroke: currentColor;
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .icon-only {
    width: 36px;
    padding: 0 !important;
  }
  .dock {
    position: fixed;
    left: 448px;
    right: 88px;
    bottom: calc(24px + env(safe-area-inset-bottom, 0px));
    transition: left 0.35s var(--ease);
  }
  .dock.wide {
    left: 24px;
  }
  .panel {
    transition:
      transform 0.4s var(--ease),
      opacity 0.3s;
  }
  .panel.collapsed {
    opacity: 0;
    transform: translateX(calc(-100% - 32px));
    pointer-events: none !important;
  }
  .view-anchor {
    position: fixed;
    z-index: 4;
    right: 24px;
    bottom: calc(24px + env(safe-area-inset-bottom, 0px));
  }
  .selectcard {
    position: fixed;
    z-index: 30;
    display: grid;
    width: 260px;
    overflow: hidden;
    background: var(--surface-solid);
    transform: translate(-50%, calc(-100% - 14px));
    animation: pop-in 0.25s var(--ease);
  }
  .selectcard :global(img) {
    display: block;
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
  }
  .selectcard .info {
    display: grid;
    gap: 2px;
    padding: 10px 12px 4px;
    font-size: 13px;
  }
  .selectcard .info span {
    color: var(--muted);
    font-size: 11.5px;
  }
  .selectcard .actions {
    display: flex;
    gap: 6px;
    padding: 8px 10px 10px;
  }
  .selectcard .btn {
    height: 36px;
    padding: 0 12px;
    font-size: 13px;
  }
  .selectcard .x {
    width: 36px;
    margin-left: auto;
    padding: 0;
    font-size: 18px;
  }
  @keyframes pop-in {
    from {
      opacity: 0;
      transform: translate(-50%, calc(-100% - 6px)) scale(0.97);
    }
  }
  .loading {
    position: fixed;
    left: calc(50% + 212px);
    top: 50%;
    width: min(420px, 100% - 48px);
    display: grid;
    gap: 12px;
    padding: 20px;
    background: var(--glass);
    transform: translate(-50%, -50%);
  }
  .icon-only.on {
    border-color: var(--accent);
    color: var(--accent-strong);
  }
  .tune-anchor {
    position: fixed;
    z-index: 20;
    top: calc(var(--nav-h) + 78px);
    right: 24px;
  }
  @media (max-width: 900px) {
    .tune-anchor {
      top: calc(var(--nav-h) + 64px);
      right: 16px;
      left: 16px;
    }
    .tune-anchor > :global(*) {
      width: 100%;
    }
  }
  .retry {
    justify-self: start;
  }
  .search-error {
    margin: 0;
    color: var(--warn);
    font-size: 13px;
  }
  .loading p {
    margin: 0;
    color: var(--text-2);
    font-size: 13px;
  }
  .meter {
    height: 6px;
    border-radius: 99px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .meter i {
    display: block;
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    transition: width 0.25s var(--ease);
  }
  @media (max-width: 480px) {
    .wide-only {
      display: none;
    }
  }
  .panel > :global(*) {
    min-width: 0;
  }
  .hovercard {
    position: fixed;
    z-index: 30;
    width: 230px;
    overflow: hidden;
    background: var(--surface-solid);
    pointer-events: none !important;
    transform: translate(-50%, calc(-100% - 18px));
  }
  .hovercard :global(img) {
    display: block;
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
  }
  .hovercard div {
    display: grid;
    gap: 2px;
    padding: 9px 11px 11px;
    font-size: 12px;
  }
  .hovercard span {
    color: var(--muted);
  }
  @media (max-width: 900px) {
    .panel {
      top: auto;
      left: 12px;
      right: 12px;
      bottom: calc(12px + env(safe-area-inset-bottom, 0px)); /* clear the iPhone home indicator */
      width: auto;
      max-height: 52dvh;
      padding: 16px;
    }
    .panel header .lede,
    .panel header .eyebrow {
      display: none;
    }
    h1 {
      font-size: 19px;
    }
    .knobs {
      left: 12px;
      right: 12px;
      top: calc(var(--nav-h) + 8px);
      justify-content: space-between;
      padding: 8px 10px;
    }
    .ef input {
      flex: 1;
      width: auto;
      min-width: 60px;
    }
    .ef {
      flex: 1;
      min-width: 0;
    }
    .buttons .btn:not(.icon-only) {
      font-size: 0;
      width: 36px;
      padding: 0;
    }
    .dock {
      display: none;
    }
    .panel.collapsed {
      transform: translateY(calc(100% + 24px));
    }
    /* Phones: the tapped photo's card docks under the controls instead of floating at the tap. */
    .selectcard {
      top: calc(var(--nav-h) + 68px) !important;
      left: 12px !important;
      right: 12px;
      width: auto;
      grid-template-columns: 96px minmax(0, 1fr);
      transform: none;
      animation: none;
    }
    .selectcard :global(.thumb) {
      grid-row: span 2;
      height: 100%;
    }
    .selectcard :global(img) {
      height: 100%;
      aspect-ratio: auto;
    }
    .selectcard .info {
      padding: 10px 12px 0;
    }
    .view-anchor {
      right: 12px;
      bottom: calc(max(var(--panel-inset, 0px), env(safe-area-inset-bottom, 0px)) + 12px);
      transition: bottom 0.35s var(--ease);
    }
    .loading {
      left: 50%;
      top: 35%;
    }
  }
</style>
