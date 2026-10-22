<script lang="ts">
  import { untrack } from "svelte";
  import { freeLook, setFreeLook } from "../lib/free-look.svelte";
  import { imageEncoder, textEncoder } from "../lib/engine/encoders.svelte";
  import { engine, type SearchOutcome } from "../lib/engine/engine.svelte";
  import type { Query } from "../lib/engine/protocol";
  import { sceneController } from "../lib/scene/scene.svelte";
  import Thumb from "../components/Thumb.svelte";
  import Race from "../components/try/Race.svelte";
  import ResultsGrid from "../components/try/ResultsGrid.svelte";
  import CategoryFilter from "../components/try/CategoryFilter.svelte";
  import SearchBox from "../components/try/SearchBox.svelte";
  import SearchStats from "../components/try/SearchStats.svelte";
  import Timeline from "../components/try/Timeline.svelte";
  import TunePanel from "../components/try/TunePanel.svelte";
  import NavHud from "../components/try/NavHud.svelte";
  import PhotoSource from "../components/try/PhotoSource.svelte";
  import ViewControls from "../components/try/ViewControls.svelte";
  import RecallGuard from "../components/try/RecallGuard.svelte";
  import Sharpening from "../components/try/Sharpening.svelte";
  import WhyNot from "../components/try/WhyNot.svelte";
  import type { BlendBase } from "../components/try/BlendBox.svelte";

  const EF_MAX = 512;
  let ef = $state(64);
  let outcome = $state.raw<SearchOutcome | null>(null);
  let encodeMs = $state<number | null>(null);
  let raceOpen = $state(false);
  let tuneOpen = $state(false);
  // Hide the search panel to explore the graph with the whole screen.
  const panelHidden = $derived(freeLook.on);
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
    // Shareable URL for text, photo and blend queries (image uploads have no stable identity, and
    // nor do the visitor's own photos: a link would open the demo set).
    if (engine.source === "mine") return;
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

  // Another set of photos (the visitor's own, or the demo again): once it's indexed, ask the same
  // description of it, or start from a random photo.
  let shownCollection = engine.collection;
  let carried: Query | null | undefined; // undefined: no switch pending
  $effect(() => {
    const collection = engine.collection;
    if (collection === shownCollection) return;
    shownCollection = collection;
    untrack(() => {
      // The old results name photos the new set doesn't have: clear them at once.
      const previous = outcome?.query;
      carried = previous?.kind === "vector" && previous.source === "text" ? previous : null;
      outcome = null;
      blendBase = null;
      sceneController.selected = null;
    });
  });
  $effect(() => {
    if (engine.status !== "ready" || carried === undefined) return;
    const query = carried;
    carried = undefined;
    untrack(() => (query ? search(query) : surprise()));
  });
  // Back on the playground after other pages used the demo: return to the visitor's photos.
  $effect(() => {
    if (engine.preferOwn && engine.own && engine.source === "demo") void engine.useOwn(engine.own);
  });

  // Search starts on the vectors' leading dimensions; each further block that arrives re-runs the
  // current query on more of them, so its results visibly sharpen.
  let shownDims = 0;
  $effect(() => {
    const dims = engine.dims;
    if (shownDims && dims > shownDims) untrack(() => outcome && search(outcome.query, { live: true }));
    shownDims = dims;
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
      if (hidden) sceneController.scene?.setInsets(0, 0);
      else if (innerWidth > 900) sceneController.scene?.setInsets(box.right, 0);
      else sceneController.scene?.setInsets(0, Math.max(0, innerHeight - box.top)); // phones: the sheet covers the bottom
    };
    const observer = new ResizeObserver(update);
    observer.observe(panel);
    addEventListener("resize", update);
    // Hiding or showing the panel slides it (a transform, which ResizeObserver doesn't see):
    // measure again once it has arrived.
    const target = panel;
    target.addEventListener("transitionend", update);
    update();
    return () => {
      observer.disconnect();
      target.removeEventListener("transitionend", update);
      removeEventListener("resize", update);
      sceneController.scene?.setInsets(0, 0);
    };
  });

  const hover = $derived(sceneController.hover);
  const selected = $derived(sceneController.selected);

  // Share the current search: the native share sheet on phones, the clipboard elsewhere.
  let toast = $state("");
  let toastTimer = 0;
  function say(message: string, ms = 2600) {
    toast = message;
    clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => (toast = ""), ms);
  }
  async function share() {
    const url = location.href;
    try {
      if (navigator.share && matchMedia("(pointer: coarse)").matches) {
        await navigator.share({ title: document.title, url });
        return;
      }
      await navigator.clipboard.writeText(url);
      say("Link copied: it opens this exact search");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return; // share sheet dismissed
      say("Couldn't copy the link; copy it from the address bar");
    }
  }

  function searchSelected() {
    const pick = sceneController.selected;
    if (!pick) return;
    sceneController.selected = null;
    if (panelHidden && innerWidth <= 900) setFreeLook(false); // show the results
    searchId(pick.id);
  }

  // Free look: everything on the page slides away (search, settings, debugger, HUD, view controls),
  // leaving the graph under the top bar, which holds the button that brings it all back. The view
  // controls' reset button turns it on (with H); hovering or tapping a star still shows its photo.
  $effect(() => {
    if (freeLook.on) tuneOpen = false;
  });
  $effect(() => () => void (freeLook.on = false)); // leaving the page ends it
  // The category filter belongs to this page: other pages (the live hero, the game) search everything.
  $effect(() => () => {
    if (engine.filter) engine.setFilter([]).catch(() => {});
  });
  $effect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || document.querySelector("dialog[open]")) return;
      if ((event.target as HTMLElement).closest("input, textarea, select, [contenteditable]")) return;
      if (event.key === "h" || event.key === "H") setFreeLook(!panelHidden);
      else if (event.key === "Escape" && panelHidden && !sceneController.selected) setFreeLook(false);
      else return;
      event.preventDefault();
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  });
</script>

<div class="try">
  <aside class="panel card" class:collapsed={panelHidden} bind:this={panel} aria-label="Search" inert={panelHidden}>
    <header>
      <p class="eyebrow">Playground</p>
      {#if engine.source === "mine"}
        <h1>Search {engine.count ? engine.count.toLocaleString() : "your"} of your photos, locally.</h1>
        <p class="lede">
          CLIP embedded them in this tab and Epzero's C++ engine searches them as WebAssembly. Nothing was uploaded. Search by
          description, photo or camera, then step through how the graph found the answer.
        </p>
      {:else}
        <h1>Search {engine.count ? engine.count.toLocaleString() : "30,000"} photos, locally.</h1>
        <p class="lede">
          Epzero's C++ engine runs in this tab as WebAssembly. Search by description, photo or camera, then step through exactly how
          the graph found the answer.
        </p>
      {/if}
    </header>

    <PhotoSource onmessage={(text) => say(text, 5000)} />

    <SearchBox
      onsearch={search}
      onsurprise={surprise}
      disabled={engine.status !== "ready" || engine.rebuilding}
      {initialText}
      {blendBase}
      {initialBlend}
    />

    {#if engine.source !== "mine"}
      <CategoryFilter onchange={() => outcome && search(outcome.query)} disabled={engine.status !== "ready" || engine.rebuilding} />
    {/if}

    <Sharpening />

    {#if searchError}<p class="search-error" role="alert">Search failed: {searchError}</p>{/if}

    {#if outcome}
      <SearchStats {outcome} {encodeMs} />
      <ResultsGrid {outcome} onpick={searchId} />
    {/if}
    <RecallGuard
      {outcome}
      {ef}
      onef={(next) => {
        ef = next;
        if (outcome) search(outcome.query, { live: true });
      }}
      onburst={async (count) => {
        for (let i = 0; i < count; i++) await search({ kind: "id", id: engine.randomId() }, { live: true });
      }}
    />
  </aside>

  <div class="knobs card" class:away={panelHidden} inert={panelHidden} role="group" aria-label="Search settings">
    <label class="ef">
      <span><span class="wide-only">Beam width&nbsp;</span><code>ef</code></span>
      <input type="range" min="16" max={EF_MAX} step="16" value={ef} oninput={(e) => onEf(Number((e.currentTarget as HTMLInputElement).value))} />
      <output class="num">{ef}</output>
    </label>
    <div class="buttons">
      <button
        class="btn btn-ghost icon-only"
        onclick={share}
        disabled={!outcome || engine.source === "mine"}
        aria-label="Share this search"
        title={engine.source === "mine" ? "Searches of your own photos can't be shared: they never leave this device" : "Share this search"}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15V3M7 8l5-5 5 5M5 13v6a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-6" /></svg>
      </button>
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

  {#if sceneController.unavailable}
    <!-- Without WebGL there's no 3D view; search, results and the race all still work. -->
    <p class="no-3d card" role="note">3D view unavailable: WebGL is turned off in this browser. Search works as normal.</p>
  {:else}
    <div class="view-anchor" class:away={panelHidden} inert={panelHidden}>
      <ViewControls />
    </div>
  {/if}

  {#if tuneOpen}
    <div class="tune-anchor">
      <TunePanel {ef} onclose={() => (tuneOpen = false)} onrebuilt={() => outcome && search(outcome.query)} />
    </div>
  {/if}

  {#if outcome}
    <!-- An exact scan of a filter's few photos has no route through the graph to navigate. -->
    {#if !panelHidden && outcome.trace.strategy !== "exact"}<NavHud {outcome} />{/if}
    <div class="dock" class:away={panelHidden} inert={panelHidden}>
      <Timeline {outcome} />
    </div>
  {/if}

  {#if engine.status !== "ready"}
    <div class="loading card" role="status" aria-live="polite">
      {#if engine.status === "error"}
        <strong>{engine.unsupported ? "This browser is too old for the engine" : "Something went wrong"}</strong>
        <p>{engine.error}</p>
        {#if !engine.unsupported}<button class="btn btn-primary retry" onclick={() => engine.retry()}>Try again</button>{/if}
      {:else}
        <strong>{engine.stage || "Starting the engine"}</strong>
        <div class="meter"><i style:width="{(engine.progress * 100).toFixed(1)}%"></i></div>
        <p>
          {engine.source === "mine"
            ? "Indexing your photos on every core of this device, with the same C++ engine compiled to WebAssembly."
            : "C++ compiled to WebAssembly. The graph arrives prebuilt; rebuild it on your own cores any time from the index settings."}
        </p>
      {/if}
    </div>
  {/if}

  <!-- The details card replaces the hover preview, which would sit under it. -->
  {#if hover && !selected}
    <div class="hovercard card" style:left="{Math.min(Math.max(hover.x, 130), innerWidth - 130)}px" style:top="{Math.max(hover.y, 220)}px">
      <Thumb id={hover.id} size="n" alt={engine.caption(hover.id)} />
      <div>
        <strong>{engine.caption(hover.id)}</strong>
        <span>{engine.source === "mine" ? "Your photo" : `© ${engine.author(hover.id)} · CC BY 2.0`} · click to search · shift-click for details</span>
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
        <span>
          {engine.source === "mine" ? "Your photo" : `© ${engine.author(selected.id)} · CC BY 2.0 · photo #${selected.id}`}
          {#if engine.original(selected.id)}· <a href={engine.original(selected.id)} target="_blank" rel="noopener noreferrer">original ↗</a>{/if}
        </span>
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
      {#if outcome}
        <WhyNot id={selected.id} {outcome} onwiden={(wider) => {
            ef = Math.min(wider, EF_MAX);
            if (outcome) search(outcome.query);
          }} />
      {/if}
    </div>
  {/if}

  {#if toast}<div class="toast card" role="status">{toast}</div>{/if}

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
    background: var(--overlay-fill);
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
    background: var(--overlay-fill);
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
  }
  .panel,
  .knobs,
  .dock,
  .view-anchor {
    transition:
      transform 0.4s var(--ease),
      opacity 0.3s;
  }
  /* Free look: the settings bar, the debugger and the view controls slide off with the panel. */
  .knobs.away,
  .dock.away,
  .view-anchor.away {
    opacity: 0;
    pointer-events: none !important;
  }
  .view-anchor.away {
    transform: translateX(calc(100% + 32px));
  }
  .knobs.away {
    transform: translateY(-24px);
  }
  .dock.away {
    transform: translateY(calc(100% + 32px));
  }
  .panel header {
    position: relative;
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
    background: var(--overlay-fill);
    transform: translate(-50%, -50%);
  }
  .no-3d {
    position: fixed;
    top: calc(var(--nav-h) + 90px);
    left: calc(424px + (100% - 424px) / 2);
    max-width: min(420px, calc(100% - 32px));
    margin: 0;
    padding: 12px 16px;
    background: var(--surface-solid);
    color: var(--text-2);
    font-size: 13.5px;
    transform: translateX(-50%);
  }
  @media (max-width: 900px) {
    .no-3d {
      left: 50%;
    }
  }
  .toast {
    position: fixed;
    z-index: 40;
    left: 50%;
    bottom: calc(96px + env(safe-area-inset-bottom, 0px));
    padding: 10px 16px;
    background: var(--surface-solid);
    font-size: 13.5px;
    transform: translateX(-50%);
    animation: toast-in 0.3s var(--ease);
  }
  @keyframes toast-in {
    from {
      opacity: 0;
      transform: translate(-50%, 8px);
    }
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
  /* Small laptops: keep the tune panel beside the search panel rather than over it. */
  @media (min-width: 901px) and (max-width: 1100px) {
    .tune-anchor {
      left: 448px;
    }
    .tune-anchor > :global(*) {
      width: 100%;
    }
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
      /* A fixed height (the first results always follow): the sheet doesn't grow under the viewer's
         thumb when they arrive, and the view controls docked above it stay put. */
      height: 52dvh;
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
    /* Just above the sheet, whose height is fixed: placed by CSS from the first frame, rather than
       jumping up once a script has measured the sheet. */
    .view-anchor {
      right: 12px;
      bottom: calc(52dvh + 24px + env(safe-area-inset-bottom, 0px));
      transition: bottom 0.35s var(--ease);
    }

    .loading {
      left: 50%;
      top: 35%;
    }
  }
</style>
