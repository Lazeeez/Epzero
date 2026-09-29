<script lang="ts">
  import { engine } from "../../lib/engine/engine.svelte";
  import { sceneController } from "../../lib/scene/scene.svelte";

  let { panelHidden, ontogglepanel }: { panelHidden: boolean; ontogglepanel: () => void } = $props();

  const layers = $derived(engine.graph ? engine.graph.maxLevel + 1 : 0);
  const coarse = matchMedia("(pointer: coarse)").matches;
  const HINT_KEY = "oryx-view-hint";

  let hint = $state(false);
  try {
    hint = !localStorage.getItem(HINT_KEY);
  } catch {
    hint = true;
  }
  function dismissHint() {
    hint = false;
    try {
      localStorage.setItem(HINT_KEY, "1");
    } catch {
      /* storage blocked: the hint just shows again next visit */
    }
  }
  // The hint fades on its own after a while, or as soon as the viewer moves the camera.
  $effect(() => {
    if (!hint) return;
    const timer = setTimeout(dismissHint, 9000);
    return () => clearTimeout(timer);
  });
  $effect(() => {
    if (hint && !sceneController.autoRotate) dismissHint();
  });

  function cycleLayer(delta: number) {
    if (!layers) return;
    // -1 (all), 0, 1, ..., top, then back to all.
    const next = ((sceneController.layerFocus + 1 + delta + layers + 1) % (layers + 1)) - 1;
    sceneController.setLayerFocus(next);
  }

  // Keyboard: + / - zoom, 0 resets the view, R toggles rotation, L cycles layers.
  $effect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || document.querySelector("dialog[open]")) return;
      if ((event.target as HTMLElement).closest("input, textarea, select, [contenteditable]")) return;
      const scene = sceneController.scene;
      if (!scene) return;
      if (event.key === "+" || event.key === "=") scene.zoom(0.7);
      else if (event.key === "-" || event.key === "_") scene.zoom(1.4);
      else if (event.key === "0") scene.overview();
      else if (event.key === "r" || event.key === "R") sceneController.setAutoRotate(!sceneController.autoRotate);
      else if (event.key === "l" || event.key === "L") cycleLayer(event.shiftKey ? -1 : 1);
      else return;
      event.preventDefault();
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  });
</script>

<div class="view card" role="toolbar" aria-label="View controls" aria-orientation="vertical">
  <button onclick={() => sceneController.scene?.zoom(0.7)} aria-label="Zoom in" title="Zoom in (+)">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg>
  </button>
  <button onclick={() => sceneController.scene?.zoom(1.4)} aria-label="Zoom out" title="Zoom out (−)">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14" /></svg>
  </button>
  <span class="sep" aria-hidden="true"></span>
  <button
    class:on={sceneController.autoRotate}
    onclick={() => sceneController.setAutoRotate(!sceneController.autoRotate)}
    aria-pressed={sceneController.autoRotate}
    aria-label="Auto-rotate"
    title="Auto-rotate (R)"
  >
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 12a8 8 0 1 1-2.3-5.7M20 4v5h-5" /></svg>
  </button>
  <button
    class="layer"
    class:on={sceneController.layerFocus >= 0}
    onclick={() => cycleLayer(1)}
    aria-label={sceneController.layerFocus < 0 ? "Showing all layers; isolate a layer" : `Showing layer ${sceneController.layerFocus}; next layer`}
    title="Isolate a layer (L)"
    disabled={!layers}
  >
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 4 8 4-8 4-8-4zM4 12l8 4 8-4M4 16l8 4 8-4" /></svg>
    <span class="num">{sceneController.layerFocus < 0 ? "all" : `L${sceneController.layerFocus}`}</span>
  </button>
  <button onclick={() => sceneController.scene?.overview()} aria-label="Reset the view" title="Reset the view (0)">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" /></svg>
  </button>
  <span class="sep" aria-hidden="true"></span>
  <button
    class:on={panelHidden}
    onclick={ontogglepanel}
    aria-pressed={panelHidden}
    aria-label={panelHidden ? "Show the search panel" : "Hide the search panel"}
    title={panelHidden ? "Show the search panel" : "Hide the panel to explore"}
  >
    <svg viewBox="0 0 24 24" aria-hidden="true">
      {#if panelHidden}<path d="M4 6h16v12H4zM9 6v12" />{:else}<path d="M4 6h16v12H4zM9 6v12M15 10l-2 2 2 2" />{/if}
    </svg>
  </button>
</div>

{#if hint}
  <button class="hint card" onclick={dismissHint} aria-label="Dismiss the controls hint">
    {#if coarse}
      <span><b>Drag</b> to orbit</span><span><b>Pinch</b> to zoom</span><span><b>Two fingers</b> to pan</span><span><b>Tap</b> a star to see its photo</span>
    {:else}
      <span><b>Drag</b> to orbit</span><span><b>Scroll</b> or pinch to zoom</span><span><b>Right-drag</b> to pan</span><span><b>Hover</b> a star to see it, click to search</span><span><b>Double-click</b> to fly in</span>
    {/if}
  </button>
{/if}

<style>
  .view {
    display: grid;
    gap: 2px;
    padding: 5px;
    background: var(--glass);
  }
  button {
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    padding: 0;
    border: 0;
    border-radius: 10px;
    background: transparent;
    color: var(--text-2);
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
    touch-action: manipulation;
  }
  button:hover:not(:disabled) {
    background: var(--surface-3);
    color: var(--text);
  }
  button.on {
    color: var(--accent-strong);
    background: var(--accent-wash);
  }
  button:disabled {
    opacity: 0.4;
    cursor: default;
  }
  svg {
    width: 18px;
    height: 18px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.9;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .layer {
    height: 46px;
    gap: 1px;
  }
  .layer svg {
    width: 16px;
    height: 16px;
  }
  .layer span {
    font: 600 9.5px var(--mono);
  }
  .sep {
    height: 1px;
    margin: 3px 6px;
    background: var(--border);
  }
  .hint {
    position: fixed;
    z-index: 6;
    top: calc(var(--nav-h) + 76px);
    left: 50%;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 4px 16px;
    width: max-content;
    max-width: min(640px, calc(100vw - 32px));
    height: auto;
    padding: 10px 16px;
    background: var(--glass);
    color: var(--text-2);
    font-size: 13px;
    transform: translateX(-50%);
    animation: hint-in 0.5s var(--ease);
  }
  .hint b {
    color: var(--text);
    font-weight: 600;
  }
  @keyframes hint-in {
    from {
      opacity: 0;
      transform: translate(-50%, -6px);
    }
  }
  @media (max-width: 900px) {
    .hint {
      top: calc(var(--nav-h) + 72px);
      font-size: 12.5px;
    }
    button {
      width: 40px;
      height: 40px;
    }
  }
</style>
