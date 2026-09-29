<script lang="ts">
  import Footer from "./components/Footer.svelte";
  import Nav from "./components/Nav.svelte";
  import { engine } from "./lib/engine/engine.svelte";
  import { router } from "./lib/router.svelte";
  import { sceneController, type SceneMode } from "./lib/scene/scene.svelte";
  import Benchmarks from "./pages/Benchmarks.svelte";
  import Landing from "./pages/Landing.svelte";
  import Play from "./pages/Play.svelte";
  import Try from "./pages/Try.svelte";

  let stage: HTMLDivElement;

  const modes: Record<string, SceneMode> = { home: "hero", try: "explore", play: "hidden", benchmarks: "hidden" };

  // Pages that show the graph start the engine and the scene; the benchmarks page needs neither.
  $effect(() => {
    const mode = modes[router.route];
    sceneController.setMode(mode);
    if (mode !== "hidden") {
      engine.start();
      sceneController.attach(stage);
    }
    if (router.route === "play") engine.start(); // the game needs the index, not the 3D scene
  });

  // Nodes appear in the scene as the index is built.
  $effect(() => {
    void engine.layout;
    void engine.built;
    void engine.graph;
    void sceneController.ready;
    sceneController.sync();
  });

  const titles = {
    home: "Oryx · A vector database, running in your browser",
    try: "Try it · Oryx",
    play: "Out-navigate Oryx · Oryx",
    benchmarks: "Benchmarks · Oryx",
  };
  $effect(() => {
    document.title = titles[router.route];
  });
</script>

<div
  class="stage"
  class:hidden={modes[router.route] === "hidden"}
  class:hero={router.route === "home"}
  bind:this={stage}
  aria-hidden={router.route !== "try"}
></div>

<Nav />

<main id="main" tabindex="-1" class:passthrough={router.route === "try"}>
  {#if router.route === "home"}
    <Landing />
  {:else if router.route === "try"}
    <Try />
  {:else if router.route === "play"}
    <Play />
  {:else}
    <Benchmarks />
  {/if}
</main>

{#if router.route !== "try"}
  <Footer />
{/if}

<style>
  .stage {
    position: fixed;
    inset: 0;
    z-index: 0;
    opacity: 1;
    transition: opacity 0.6s var(--ease);
  }
  .stage.hidden {
    opacity: 0;
    visibility: hidden;
  }
  /* On the landing page the graph is a backdrop that fades as the story scrolls past the hero. */
  .stage.hero {
    opacity: var(--hero-scene-opacity, 1);
    transition: none;
  }
  .stage :global(canvas) {
    display: block;
    width: 100% !important;
    height: 100% !important;
  }
  main {
    position: relative;
    z-index: 1;
    outline: none; /* focused programmatically after navigation, not by the user */
  }
  /* The playground's 3D graph sits behind the page: let pointer and touch input through to it
     everywhere except the page's own panels (which opt back in). */
  main.passthrough {
    pointer-events: none;
  }
</style>
