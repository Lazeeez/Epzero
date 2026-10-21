<script lang="ts">
  import Ambient from "./components/Ambient.svelte";
  import Footer from "./components/Footer.svelte";
  import Nav from "./components/Nav.svelte";
  import { engine } from "./lib/engine/engine.svelte";
  import { warmUpEncoders } from "./lib/engine/encoders.svelte";
  import { router } from "./lib/router.svelte";
  import { sceneController, type SceneMode } from "./lib/scene/scene.svelte";
  import Benchmarks from "./pages/Benchmarks.svelte";
  import Landing from "./pages/Landing.svelte";
  import NotFound from "./pages/NotFound.svelte";
  import Play from "./pages/Play.svelte";
  import Try from "./pages/Try.svelte";

  let stage: HTMLDivElement;

  const modes: Record<string, SceneMode> = { home: "hero", try: "explore", play: "hidden", cluster: "hidden", benchmarks: "hidden", notfound: "hidden" };
  // The cluster page carries its own (Go) engine: load its code only when it is opened.
  const loadCluster = () => import("./pages/Cluster.svelte");

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

  // Once the index is ready, warm the CLIP models in the background (not on the benchmarks page,
  // which doesn't use them): the first text or photo search then answers without a download.
  let warmed = false;
  $effect(() => {
    if (warmed || engine.status !== "ready" || router.route === "benchmarks" || router.route === "cluster") return;
    warmed = true;
    warmUpEncoders();
  });

  // Nodes appear in the scene as the index is built.
  $effect(() => {
    void engine.layout;
    void engine.built;
    void engine.graph;
    void engine.filter;
    void sceneController.ready;
    sceneController.sync();
  });

  const titles = {
    home: "Epzero · A vector database, running in your browser",
    try: "Try it · Epzero",
    play: "Out-navigate Epzero · Epzero",
    cluster: "Cluster · Epzero",
    benchmarks: "Benchmarks · Epzero",
    notfound: "Page not found · Epzero",
  };
  // Per-page title and canonical URL (search engines index each page on its own address; a
  // not-found page points back to the home page).
  $effect(() => {
    document.title = titles[router.route];
    const canonical = document.querySelector<HTMLLinkElement>('link[rel="canonical"]');
    if (canonical) canonical.href = new URL(router.href(router.route === "notfound" ? "home" : router.route), "https://epzero.dev").href;
  });
</script>

<div
  class="stage"
  class:hidden={modes[router.route] === "hidden"}
  class:hero={router.route === "home"}
  bind:this={stage}
  aria-hidden={router.route !== "try"}
></div>

{#if modes[router.route] === "hidden"}<Ambient />{/if}

<Nav />

<main id="main" tabindex="-1" class:passthrough={router.route === "try"}>
  {#if router.route === "home"}
    <Landing />
  {:else if router.route === "try"}
    <Try />
  {:else if router.route === "play"}
    <Play />
  {:else if router.route === "cluster"}
    {#await loadCluster()}
      <div class="page-pending"></div>
    {:then { default: Cluster }}
      <Cluster />
    {/await}
  {:else if router.route === "benchmarks"}
    <Benchmarks />
  {:else}
    <NotFound />
  {/if}
</main>

{#if router.route !== "try"}
  <Footer />
{/if}

<style>
  /* Holds the footer below the fold while a lazily loaded page arrives. */
  .page-pending {
    min-height: 100vh;
  }
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
