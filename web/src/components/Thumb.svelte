<script lang="ts">
  import { engine } from "../lib/engine/engine.svelte";

  let { id, size = "n", alt = "" }: { id: number; size?: "q" | "n" | "z"; alt?: string } = $props();
  let loaded = $state(false);
  let failed = $state(false);
  let attempt = $state(0);

  // A new photo in the same tile starts over.
  $effect(() => {
    void id;
    loaded = failed = false;
    attempt = 0;
  });

  // A failed download (a network blip, a busy photo host) is retried twice, a little later each
  // time, before the tile shows that the photo is unavailable.
  let timer = 0;
  function onerror() {
    if (attempt >= 2) {
      failed = true;
      return;
    }
    clearTimeout(timer);
    timer = window.setTimeout(() => attempt++, 1200 * (attempt + 1));
  }
  $effect(() => () => clearTimeout(timer));

  const src = $derived(engine.thumbnail(id, size) + (attempt ? `?retry=${attempt}` : ""));
</script>

<!-- Flickr thumbnails, fetched with CORS (required under cross-origin isolation) and faded in. -->
<span class="thumb" class:loaded class:failed title={failed ? "This photo couldn't be loaded" : undefined}>
  <img {src} {alt} crossorigin="anonymous" referrerpolicy="no-referrer" loading="lazy" decoding="async" onload={() => (loaded = true)} {onerror} />
  {#if failed}
    <svg class="missing" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 6h16v12H4zM4 15l4-4 4 4 3-3 5 5M3 3l18 18" />
    </svg>
  {/if}
</span>

<style>
  .thumb {
    display: block;
    position: relative;
    background: linear-gradient(110deg, var(--surface-2) 40%, var(--surface-3) 50%, var(--surface-2) 60%);
    background-size: 250% 100%;
    animation: shimmer 1.4s linear infinite;
  }
  .thumb.loaded,
  .thumb.failed {
    animation: none;
  }
  img {
    opacity: 0;
    transition: opacity 0.35s;
  }
  .loaded img {
    opacity: 1;
  }
  /* The photo couldn't be fetched: a crossed-out picture instead of an empty box. */
  .missing {
    position: absolute;
    inset: 0;
    width: min(28px, 50%);
    height: auto;
    margin: auto;
    fill: none;
    stroke: var(--muted);
    stroke-width: 1.5;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  @keyframes shimmer {
    to {
      background-position: -150% 0;
    }
  }
</style>
