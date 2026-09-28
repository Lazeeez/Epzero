<script lang="ts">
  import { engine } from "../lib/engine/engine.svelte";

  let { id, size = "n", alt = "" }: { id: number; size?: "q" | "n" | "z"; alt?: string } = $props();
  let loaded = $state(false);
  let failed = $state(false);
</script>

<!-- Flickr thumbnails, fetched with CORS (required under cross-origin isolation) and faded in. -->
<span class="thumb" class:loaded class:failed>
  <img
    src={engine.thumbnail(id, size)}
    {alt}
    crossorigin="anonymous"
    referrerpolicy="no-referrer"
    loading="lazy"
    decoding="async"
    onload={() => (loaded = true)}
    onerror={() => (failed = true)}
  />
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
  @keyframes shimmer {
    to {
      background-position: -150% 0;
    }
  }
</style>
