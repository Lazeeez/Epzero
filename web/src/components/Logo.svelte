<script lang="ts">
  let { size = 24 }: { size?: number } = $props();
</script>

<!-- The name in one glyph: HNSW's layers, the entry point ("ep", the ring on the top layer) and the
     search's hops down to its answer on layer zero (the dot). -->
<svg class="logo" width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
  <defs>
    <linearGradient id="epzero-logo" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="var(--accent-strong)" />
      <stop offset="1" stop-color="var(--accent-2)" />
    </linearGradient>
  </defs>
  <path d="M4.5 9.5 16 4l11.5 5.5L16 15z" fill="none" stroke="url(#epzero-logo)" stroke-width="1.5" stroke-linejoin="round" opacity="0.5" />
  <path d="M4.5 16 16 10.5 27.5 16 16 21.5z" fill="none" stroke="url(#epzero-logo)" stroke-width="1.5" stroke-linejoin="round" opacity="0.75" />
  <path d="M4.5 22.5 16 17l11.5 5.5L16 28z" fill="none" stroke="url(#epzero-logo)" stroke-width="1.7" stroke-linejoin="round" />
  <path d="M19.4 8.9 13.6 15.4 17.2 22.3" fill="none" stroke="var(--result)" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" />
  <circle cx="19.4" cy="8.9" r="2" fill="var(--page)" stroke="var(--result)" stroke-width="1.5" />
  <circle cx="17.2" cy="22.6" r="2.4" fill="var(--result)" />
</svg>

<style>
  .logo {
    flex: none;
  }
</style>
