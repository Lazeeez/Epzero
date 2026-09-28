<script lang="ts">
  let { size = 24 }: { size?: number } = $props();
</script>

<!-- Three stacked layers with a query path dropping through them: HNSW in one glyph. -->
<svg class="logo" width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
  <defs>
    <linearGradient id="oryx-logo" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="var(--accent-strong)" />
      <stop offset="1" stop-color="var(--accent-2)" />
    </linearGradient>
  </defs>
  <path d="M5 9.5 16 4l11 5.5L16 15z" fill="none" stroke="url(#oryx-logo)" stroke-width="1.6" stroke-linejoin="round" opacity="0.55" />
  <path d="M5 16 16 10.5 27 16 16 21.5z" fill="none" stroke="url(#oryx-logo)" stroke-width="1.6" stroke-linejoin="round" opacity="0.8" />
  <path d="M5 22.5 16 17l11 5.5L16 28z" fill="none" stroke="url(#oryx-logo)" stroke-width="1.8" stroke-linejoin="round" />
  <path d="M16 6.5v18.5" stroke="var(--result)" stroke-width="1.8" stroke-linecap="round" />
  <circle cx="16" cy="25" r="2.4" fill="var(--result)" />
</svg>

<style>
  .logo {
    flex: none;
  }
</style>
