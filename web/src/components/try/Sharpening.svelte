<script lang="ts">
  // While the rest of the vectors stream in, searches use only their leading dimensions (see
  // collection.worker.ts): this says how many, how close that gets to the full vectors, and how far
  // the download has come. Once the vectors are complete it says so briefly and goes away.
  import { fade } from "svelte/transition";
  import { engine } from "../../lib/engine/engine.svelte";

  let done = $state(false);
  let seen = false;
  $effect(() => {
    if (engine.sharpening) seen = true;
    else if (seen && engine.status === "ready") {
      seen = false;
      done = true;
      const timer = setTimeout(() => (done = false), 3200);
      return () => clearTimeout(timer);
    }
  });
</script>

{#if engine.sharpening}
  <div class="sharpening" role="status" transition:fade={{ duration: 200 }}>
    <span class="label">
      <i class="pulse" aria-hidden="true"></i>
      <span>Sharpening · <b class="num">{engine.dims}</b> of <span class="num">{engine.dim}</span> dimensions</span>
      <span class="num pct">{Math.round(engine.streamed * 100)}%</span>
    </span>
    <span class="meter" aria-hidden="true"><i style:width="{(engine.streamed * 100).toFixed(1)}%"></i></span>
    <span class="note">
      Searching already, on the vectors' strongest {engine.dims} axes, where an exact search finds
      <span class="num">{Math.round(engine.stageRecall * 100)}%</span> of a photo's true top 10. Results update as the rest arrives.
    </span>
  </div>
{:else if done}
  <div class="sharpening done" role="status" transition:fade={{ duration: 300 }}>
    <span class="label"><i aria-hidden="true">✓</i><span>Full precision · all {engine.dim} dimensions</span></span>
  </div>
{/if}

<style>
  .sharpening {
    display: grid;
    gap: 6px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
    font-size: 12.5px;
  }
  .label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-2);
  }
  .label b {
    color: var(--text);
    font-weight: 600;
  }
  .pct {
    margin-left: auto;
    color: var(--muted);
    font-family: var(--mono);
  }
  .pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.4s ease-in-out infinite;
  }
  .done i {
    color: var(--accent-strong);
    font-style: normal;
    font-weight: 700;
  }
  .meter {
    height: 3px;
    border-radius: 3px;
    background: var(--grid);
    overflow: hidden;
  }
  .meter i {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 0.25s linear;
  }
  .note {
    color: var(--muted);
    line-height: 1.45;
  }
  @keyframes pulse {
    50% {
      opacity: 0.35;
      transform: scale(0.8);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .pulse {
      animation: none;
    }
  }
</style>
