<script lang="ts">
  import type { SearchOutcome } from "../../lib/engine/engine.svelte";
  import { analyze } from "../../lib/engine/trace-analysis";
  import { sceneController } from "../../lib/scene/scene.svelte";
  import Thumb from "../Thumb.svelte";

  let { outcome }: { outcome: SearchOutcome } = $props();

  const info = $derived(analyze(outcome.trace, outcome.ef));
  const total = $derived(info.length);
  const playhead = $derived(sceneController.playhead);
  const current = $derived(playhead.event >= 0 ? info[playhead.event] : null);
  // Where the greedy descent hands over to the beam search, as a fraction of the track.
  const split = $derived.by(() => {
    const first = info.findIndex((e) => e.phase === "beam");
    return first < 0 ? 1 : first / Math.max(total, 1);
  });

  let speed = $state(1);
  const speeds = [0.25, 0.5, 1, 2, 4];

  function toggle() {
    const scene = sceneController.scene;
    if (!scene) return;
    if (playhead.playing) scene.pause();
    else scene.resume();
  }

  function step(delta: number) {
    const base = playhead.event < 0 && delta > 0 ? -1 : playhead.event;
    sceneController.scene?.seekEvent(Math.max(-1, Math.min(total - 1, base + delta)));
  }

  function setSpeed(value: number) {
    speed = value;
    sceneController.scene?.setSpeed(value);
  }

  // Space: play/pause. Arrows: step (shift = 10 steps). Left to the browser while a control has
  // focus (Space presses a focused button, arrows move a slider) or a dialog is open.
  $effect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (event.metaKey || event.ctrlKey || event.altKey || document.querySelector("dialog[open]")) return;
      if (target.closest("input, textarea, select, [contenteditable], [role=\"tablist\"]")) return;
      if (event.key === " " && target.closest("button, a")) return;
      if (event.key === " ") {
        event.preventDefault();
        toggle();
      } else if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        step((event.key === "ArrowRight" ? 1 : -1) * (event.shiftKey ? 10 : 1));
      }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  });

  const fmt = (d: number) => (Number.isFinite(d) ? d.toFixed(3) : "∞");
</script>

<section class="timeline card" aria-label="Search debugger">
  <div class="controls">
    <button class="ctl" onclick={() => step(-1)} aria-label="Previous step" title="Previous step (←)">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14M18 6l-9 6 9 6z" /></svg>
    </button>
    <button class="ctl play" onclick={toggle} aria-label={playhead.playing ? "Pause" : "Play"} title="Play / pause (space)">
      {#if playhead.playing}
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14" /></svg>
      {:else}
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5l12 7-12 7z" /></svg>
      {/if}
    </button>
    <button class="ctl" onclick={() => step(1)} aria-label="Next step" title="Next step (→)">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 5v14M6 6l9 6-9 6z" /></svg>
    </button>

    <div class="track">
      <div class="phases" aria-hidden="true">
        <span class="descent" style:width="{split * 100}%"></span>
        <span class="beam"></span>
      </div>
      <input
        type="range"
        min="-1"
        max={total - 1}
        value={playhead.event}
        oninput={(e) => sceneController.scene?.seekEvent(Number((e.currentTarget as HTMLInputElement).value))}
        aria-label="Search step"
        aria-valuetext="Step {playhead.event + 1} of {total}"
      />
    </div>

    <span class="count num">{Math.max(playhead.event + 1, 0).toLocaleString()} / {total.toLocaleString()}</span>

    <div class="speeds" role="group" aria-label="Replay speed">
      {#each speeds as s (s)}
        <button class:active={speed === s} onclick={() => setSpeed(s)}>{s}×</button>
      {/each}
    </div>
  </div>

  <div class="inspector" aria-live="polite">
    {#if current}
      <span class="phase" class:beam={current.phase === "beam"}>
        {current.phase === "beam" ? "Layer 0 · beam search" : `Layer ${current.layer} · greedy descent`}
      </span>
      <span class="hop">
        <span class="node"><Thumb id={current.from} size="q" /><span class="num">#{current.from}</span></span>
        <span class="arrow" aria-hidden="true">→</span>
        <span class="node"><Thumb id={current.to} size="q" /><span class="num">#{current.to}</span></span>
      </span>
      <span class="explain">
        Expanding <span class="num">#{current.from}</span>, scored neighbor <span class="num">#{current.to}</span> at distance
        <strong class="num">{fmt(current.distance)}</strong>:
        {#if current.accepted}
          <strong class="kept">{current.phase === "beam" ? "kept in the beam" : "closer, move here"}</strong>
        {:else}
          <strong class="dropped">{current.phase === "beam" ? "not better than the beam's worst" : "not closer"}</strong>
        {/if}
        {#if current.phase === "beam"}
          (beam <span class="num">{current.beamSize}/{outcome.ef}</span>, bound <span class="num">{fmt(current.bound)}</span>)
        {:else}
          (best so far <span class="num">{fmt(current.bound)}</span>)
        {/if}
      </span>
    {:else}
      <span class="explain muted">
        Step through the search: <kbd>space</kbd> play/pause, <kbd>←</kbd><kbd>→</kbd> step, <kbd>shift</kbd> for ×10.
      </span>
    {/if}
  </div>
</section>

<style>
  .timeline {
    display: grid;
    gap: 10px;
    padding: 12px 14px;
    background: var(--glass);
    container-type: inline-size; /* the dock's width varies with the window and the panel */
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .ctl {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
  }
  .ctl:hover {
    border-color: var(--border-strong);
  }
  .ctl.play {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--text);
    color: var(--page);
  }
  .ctl svg {
    width: 15px;
    height: 15px;
    fill: currentColor;
    stroke: currentColor;
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .track {
    position: relative;
    flex: 1;
    min-width: 80px;
    height: 34px;
  }
  .phases {
    position: absolute;
    inset: 14px 0;
    display: flex;
    border-radius: 99px;
    overflow: hidden;
  }
  .phases .descent {
    background: color-mix(in srgb, var(--result) 55%, transparent);
  }
  .phases .beam {
    flex: 1;
    background: color-mix(in srgb, var(--accent) 45%, transparent);
  }
  input[type="range"] {
    position: absolute;
    inset: 0;
    width: 100%;
    margin: 0;
    appearance: none;
    background: transparent;
    cursor: pointer;
  }
  input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    width: 14px;
    height: 22px;
    border-radius: 5px;
    background: var(--text);
    box-shadow: 0 0 0 3px var(--page);
  }
  input[type="range"]::-moz-range-thumb {
    width: 14px;
    height: 22px;
    border: 0;
    border-radius: 5px;
    background: var(--text);
    box-shadow: 0 0 0 3px var(--page);
  }
  .count {
    min-width: 92px;
    color: var(--text-2);
    font: 12px var(--mono);
    text-align: right;
  }
  .speeds {
    display: flex;
    gap: 2px;
    padding: 2px;
    border: 1px solid var(--border);
    border-radius: 9px;
  }
  .speeds button {
    padding: 3px 7px;
    border: 0;
    border-radius: 7px;
    background: transparent;
    color: var(--muted);
    font: 11.5px var(--mono);
    cursor: pointer;
  }
  .speeds button.active {
    background: var(--surface-3);
    color: var(--text);
  }
  .inspector {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 12px;
    min-height: 36px;
    font-size: 13px;
    color: var(--text-2);
  }
  .phase {
    flex: none;
    padding: 3px 9px;
    border-radius: 99px;
    background: color-mix(in srgb, var(--result) 18%, transparent);
    color: var(--result);
    font: 600 11px var(--mono);
  }
  .phase.beam {
    background: var(--accent-wash);
    color: var(--accent-strong);
  }
  .hop {
    display: flex;
    flex: none;
    align-items: center;
    gap: 6px;
  }
  .node {
    display: flex;
    align-items: center;
    gap: 5px;
    font: 11px var(--mono);
  }
  .node :global(.thumb) {
    width: 30px;
    height: 30px;
    border-radius: 7px;
    overflow: hidden;
  }
  .node :global(img) {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .arrow {
    color: var(--muted);
  }
  .explain {
    flex: 1 1 280px;
    min-width: 0;
    line-height: 1.45;
    overflow-wrap: anywhere;
  }
  /* A narrow dock (small laptops, tablets): drop the speed picker, then the thumbnails, so the
     controls never spill past the card. */
  @container (max-width: 600px) {
    .speeds {
      display: none;
    }
  }
  @container (max-width: 470px) {
    .hop {
      display: none;
    }
    .count {
      min-width: 0;
    }
  }
  .explain strong {
    color: var(--text);
  }
  .kept {
    color: var(--good) !important;
  }
  .dropped {
    color: var(--muted) !important;
  }
  .muted {
    color: var(--muted);
  }
  @media (max-width: 760px) {
    .speeds,
    .hop,
    .count {
      display: none;
    }
    .inspector {
      font-size: 12px;
    }
  }
</style>
