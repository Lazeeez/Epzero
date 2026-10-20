<script lang="ts">
  import { untrack } from "svelte";
  import type { SearchOutcome } from "../../lib/engine/engine.svelte";
  import { agreement } from "../../lib/engine/trace-analysis";
  import { decide, wilson } from "../../lib/recall-guard";

  // "Hold recall >= target": every search here is also answered exactly, so each one is a recall
  // sample. While the guard is on, it moves the beam width (ef) to the smallest value whose
  // measured recall meets the target, the way the engine's RecallGuard does in production.
  let {
    outcome,
    ef,
    onef,
    onburst,
  }: { outcome: SearchOutcome | null; ef: number; onef: (ef: number) => void; onburst: (count: number) => Promise<void> } = $props();
  let bursting = $state(false);
  async function burst() {
    bursting = true;
    try {
      await onburst(30);
    } finally {
      bursting = false;
    }
  }

  const TARGETS = [0.9, 0.95, 0.99];
  const WINDOW = 30;
  let on = $state(false);
  let target = $state(0.95);
  let recent = $state.raw<number[]>([]); // per-query recall at the current ef, oldest first
  let history = $state.raw<number[]>([]); // every sample, for the sparkline
  let note = $state("");
  let seen: SearchOutcome | null = null;

  const estimate = $derived(wilson(recent));

  $effect(() => {
    const current = outcome;
    if (!current || current === seen) return;
    seen = current;
    const k = current.trace.results.length;
    if (!k) return;
    const recall = agreement(current.trace) / k;
    untrack(() => {
      history = [...history, recall].slice(-60);
      if (!on) return;
      if (current.ef !== ef) return; // a search from before the last change
      recent = [...recent, recall].slice(-WINDOW);
      const next = decide(recent, ef, { target, minSamples: 8, margin: 0.02, minEf: 16, maxEf: 512 });
      if (next) {
        note = `${next.reason}: ef ${ef} → ${next.ef}`;
        recent = [];
        onef(next.ef);
      }
    });
  });

  function toggle() {
    on = !on;
    recent = [];
    note = on ? `Holding recall ≥ ${target}. Every search counts as a sample.` : "";
  }
  function setTarget(t: number) {
    target = t;
    recent = [];
    if (on) note = `Holding recall ≥ ${t}.`;
  }

  // Sparkline: per-query recall from 0.5 (bottom) to 1 (top), with the target as a dashed line.
  const W = 220;
  const H = 40;
  const y = (r: number) => H - Math.max(0, Math.min(1, (r - 0.5) / 0.5)) * H;
  const points = $derived(history.map((r, i) => `${((i + 0.5) / 60) * W},${y(r)}`).join(" "));
</script>

<section class="guard" aria-labelledby="guard-title">
  <header>
    <div>
      <h2 id="guard-title">Recall guard</h2>
      <p>Holds recall at a target by moving ef, as the engine does in production.</p>
    </div>
    <button class="switch" role="switch" aria-checked={on} aria-label="Hold recall at a target" onclick={toggle}>
      <span></span>
    </button>
  </header>

  <div class="row">
    <span class="label">Target</span>
    <div class="targets" role="radiogroup" aria-label="Recall target">
      {#each TARGETS as t (t)}
        <button role="radio" aria-checked={target === t} class:on={target === t} onclick={() => setTarget(t)}>≥ {t}</button>
      {/each}
    </div>
  </div>

  <div class="meter">
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Recall of the last {history.length} searches">
      <line class="target" x1="0" x2={W} y1={y(target)} y2={y(target)} />
      {#if history.length > 1}<polyline points={points} />{/if}
    </svg>
    <div class="numbers num">
      {#if on && estimate.samples}
        <strong class:low={estimate.recall < target}>{estimate.recall.toFixed(3)}</strong>
        <span>{estimate.lower.toFixed(2)}–{estimate.upper.toFixed(2)} · {estimate.samples} searches at ef {ef}</span>
      {:else if history.length}
        <strong>{(history.reduce((a, b) => a + b, 0) / history.length).toFixed(3)}</strong>
        <span>mean recall, last {history.length} searches</span>
      {:else}
        <span>Search to measure recall.</span>
      {/if}
    </div>
  </div>
  {#if note}<p class="note" aria-live="polite">{note}</p>{/if}
  <button class="btn btn-ghost burst" onclick={burst} disabled={bursting || !outcome}>
    {bursting ? "Searching…" : "Run 30 random searches"}
  </button>
</section>

<style>
  .guard {
    display: grid;
    gap: 10px;
    padding: 14px 16px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--surface);
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    gap: 12px;
  }
  h2 {
    margin: 0;
    font-size: 14px;
    letter-spacing: -0.01em;
  }
  header p {
    margin: 2px 0 0;
    color: var(--muted);
    font-size: 12px;
  }
  .switch {
    flex: none;
    position: relative;
    width: 40px;
    height: 24px;
    border: 1px solid var(--border-strong);
    border-radius: 99px;
    background: var(--surface-3);
    cursor: pointer;
    transition: background 0.2s, border-color 0.2s;
  }
  .switch span {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--text-2);
    transition: transform 0.2s var(--ease), background 0.2s;
  }
  .switch[aria-checked="true"] {
    border-color: var(--accent);
    background: var(--accent-wash);
  }
  .switch[aria-checked="true"] span {
    background: var(--accent-strong);
    transform: translateX(16px);
  }
  .row {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .label {
    color: var(--muted);
    font: 600 10.5px var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .targets {
    display: flex;
    gap: 4px;
  }
  .targets button {
    padding: 3px 9px;
    border: 1px solid var(--border);
    border-radius: 99px;
    background: none;
    color: var(--text-2);
    font: 12px var(--mono);
    cursor: pointer;
  }
  .targets button.on {
    border-color: var(--accent);
    background: var(--accent-wash);
    color: var(--accent-strong);
  }
  .meter {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 12px;
  }
  svg {
    width: 100%;
    height: 40px;
    overflow: visible;
  }
  polyline {
    fill: none;
    stroke: var(--accent);
    stroke-width: 1.5;
    stroke-linejoin: round;
  }
  .target {
    stroke: var(--warn);
    stroke-dasharray: 3 3;
    stroke-width: 1;
  }
  .numbers {
    display: grid;
    justify-items: end;
    text-align: right;
  }
  .numbers strong {
    color: var(--good);
    font-size: 18px;
  }
  .numbers strong.low {
    color: var(--warn);
  }
  .numbers span {
    color: var(--muted);
    font-size: 11px;
  }
  .burst {
    justify-self: start;
    height: 32px;
    padding: 0 12px;
    font-size: 12.5px;
  }
  .note {
    margin: 0;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    color: var(--text-2);
    font-size: 12px;
  }
</style>
