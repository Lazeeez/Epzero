<script lang="ts">
  import { engine, type SearchOutcome } from "../../lib/engine/engine.svelte";
  import type { Explanation } from "../../lib/engine/protocol";

  // For a photo picked in the 3D view: is it in the current search's results, and if not, why
  // not? The engine compares it with the exact answer and replays the search with wider beams.
  let { id, outcome, onwiden }: { id: number; outcome: SearchOutcome; onwiden: (ef: number) => void } = $props();

  const k = $derived(outcome.trace.results.length);
  const rank = $derived(outcome.trace.results.indexOf(id) + 1);
  let why = $state.raw<Explanation | null>(null);
  let failed = $state(false);

  $effect(() => {
    why = null;
    failed = false;
    if (rank > 0) return; // it is a result: nothing to explain
    let current = true;
    engine.explain(outcome.query, id, outcome.ef, k).then(
      (answer) => current && (why = answer),
      () => current && (failed = true),
    );
    return () => void (current = false);
  });

  const nth = (n: number) => `#${n.toLocaleString()}`;
</script>

<div class="why" aria-live="polite">
  {#if rank > 0}
    <p><strong class="good">Result {nth(rank)}</strong> of this search.</p>
  {:else if failed}
    <p>Couldn't check this photo against the search.</p>
  {:else if !why}
    <p class="muted">Checking why it isn't in the results…</p>
  {:else if why.cause === "not-near-enough"}
    <p>
      <strong>Not a miss.</strong> By exact distance it's {nth(why.exactRank)} for this query, and the search returns the top {k}:
      every result is genuinely closer.
    </p>
  {:else if why.cause === "search-missed"}
    <p>
      <strong class="warn">A miss.</strong> By exact distance it's {nth(why.exactRank)}, so it belongs in the top {k}.
      {why.scored
        ? `The walk reached it but kept closer-looking candidates at ef ${outcome.ef}.`
        : `The walk never reached it at ef ${outcome.ef}.`}
      {#if why.efNeeded}A wider beam finds it.{:else}Even ef 4,096 doesn't.{/if}
    </p>
    {#if why.efNeeded}
      <button class="btn btn-ghost widen" onclick={() => onwiden(why!.efNeeded)}>Search again at ef {why.efNeeded}</button>
    {/if}
  {:else if why.cause === "filtered-out"}
    <p><strong>Filtered out:</strong> it doesn't match the category filter.</p>
  {:else if why.cause === "not-in-index"}
    <p>This photo isn't in the index.</p>
  {:else}
    <p><strong class="good">Returned</strong> by this search.</p>
  {/if}
</div>

<style>
  .why {
    grid-column: 1 / -1;
    padding: 10px 12px 12px;
    border-top: 1px solid var(--border);
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--text-2);
  }
  p {
    margin: 0;
  }
  strong {
    color: var(--text);
  }
  .good {
    color: var(--good);
  }
  .warn {
    color: var(--warn);
  }
  .muted {
    color: var(--muted);
  }
  .widen {
    height: 32px;
    margin-top: 8px;
    padding: 0 12px;
    font-size: 12.5px;
  }
</style>
