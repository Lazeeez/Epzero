<script lang="ts">
  import { engine } from "../../lib/engine/engine.svelte";
  import { forgetKept, keptInfo, loadKept } from "../../lib/engine/own-photos.svelte";
  import OwnPhotos from "./OwnPhotos.svelte";

  // Which photos the playground searches: the demo set, or the visitor's own.
  let { onmessage }: { onmessage: (text: string) => void } = $props();

  let kept = $state<{ count: number; savedAt: number } | null>(null);
  let dialog = $state<null | "new" | "add">(null);
  let switching = $state(false);

  $effect(() => {
    keptInfo().then((info) => (kept = info));
  });

  const mine = $derived(engine.source === "mine");
  const ownCount = $derived(engine.own?.photos.length ?? kept?.count ?? 0);
  const busy = $derived(switching || engine.status === "loading" || engine.status === "building");

  async function useMine() {
    if (mine || busy) return;
    if (engine.own) {
      switching = true;
      await engine.useOwn(engine.own).finally(() => (switching = false));
      return;
    }
    if (!kept) {
      dialog = "new";
      return;
    }
    switching = true;
    try {
      const collection = await loadKept();
      if (collection) await engine.useOwn(collection);
      else dialog = "new";
    } catch {
      onmessage("Couldn't read the photos kept in this browser; add them again.");
      dialog = "new";
    } finally {
      switching = false;
    }
  }

  function useDemo() {
    if (!mine || busy) return;
    engine.preferOwn = false;
    void engine.useDemo();
  }

  async function forget() {
    await forgetKept();
    kept = null;
    engine.dropOwn();
    onmessage("Your photos are gone from this page and this browser.");
  }
</script>

<div class="source">
  <div class="seg" role="radiogroup" aria-label="Photos to search">
    <button role="radio" aria-checked={!mine} class:on={!mine} onclick={useDemo} disabled={busy && mine}>
      Demo photos
      <span class="num">29,825</span>
    </button>
    <button role="radio" aria-checked={mine} class:on={mine} onclick={useMine} disabled={busy && !mine}>
      {#if switching}<i class="spin" aria-hidden="true"></i>{:else if !ownCount}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg>{/if}
      Your photos
      {#if ownCount}<span class="num">{ownCount.toLocaleString()}</span>{/if}
    </button>
  </div>
  {#if mine && !busy}
    <div class="manage">
      <button onclick={() => (dialog = "add")}>Add photos</button>
      <span aria-hidden="true">·</span>
      <button onclick={() => (dialog = "new")}>Start over</button>
      <span aria-hidden="true">·</span>
      <button onclick={forget}>Forget them</button>
    </div>
  {/if}
</div>

{#if dialog}
  <OwnPhotos
    add={dialog === "add"}
    onclose={() => (dialog = null)}
    ondone={(summary) => {
      dialog = null;
      keptInfo().then((info) => (kept = info));
      onmessage(summary);
    }}
  />
{/if}

<style>
  .source {
    display: grid;
    gap: 6px;
  }
  .seg {
    display: grid;
    grid-template-columns: 1fr 1fr;
    padding: 3px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
  }
  .seg button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    min-height: 36px;
    padding: 0 10px;
    border: 0;
    border-radius: 9px;
    background: transparent;
    color: var(--text-2);
    font: 500 13px var(--font);
    white-space: nowrap;
    cursor: pointer;
    transition:
      background 0.2s,
      color 0.2s;
  }
  .seg button:hover:not(:disabled) {
    color: var(--text);
  }
  .seg button.on {
    background: var(--surface-3);
    color: var(--text);
    box-shadow: 0 1px 0 var(--border-strong) inset;
  }
  .seg button:disabled {
    cursor: default;
  }
  .seg button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  .seg .num {
    color: var(--muted);
    font: 11.5px var(--mono);
  }
  .seg button.on .num {
    color: var(--accent-strong);
  }
  .seg svg {
    width: 14px;
    height: 14px;
    fill: none;
    stroke: currentColor;
    stroke-width: 2.2;
    stroke-linecap: round;
  }
  .spin {
    width: 12px;
    height: 12px;
    border: 2px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  /* Narrow phones: each label and its count on one line. */
  @media (max-width: 400px) {
    .seg button {
      gap: 5px;
      padding: 0 6px;
      font-size: 12.5px;
    }
    .seg .num {
      font-size: 11px;
    }
  }
  .manage {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 6px;
    color: var(--muted);
    font-size: 12px;
  }
  .manage button {
    padding: 2px 0;
    border: 0;
    background: none;
    color: var(--text-2);
    font: inherit;
    cursor: pointer;
  }
  .manage button:hover {
    color: var(--accent-strong);
  }
</style>
