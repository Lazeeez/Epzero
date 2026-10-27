<script lang="ts">
  import { engine } from "../../lib/engine/engine.svelte";

  // Restricts the playground's searches to photos with any of the chosen labels. The engine
  // decides per query how to honor the filter (SearchStats says which way it went).
  let { onchange, disabled = false }: { onchange: () => void; disabled?: boolean } = $props();

  const selected = $derived(engine.filter?.labels ?? []);
  let open = $state(false);
  let query = $state("");
  let applying = $state(false);
  let root: HTMLElement | undefined = $state();
  let input: HTMLInputElement | undefined = $state();

  const categories = $derived(engine.meta ? engine.categories() : []);
  const matches = $derived.by(() => {
    const q = query.trim().toLowerCase();
    return (q ? categories.filter((c) => c.label.toLowerCase().includes(q)) : categories).slice(0, 60);
  });
  // A few picks that read well in photos, shown before the picker is opened.
  const QUICK = ["Dog", "Cat", "Car", "Food", "Flower", "Bird", "Building", "Tree"];
  const quick = $derived(QUICK.filter((label) => categories.some((c) => c.label === label)));

  async function apply(labels: string[]) {
    applying = true;
    try {
      await engine.setFilter(labels);
      onchange();
    } finally {
      applying = false;
    }
  }
  const toggle = (label: string) =>
    apply(selected.includes(label) ? selected.filter((l) => l !== label) : [...selected, label]);

  let picker: HTMLElement | undefined = $state();
  $effect(() => {
    if (!open) return;
    input?.focus({ preventScroll: true });
    picker?.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    const away = (event: PointerEvent) => {
      if (root && !root.contains(event.target as Node)) open = false;
    };
    addEventListener("pointerdown", away);
    return () => removeEventListener("pointerdown", away);
  });
</script>

<div class="filter" bind:this={root} role="group" aria-label="Filter by category" aria-busy={applying}>
  <div class="row">
    <span class="label">Only</span>
    {#each selected as label (label)}
      <button class="chip on" onclick={() => toggle(label)} disabled={disabled || applying} aria-label="Remove the {label} filter">
        {label}<span aria-hidden="true">×</span>
      </button>
    {:else}
      {#each quick as label (label)}
        <button class="chip" onclick={() => toggle(label)} disabled={disabled || applying}>{label}</button>
      {/each}
    {/each}
    <button
      class="chip add"
      onclick={() => (open = !open)}
      disabled={disabled || !categories.length}
      aria-expanded={open}
      aria-controls="category-picker"
    >
      + {selected.length ? "Add" : "More"}
    </button>
    {#if selected.length}
      <button class="clear" onclick={() => apply([])} disabled={disabled || applying}>All photos</button>
    {/if}
  </div>
  {#if engine.filter}
    <p class="summary num" aria-live="polite">
      {engine.filter.allowed.toLocaleString()} of {engine.count.toLocaleString()} photos ({(engine.filter.selectivity * 100).toFixed(1)}%)
    </p>
  {/if}

  {#if open}
    <div id="category-picker" class="picker card" bind:this={picker}>
      <input
        bind:this={input}
        bind:value={query}
        type="search"
        placeholder="Search {categories.length.toLocaleString()} categories"
        aria-label="Search categories"
        onkeydown={(e) => e.key === "Escape" && (open = false)}
      />
      <ul aria-label="Categories">
        {#each matches as category (category.label)}
          <li>
            <button
              aria-pressed={selected.includes(category.label)}
              onclick={() => toggle(category.label)}
              disabled={applying}
            >
              <span>{category.label}</span>
              <span class="num count">{category.count.toLocaleString()}</span>
            </button>
          </li>
        {:else}
          <li class="none">No category matches “{query}”.</li>
        {/each}
      </ul>
    </div>
  {/if}
</div>

<style>
  .filter {
    position: relative;
    display: grid;
    gap: 6px;
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }
  .label {
    color: var(--muted);
    font: 600 10.5px var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .chip,
  .clear {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border: 1px solid var(--border);
    border-radius: 99px;
    background: var(--surface);
    color: var(--text-2);
    font-size: 12.5px;
    cursor: pointer;
    transition: border-color 0.2s, color 0.2s, background 0.2s;
  }
  .chip:hover:not(:disabled),
  .clear:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--text);
  }
  .chip.on {
    border-color: var(--accent);
    background: var(--accent-wash);
    color: var(--accent-strong);
  }
  .chip.add {
    border-style: dashed;
  }
  .clear {
    border-color: transparent;
    background: none;
    color: var(--muted);
  }
  button:disabled {
    opacity: 0.5;
    cursor: progress;
  }
  .summary {
    margin: 0;
    color: var(--muted);
    font-size: 12px;
  }
  /* In the flow, not floating: inside the scrolling search panel (a short sheet on phones) a
     floating list would be clipped by the panel's edge. */
  .picker {
    display: grid;
    gap: 8px;
    padding: 10px;
    background: var(--surface-solid);
    box-shadow: var(--shadow);
  }
  input {
    height: 36px;
    padding: 0 12px;
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    background: var(--surface);
    color: var(--text);
    font: inherit;
    font-size: 13.5px;
  }
  input:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  ul {
    max-height: 240px;
    margin: 0;
    padding: 0;
    overflow-y: auto;
    list-style: none;
    scrollbar-width: thin;
  }
  li button {
    display: flex;
    justify-content: space-between;
    width: 100%;
    padding: 7px 10px;
    border: 0;
    border-radius: 8px;
    background: none;
    color: var(--text-2);
    font-size: 13px;
    text-align: left;
    cursor: pointer;
  }
  li button:hover,
  li button:focus-visible {
    background: var(--surface-3);
    color: var(--text);
  }
  li button[aria-pressed="true"] {
    color: var(--accent-strong);
    font-weight: 600;
  }
  li button[aria-pressed="true"] span:first-child::before {
    content: "✓ ";
  }
  .count {
    color: var(--muted);
    font-size: 12px;
  }
  .none {
    padding: 8px 10px;
    color: var(--muted);
    font-size: 13px;
  }
  /* Phones: one row that scrolls sideways, with bigger taps (see SearchBox's chips). */
  @media (max-width: 900px) {
    .row {
      flex-wrap: nowrap;
      overflow-x: auto;
      margin-inline: -16px;
      padding-inline: 16px;
      scrollbar-width: none;
    }
    .row::-webkit-scrollbar {
      display: none;
    }
    .row > :global(*) {
      flex: none;
    }
    .chip,
    .clear {
      padding: 8px 13px;
      font-size: 13.5px;
    }
  }
</style>
