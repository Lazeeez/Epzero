<script lang="ts">
  import { engine, type SearchOutcome } from "../../lib/engine/engine.svelte";
  import Thumb from "../Thumb.svelte";

  let { outcome, onpick }: { outcome: SearchOutcome; onpick: (id: number) => void } = $props();

  const query = $derived(outcome.query);
  const exact = $derived(new Set(outcome.trace.exact));
</script>

<div class="results">
  <div class="head">
    <h2>
      {#if query.kind === "vector"}
        {#if query.preview}<img class="query-img" src={query.preview} alt="" />{/if}
        Results for {query.source === "text" ? `“${query.text}”` : query.text}
      {:else}
        Photos like #{query.id}
        <a class="source" href={engine.source(query.id)} target="_blank" rel="noopener noreferrer" title="The original photo on Flickr">original ↗</a>
      {/if}
    </h2>
    <span class="hint">Click one to search for similar photos</span>
  </div>
  <ol>
    {#each Array.from(outcome.trace.results) as id, rank (id)}
      <li style:animation-delay="{rank * 28}ms">
        <button
          class="result"
          onclick={() => onpick(id)}
          aria-label="Result {rank + 1}: {engine.caption(id)}, by {engine.author(id)}. Search for similar photos."
        >
          <Thumb {id} size="n" alt={engine.caption(id)} eager={rank < 12} />
          <span class="rank num">{query.kind === "id" && query.id === id ? "query" : `#${rank + 1}`}</span>
          <span class="score num" title="Cosine similarity">{(1 - outcome.trace.resultDistances[rank]).toFixed(2)}</span>
          {#if !exact.has(id)}<span class="approx" title="Not in the exact top-{outcome.trace.results.length}">≈</span>{/if}
          <span class="caption">
            <span>{engine.caption(id)}</span>
            <span class="by">© {engine.author(id)}</span>
          </span>
        </button>
      </li>
    {/each}
  </ol>
</div>

<style>
  .head {
    display: grid;
    gap: 2px;
    margin-bottom: 10px;
  }
  h2 {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 8px;
    min-width: 0;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: -0.01em;
    overflow-wrap: anywhere; /* long queries wrap instead of being cut off */
  }
  .source {
    font-size: 12px;
    font-weight: 500;
  }
  .query-img {
    width: 22px;
    height: 22px;
    border-radius: 6px;
    object-fit: cover;
  }
  .hint {
    color: var(--muted);
    font-size: 12px;
  }
  ol {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  li {
    animation: rise 0.45s var(--ease) both;
  }
  .result {
    position: relative;
    display: block;
    width: 100%;
    padding: 0;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface-2);
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.25s var(--ease), border-color 0.2s;
  }
  .result:hover,
  .result:focus-visible {
    transform: translateY(-2px);
    border-color: var(--result);
  }
  .result :global(img) {
    display: block;
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    transition: transform 0.5s var(--ease);
  }
  .result:hover :global(img) {
    transform: scale(1.06);
  }
  .rank,
  .score,
  .approx {
    position: absolute;
    top: 6px;
    padding: 1px 7px;
    border-radius: 99px;
    background: rgba(0, 0, 0, 0.62);
    color: #fff;
    font: 600 10.5px/1.6 var(--mono);
  }
  .rank {
    left: 6px;
  }
  .score {
    right: 6px;
  }
  .approx {
    right: 6px;
    top: 28px;
    background: color-mix(in srgb, var(--warn) 80%, black);
  }
  .caption {
    position: absolute;
    inset: auto 0 0 0;
    display: grid;
    padding: 18px 7px 6px;
    background: linear-gradient(transparent, rgba(0, 0, 0, 0.78));
    color: #fff;
    font-size: 12px;
    line-height: 1.3;
    text-align: left;
    opacity: 0;
    transition: opacity 0.2s;
  }
  .result:hover .caption,
  .result:focus-visible .caption {
    opacity: 1;
  }
  .caption span {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
  .by {
    color: rgba(255, 255, 255, 0.7);
  }
  @keyframes rise {
    from {
      opacity: 0;
      transform: translateY(8px) scale(0.98);
    }
  }
</style>
