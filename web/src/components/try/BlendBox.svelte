<script lang="ts" module>
  /** The photo a blend starts from: one of the collection's photos, or an uploaded/camera image. */
  export interface BlendBase {
    id?: number;
    vector?: Float32Array;
    preview?: string;
  }
</script>

<script lang="ts">
  import { untrack } from "svelte";
  import { textEncoder } from "../../lib/engine/encoders.svelte";
  import { engine } from "../../lib/engine/engine.svelte";
  import type { Query } from "../../lib/engine/protocol";
  import Thumb from "../Thumb.svelte";

  let {
    base,
    onsearch,
    disabled = false,
    initialWords = "",
    initialMix = 0.5,
  }: {
    base: BlendBase | null;
    onsearch: (query: Query, options: { live: boolean }) => Promise<void>;
    disabled?: boolean;
    initialWords?: string;
    initialMix?: number;
  } = $props();

  const suggestions = ["at night", "in the snow", "black and white", "a watercolor painting", "underwater", "made of lego"];

  // svelte-ignore state_referenced_locally
  let words = $state(initialWords);
  // svelte-ignore state_referenced_locally
  let mix = $state(initialMix);
  let wordsVector = $state.raw<Float32Array | null>(null);
  let wordsFor = "";
  let baseVector = $state.raw<Float32Array | null>(null);
  let busy = $state(false);
  let error = $state("");

  // CLIP puts text and images in different regions of the same space (the "modality gap"): a
  // photo's similarity to other photos varies ~2.5x more than a sentence's. Scaling the words by
  // that factor makes the slider's midpoint an even mix instead of "the photo, barely changed".
  const TEXT_GAIN = 2.5;

  // Resolve the base photo's vector whenever the base changes.
  $effect(() => {
    const current = base;
    baseVector = null;
    if (!current) return;
    if (current.vector) baseVector = current.vector;
    else if (current.id !== undefined)
      engine.vector(current.id).then(
        (v) => base === current && (baseVector = v),
        (e) => (error = String(e)),
      );
  });

  // Re-run the blend when the base photo or the words change (not on every slider move: the
  // slider runs its own live searches).
  $effect(() => {
    if (baseVector && wordsVector) untrack(() => go(false));
  });

  async function setWords(value = words) {
    const trimmed = value.trim();
    if (!trimmed) return;
    words = trimmed;
    if (trimmed === wordsFor && wordsVector) return go(false); // same words: the effect won't re-run
    busy = true;
    error = "";
    try {
      wordsVector = await textEncoder.encode(trimmed);
      wordsFor = trimmed;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  function go(live: boolean) {
    if (!baseVector || !wordsVector) return;
    const q = new Float32Array(baseVector.length);
    let norm = 0;
    for (let i = 0; i < q.length; i++) {
      q[i] = (1 - mix) * baseVector[i] + mix * TEXT_GAIN * wordsVector[i];
      norm += q[i] * q[i];
    }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < q.length; i++) q[i] /= norm;
    inFlight = true;
    onsearch(
      {
        kind: "vector",
        source: "blend",
        text: `photo + “${wordsFor}”`,
        vector: q,
        preview: base?.preview,
        blend: { id: base?.id, words: wordsFor, mix },
      },
      { live },
    )
      .catch(() => {})
      .finally(() => {
        inFlight = false;
        // The slider moved while that search ran: catch up with its latest position.
        if (stale) {
          stale = false;
          go(true);
        }
      });
  }

  // Dragging the slider searches continuously. The HNSW search itself takes well under a
  // millisecond, but each also runs an exact scan for comparison, so at most one is in flight:
  // moves during a search collapse into one follow-up with the latest position. Results keep up
  // with the finger on any device instead of lagging behind a queue.
  let inFlight = false;
  let stale = false;
  function onMix(value: number) {
    mix = value;
    if (inFlight) stale = true;
    else go(true);
  }

  // Start from the URL's blend (a shared link); the search runs once the base photo is ready.
  $effect(() => {
    untrack(() => initialWords && setWords(initialWords));
  });
</script>

<div class="blend">
  <div class="row">
    <div class="base" title="The photo you're blending from">
      {#if base?.id !== undefined}
        <Thumb id={base.id} size="q" alt={engine.caption(base.id)} />
      {:else if base?.preview}
        <img src={base.preview} alt="Uploaded query" />
      {:else}
        <span class="empty">?</span>
      {/if}
      <span class="label">photo</span>
    </div>
    <span class="plus" aria-hidden="true">+</span>
    <form class="field" onsubmit={(e) => (e.preventDefault(), setWords())}>
      <input
        bind:value={words}
        placeholder={textEncoder.status === "ready" ? "add words, e.g. at night" : "Loading the text model…"}
        aria-label="Words to blend with the photo"
        disabled={disabled || textEncoder.status !== "ready"}
        autocomplete="off"
        spellcheck="false"
      />
      {#if busy}<span class="spinner" aria-hidden="true"></span>{/if}
    </form>
  </div>

  <div class="chips" role="group" aria-label="Blend ideas">
    {#each suggestions as s (s)}
      <button onclick={() => setWords(s)} disabled={disabled || textEncoder.status !== "ready"}>{s}</button>
    {/each}
  </div>

  <label class="mix" class:off={!wordsVector}>
    <span>Photo</span>
    <input
      type="range"
      min="0"
      max="1"
      step="0.01"
      value={mix}
      oninput={(e) => onMix(Number((e.currentTarget as HTMLInputElement).value))}
      disabled={!wordsVector || !baseVector}
      aria-label="Balance between the photo and the words"
      aria-valuetext="{Math.round((1 - mix) * 100)}% photo, {Math.round(mix * 100)}% words"
    />
    <span>Words</span>
  </label>
  <p class="hint">
    {#if !wordsVector}
      Starts from the current photo; click any result to change it. Then drag the slider and watch the results morph.
    {:else}
      Query = normalize({(1 - mix).toFixed(2)} · photo + {mix.toFixed(2)} · words). Every move is a fresh HNSW search.
    {/if}
  </p>
  {#if error}<p class="error" role="alert">{error}</p>{/if}
</div>

<style>
  .blend {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 12px;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .base {
    position: relative;
    flex: none;
    width: 50px;
    height: 50px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 0 2px var(--result);
  }
  .base :global(.thumb),
  .base img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .base :global(img) {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .empty {
    display: grid;
    place-items: center;
    height: 100%;
    color: var(--muted);
  }
  .label {
    position: absolute;
    inset: auto 0 0 0;
    padding: 1px 0;
    background: rgba(0, 0, 0, 0.55);
    color: #fff;
    font: 600 9.5px var(--mono);
    text-align: center;
  }
  .plus {
    color: var(--muted);
    font-size: 18px;
  }
  .field {
    display: flex;
    flex: 1;
    align-items: center;
    min-width: 0;
    height: 50px;
    padding: 0 14px;
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    background: var(--surface-solid);
  }
  .field:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 4px var(--accent-wash);
  }
  .field input {
    flex: 1;
    width: 0;
    min-width: 0;
    border: 0;
    outline: 0;
    background: transparent;
    color: var(--text);
    font: 16px var(--font);
  }
  .field input::placeholder {
    color: var(--muted);
  }
  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .chips button {
    padding: 5px 11px;
    border: 1px solid var(--border);
    border-radius: 99px;
    background: var(--surface);
    color: var(--text-2);
    font-size: 12.5px;
    cursor: pointer;
  }
  .chips button:hover:not(:disabled) {
    border-color: var(--border-strong);
    color: var(--text);
  }
  .chips button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .mix {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text-2);
    font: 12.5px var(--mono);
  }
  .mix input {
    flex: 1;
    accent-color: var(--result);
  }
  .mix.off {
    opacity: 0.5;
  }
  .hint {
    margin: 0;
    color: var(--muted);
    font-size: 12.5px;
    line-height: 1.5;
  }
  .error {
    margin: 0;
    color: var(--warn);
    font-size: 13px;
  }
</style>
