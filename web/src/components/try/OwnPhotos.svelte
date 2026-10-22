<script lang="ts" module>
  // Every file under the dropped items, folders included (DataTransfer.files lists folders as
  // empty entries).
  export async function droppedFiles(data: DataTransfer | null): Promise<File[]> {
    if (!data) return [];
    const entries = Array.from(data.items ?? [])
      .map((item) => item.webkitGetAsEntry?.())
      .filter((entry): entry is FileSystemEntry => !!entry);
    if (!entries.length) return Array.from(data.files);
    const files: File[] = [];
    const walk = async (entry: FileSystemEntry): Promise<void> => {
      if (entry.isFile) {
        files.push(await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject)));
      } else if (entry.isDirectory) {
        const reader = (entry as FileSystemDirectoryEntry).createReader();
        for (;;) {
          const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
          if (!batch.length) break;
          for (const child of batch) await walk(child);
        }
      }
    };
    for (const entry of entries) await walk(entry);
    return files;
  }
</script>

<script lang="ts">
  import { engine } from "../../lib/engine/engine.svelte";
  import { forgetKept, imageFiles, keepCollection, photoIndexer as indexer, photoLimit, poolSize } from "../../lib/engine/own-photos.svelte";

  // Picks the visitor's photos and indexes them: embedding (a pool of CLIP workers), the map
  // (UMAP), then the graph (the engine, on every core). `add`: append to the current collection.
  let { add = false, onclose, ondone }: { add?: boolean; onclose: () => void; ondone: (summary: string) => void } = $props();

  const KEEP_KEY = "epzero-keep-photos";
  let keep = $state(readKeep());
  function readKeep(): boolean {
    try {
      return localStorage.getItem(KEEP_KEY) !== "0";
    } catch {
      return true;
    }
  }
  function setKeep(on: boolean) {
    keep = on;
    try {
      localStorage.setItem(KEEP_KEY, on ? "1" : "0");
    } catch {
      /* not remembered */
    }
  }

  const limit = photoLimit();
  const workers = poolSize();
  const touch = matchMedia("(pointer: coarse)").matches;
  let dialog: HTMLDialogElement | undefined = $state();
  let dragging = $state(false);
  let folderInput: HTMLInputElement | undefined = $state();
  let notice = $state("");
  let building = $state(false);
  let started = 0;

  $effect(() => {
    if (!dialog) return;
    dialog.showModal();
    return () => dialog?.close();
  });

  const working = $derived(indexer.busy || building);
  // The graph step: the engine reports it once the vectors are in.
  const step = $derived(
    building ? 3 : indexer.phase === "layout" || indexer.phase === "saving" ? 2 : indexer.phase === "embedding" ? 1 : indexer.phase === "model" ? 0 : -1,
  );
  const steps = ["Image model", "Embed photos", "Lay out the map", "Build the graph"];

  async function start(list: FileList | File[] | null | undefined) {
    if (!list || working) return;
    let files = imageFiles(list);
    const room = limit - (add ? (engine.own?.photos.length ?? 0) : 0);
    notice = "";
    if (files.length === 0) {
      notice = "No photos there. Pick image files (JPEG, PNG, WebP, AVIF, GIF; HEIC in Safari), or a folder of them.";
      return;
    }
    if (room <= 0) {
      notice = `This collection is at the ${limit.toLocaleString()}-photo limit for this device.`;
      return;
    }
    if (files.length > room) {
      notice = `Using the first ${room.toLocaleString()} of ${files.length.toLocaleString()} photos: the limit on this device is ${limit.toLocaleString()}.`;
      files = files.slice(0, room);
    }
    started = performance.now();
    const collection = await indexer.run(files, add ? engine.own : null);
    if (!collection) return; // cancelled, or failed (the error shows)
    building = true;
    try {
      await engine.useOwn(collection);
      if (engine.status !== "ready") throw new Error(engine.error || "The graph could not be built.");
      if (keep) await keepCollection(collection).catch(() => {});
      else await forgetKept();
      const seconds = Math.max(1, Math.round((performance.now() - started) / 1000));
      const skipped = indexer.skipped ? ` ${indexer.skipped.toLocaleString()} ${indexer.skipped === 1 ? "file was" : "files were"} skipped (not readable in this browser).` : "";
      ondone(`${collection.photos.length.toLocaleString()} of your photos indexed in ${seconds} s.${skipped}`);
      indexer.reset();
    } catch (error) {
      indexer.error = error instanceof Error ? error.message : String(error);
      indexer.phase = "error";
    } finally {
      building = false;
    }
  }

  function close() {
    if (building) return; // the graph build finishes in a moment
    if (indexer.busy) indexer.cancel();
    else indexer.reset();
    onclose();
  }

  const clock = (seconds: number) => (seconds >= 90 ? `${Math.round(seconds / 60)} min` : `${Math.max(1, Math.round(seconds))} s`);
  // The mosaic: the newest thumbnails, with empty cells shimmering until photos arrive.
  const CELLS = 24;
  const cells = $derived.by(() => {
    const shown: (string | null)[] = indexer.recent.slice(-CELLS);
    while (shown.length < CELLS) shown.push(null);
    return shown;
  });
</script>

<dialog
  class="own card"
  aria-labelledby="own-title"
  bind:this={dialog}
  oncancel={(e) => {
    e.preventDefault(); // Escape: the same as the close button
    close();
  }}
  onclick={(e) => {
    const box = dialog!.getBoundingClientRect();
    const outside = e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom;
    if (e.target === dialog && outside && !working) close();
  }}
>
  <header>
    <div>
      <h2 id="own-title">{add ? "Add photos" : "Search your own photos"}</h2>
      <p>CLIP turns each photo into a vector right here in this tab, and Epzero indexes them. Then search them by description, by photo or with your camera.</p>
    </div>
    <button class="close" onclick={close} aria-label={indexer.busy ? "Cancel" : "Close"} disabled={building}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" /></svg>
    </button>
  </header>

  {#if !working && indexer.phase !== "error"}
    <label
      class="drop"
      class:dragging
      ondragover={(e) => {
        e.preventDefault();
        dragging = true;
      }}
      ondragleave={() => (dragging = false)}
      ondrop={async (e) => {
        e.preventDefault();
        dragging = false;
        start(await droppedFiles(e.dataTransfer));
      }}
    >
      <input class="sr" type="file" accept="image/*" multiple onchange={(e) => start(e.currentTarget.files)} />
      <svg class="art" viewBox="0 0 64 64" aria-hidden="true">
        <rect x="14" y="10" width="38" height="30" rx="5" class="back" />
        <rect x="8" y="18" width="40" height="32" rx="5" class="front" />
        <path d="M14 44l9-10 7 7 5-5 9 8" />
        <circle cx="35" cy="28" r="3.2" />
      </svg>
      <strong>{touch ? "Choose photos" : "Drop photos or a folder here"}</strong>
      <span class="hint">{touch ? "From your library or camera roll" : "or pick them yourself"}</span>
      <span class="buttons">
        <span class="btn btn-primary" aria-hidden="true">Choose photos</span>
        {#if !touch}
          <button
            type="button"
            class="btn btn-ghost"
            onclick={(e) => {
              e.preventDefault();
              folderInput?.click();
            }}>Choose a folder</button
          >
        {/if}
      </span>
    </label>
    <input class="sr" type="file" bind:this={folderInput} webkitdirectory multiple onchange={(e) => start(e.currentTarget.files)} tabindex="-1" aria-hidden="true" />

    {#if notice}<p class="notice" role="status">{notice}</p>{/if}

    <ul class="points">
      <li>
        <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></svg>
        <span><b>Nothing is uploaded.</b> The model and the engine both run on this device.</span>
      </li>
      <li>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 3 4 14h7l-1 7 9-11h-7z" /></svg>
        <span><b>Up to {limit.toLocaleString()} photos.</b> {workers > 1 ? `${workers} encoders run side by side: about a minute per 1,000 photos here.` : "A few photos a second on this device."}</span>
      </li>
      <li>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h10" /></svg>
        <span><b>JPEG, PNG, WebP, AVIF, GIF.</b> HEIC opens in Safari; elsewhere it's skipped.</span>
      </li>
    </ul>

    <label class="keep">
      <input type="checkbox" role="switch" checked={keep} onchange={(e) => setKeep(e.currentTarget.checked)} />
      <span>Keep them in this browser for next time <small>stored only here, as small thumbnails and vectors; forget them any time</small></span>
    </label>
  {:else}
    <ol class="steps" aria-label="Progress">
      {#each steps as name, i (name)}
        <li class:done={i < step} class:now={i === step}>
          <span class="dot" aria-hidden="true">
            {#if i < step}<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>{/if}
          </span>
          {name}
        </li>
      {/each}
    </ol>

    <div class="mosaic" aria-hidden="true">
      {#each cells as url, i (url ?? `empty-${i}`)}
        {#if url}<img src={url} alt="" />{:else}<span class="empty"></span>{/if}
      {/each}
    </div>

    {#if indexer.phase === "error"}
      <div class="failed" role="alert">
        <strong>That didn't work</strong>
        <p>{indexer.error}</p>
        <button class="btn btn-ghost" onclick={() => indexer.reset()}>Try again</button>
      </div>
    {:else}
      <div class="count" aria-live="polite">
        {#if building}
          <strong class="num">{engine.built.toLocaleString()}<small>&nbsp;/ {engine.count.toLocaleString()}</small></strong>
          <span>Building the HNSW graph on every core</span>
        {:else if indexer.phase === "model"}
          <strong class="num">{Math.round(indexer.modelProgress * 100)}<small>%</small></strong>
          <span>Loading the image model (89 MB, once: it's cached after this)</span>
        {:else if indexer.phase === "layout"}
          <strong class="num">{Math.round(indexer.layoutProgress * 100)}<small>%</small></strong>
          <span>Laying out the map of your photos</span>
        {:else}
          <strong class="num">{indexer.done.toLocaleString()}<small>&nbsp;/ {indexer.total.toLocaleString()}</small></strong>
          <span class="num">
            {#if indexer.rate > 0}{indexer.rate.toFixed(indexer.rate < 10 ? 1 : 0)} photos/s · about {clock(indexer.eta)} left · {indexer.workers} {indexer.workers === 1 ? "encoder" : "encoders"}{:else}Starting the encoders{/if}
          </span>
        {/if}
      </div>
      <div class="meter">
        <i
          style:width="{(building
            ? engine.built / Math.max(engine.count, 1)
            : indexer.phase === "model"
              ? indexer.modelProgress
              : indexer.phase === "layout"
                ? indexer.layoutProgress
                : indexer.done / Math.max(indexer.total, 1)) * 100}%"
        ></i>
      </div>
      {#if notice}<p class="notice">{notice}</p>{/if}
      <footer>
        <span class="private">
          <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></svg>
          On this device only
        </span>
        <button class="btn btn-ghost" onclick={close} disabled={building}>Cancel</button>
      </footer>
    {/if}
  {/if}
</dialog>


<style>
  .own {
    display: grid;
    gap: 18px;
    width: min(640px, 100% - 32px);
    max-width: none;
    max-height: calc(100dvh - 32px);
    margin: auto;
    overflow: auto;
    padding: 24px;
    background: var(--surface-solid);
    color: var(--text);
    animation: pop 0.35s var(--ease);
  }
  .own::backdrop {
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(6px);
    animation: fade 0.25s var(--ease);
  }
  header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
  }
  h2 {
    font-size: 20px;
    letter-spacing: -0.02em;
  }
  header p {
    margin: 6px 0 0;
    color: var(--text-2);
    font-size: 13.5px;
  }
  .close {
    display: grid;
    place-items: center;
    flex: none;
    width: 34px;
    height: 34px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    color: var(--text-2);
    cursor: pointer;
  }
  .close:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .close svg {
    width: 16px;
    height: 16px;
    stroke: currentColor;
    stroke-width: 2;
    stroke-linecap: round;
  }
  .sr {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
  }
  .drop {
    position: relative;
    display: grid;
    justify-items: center;
    gap: 6px;
    padding: 34px 20px 28px;
    border: 1.5px dashed var(--border-strong);
    border-radius: 18px;
    background: radial-gradient(120% 90% at 50% 0%, var(--accent-wash), transparent 70%);
    text-align: center;
    cursor: pointer;
    transition:
      border-color 0.2s,
      background-color 0.2s,
      transform 0.2s var(--ease);
  }
  .drop:hover,
  .drop:has(input:focus-visible) {
    border-color: var(--accent);
  }
  .drop:has(input:focus-visible) {
    outline: 2px solid var(--accent);
    outline-offset: 3px;
  }
  .drop.dragging {
    border-color: var(--accent);
    background-color: var(--accent-wash);
    transform: scale(1.01);
  }
  .art {
    width: 64px;
    height: 64px;
    margin-bottom: 6px;
    fill: none;
    stroke: var(--accent-strong);
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .art .back {
    opacity: 0.45;
  }
  .art .front {
    fill: var(--surface-solid);
  }
  .drop strong {
    font-size: 16px;
  }
  .hint {
    color: var(--muted);
    font-size: 13px;
  }
  .buttons {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin-top: 14px;
  }
  .buttons .btn {
    height: 40px;
    padding: 0 16px;
    font-size: 14px;
  }
  .points {
    display: grid;
    gap: 10px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .points li {
    display: grid;
    grid-template-columns: 18px 1fr;
    gap: 10px;
    color: var(--text-2);
    font-size: 13px;
  }
  .points b {
    color: var(--text);
    font-weight: 600;
  }
  .points svg,
  .private svg {
    width: 18px;
    height: 18px;
    fill: none;
    stroke: var(--accent-strong);
    stroke-width: 1.8;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .keep {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px 14px;
    border: 1px solid var(--border);
    border-radius: 12px;
    font-size: 13.5px;
    cursor: pointer;
  }
  .keep input {
    flex: none;
    margin-top: 2px;
    accent-color: var(--accent);
  }
  .keep small {
    display: block;
    color: var(--muted);
    font-size: 12px;
  }
  .notice {
    margin: 0;
    padding: 10px 12px;
    border-radius: 10px;
    background: color-mix(in srgb, var(--warn) 12%, transparent);
    color: var(--warn);
    font-size: 13px;
  }
  .steps {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: 0;
    padding: 0;
    list-style: none;
    counter-reset: step;
  }
  .steps li {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--muted);
    font-size: 12.5px;
    white-space: nowrap;
  }
  .steps .dot {
    display: grid;
    place-items: center;
    flex: none;
    width: 18px;
    height: 18px;
    border: 2px solid var(--border-strong);
    border-radius: 50%;
  }
  .steps .now {
    color: var(--text);
  }
  .steps .now .dot {
    border-color: var(--accent);
    border-top-color: transparent;
    animation: spin 0.8s linear infinite;
  }
  .steps .done {
    color: var(--text-2);
  }
  .steps .done .dot {
    border-color: var(--good);
    background: var(--good);
  }
  .steps svg {
    width: 12px;
    height: 12px;
    fill: none;
    stroke: var(--page);
    stroke-width: 3;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .mosaic {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    gap: 6px;
  }
  .mosaic img,
  .mosaic .empty {
    width: 100%;
    aspect-ratio: 1;
    border-radius: 8px;
  }
  .mosaic img {
    object-fit: cover;
    animation: arrive 0.45s var(--ease);
  }
  .mosaic .empty {
    background: linear-gradient(110deg, var(--surface-2) 40%, var(--surface-3) 50%, var(--surface-2) 60%);
    background-size: 250% 100%;
    animation: shimmer 1.4s linear infinite;
  }
  @keyframes pop {
    from {
      opacity: 0;
      transform: translateY(10px) scale(0.98);
    }
  }
  @keyframes fade {
    from {
      opacity: 0;
    }
  }
  @keyframes shimmer {
    to {
      background-position: -150% 0;
    }
  }
  @keyframes arrive {
    from {
      opacity: 0;
      transform: scale(0.85);
    }
  }
  .count {
    display: grid;
    gap: 2px;
  }
  .count strong {
    font: 600 34px var(--mono);
    letter-spacing: -0.03em;
  }
  .count small {
    color: var(--muted);
    font-size: 18px;
  }
  .count span {
    color: var(--text-2);
    font-size: 13px;
  }
  .meter {
    height: 6px;
    border-radius: 99px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .meter i {
    display: block;
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    transition: width 0.3s var(--ease);
  }
  footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  footer .btn {
    height: 38px;
    padding: 0 16px;
    font-size: 13.5px;
  }
  .private {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--muted);
    font-size: 12.5px;
  }
  .failed {
    display: grid;
    gap: 8px;
    justify-items: start;
  }
  .failed p {
    margin: 0;
    color: var(--text-2);
    font-size: 13.5px;
  }
  @media (max-width: 600px) {
    .own {
      width: 100%;
      max-height: 100dvh;
      margin: auto 0 0;
      padding: 20px 16px calc(20px + env(safe-area-inset-bottom, 0px));
      border-radius: 20px 20px 0 0;
    }
    .mosaic {
      grid-template-columns: repeat(6, 1fr);
    }
    .mosaic > :nth-child(-n + 6) {
      display: none;
    }
    .steps {
      grid-template-columns: 1fr 1fr;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .mosaic img,
    .own {
      animation: none;
    }
  }
</style>
