<script lang="ts">
  import { untrack } from "svelte";
  import { imageEncoder, textEncoder } from "../../lib/engine/encoders.svelte";
  import type { Query } from "../../lib/engine/protocol";
  import BlendBox, { type BlendBase } from "./BlendBox.svelte";

  let {
    onsearch,
    onsurprise,
    disabled = false,
    initialText = "",
    blendBase = null,
    initialBlend = null,
  }: {
    onsearch: (query: Query, options?: { live: boolean }) => Promise<void>;
    onsurprise: () => void;
    disabled?: boolean;
    initialText?: string;
    blendBase?: BlendBase | null;
    initialBlend?: { words: string; mix: number } | null;
  } = $props();

  type Mode = "text" | "photo" | "camera" | "blend";
  let mode = $state<Mode>(untrack(() => (initialBlend ? "blend" : "text")));
  let text = $state(untrack(() => initialText)); // the box is editable; the prop only seeds it
  let busy = $state(false);
  let error = $state("");
  let input: HTMLInputElement | undefined = $state();
  let preview = $state<string | null>(null);
  let dragging = $state(false);

  // Camera
  let video: HTMLVideoElement | undefined = $state();
  let stream: MediaStream | null = null;
  let cameraOn = $state(false);
  let live = $state(false);
  let liveTimer = 0;

  const examples = ["a dog on the beach", "snowy mountains", "city lights at night", "a bowl of fruit", "vintage black and white photo", "a red sports car"];

  $effect(() => {
    if (mode === "text" || mode === "blend") textEncoder.start();
    else imageEncoder.start();
  });

  // "/" focuses the search box from anywhere on the page.
  $effect(() => {
    const onKey = (event: KeyboardEvent) => {
      const typing = (event.target as HTMLElement).closest?.("input, textarea, [contenteditable]");
      if (event.key === "/" && !typing && !event.ctrlKey && !event.metaKey && !document.querySelector("dialog[open]")) {
        event.preventDefault();
        mode = "text";
        queueMicrotask(() => input?.focus());
      }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  });

  // Stop the camera when leaving camera mode or the page.
  $effect(() => {
    if (mode !== "camera") stopCamera();
    return () => stopCamera();
  });

  async function run(task: () => Promise<Query | null>) {
    if (busy) return;
    busy = true;
    error = "";
    try {
      const query = await task();
      if (query) await onsearch(query);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  function searchText(value = text) {
    const trimmed = value.trim();
    if (!trimmed) return;
    text = trimmed;
    run(async () => ({ kind: "vector", source: "text", text: trimmed, vector: await textEncoder.encode(trimmed) }));
  }

  let lastFrameUrl: string | null = null;

  function searchImage(blob: Blob, source: "photo" | "camera") {
    const url = URL.createObjectURL(blob);
    if (source === "photo") {
      if (preview) URL.revokeObjectURL(preview);
      preview = url;
    } else {
      // Live mode captures a frame every few hundred milliseconds: release the previous one.
      if (lastFrameUrl) URL.revokeObjectURL(lastFrameUrl);
      lastFrameUrl = url;
    }
    run(async () => ({
      kind: "vector",
      source,
      text: source === "photo" ? "your photo" : "camera frame",
      preview: url,
      vector: await imageEncoder.encode(blob),
    }));
  }

  function onFiles(files: FileList | null | undefined) {
    const file = files?.[0];
    if (file && file.type.startsWith("image/")) searchImage(file, "photo");
    else if (file) error = "That file is not an image.";
  }

  function onPaste(event: ClipboardEvent) {
    const item = [...(event.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
    const file = item?.getAsFile();
    if (file) {
      mode = "photo";
      searchImage(file, "photo");
    }
  }

  async function startCamera() {
    error = "";
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" } });
      cameraOn = true;
      queueMicrotask(() => {
        if (video) {
          video.srcObject = stream;
          video.play();
        }
      });
    } catch (e) {
      error = e instanceof DOMException && e.name === "NotAllowedError" ? "Camera permission was denied." : "No camera is available.";
    }
  }

  function stopCamera() {
    live = false;
    liveGeneration++;
    clearTimeout(liveTimer);
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    cameraOn = false;
  }

  function grabFrame(): Promise<Blob | null> {
    if (!video || !video.videoWidth) return Promise.resolve(null);
    const canvas = document.createElement("canvas");
    // A centered square crop: what CLIP sees anyway (it center-crops to 224x224).
    const side = Math.min(video.videoWidth, video.videoHeight);
    canvas.width = canvas.height = 320;
    canvas.getContext("2d")!.drawImage(video, (video.videoWidth - side) / 2, (video.videoHeight - side) / 2, side, side, 0, 0, 320, 320);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  }

  async function capture() {
    const blob = await grabFrame();
    if (blob) searchImage(blob, "camera");
  }

  // Live mode: search the camera feed continuously (as fast as the encoder allows). Each start gets
  // a new generation, so toggling quickly never leaves two loops running.
  let liveGeneration = 0;
  async function liveLoop(generation: number) {
    if (!live || !cameraOn || generation !== liveGeneration) return;
    if (!busy && !document.hidden) await capture();
    if (generation === liveGeneration) liveTimer = window.setTimeout(() => liveLoop(generation), 350);
  }

  function toggleLive() {
    live = !live;
    clearTimeout(liveTimer);
    liveGeneration++;
    if (live) liveLoop(liveGeneration);
  }

  const pct = (v: number) => `${Math.round(v * 100)}%`;

  // Arrow keys move between the tabs (the tablist keyboard pattern).
  const modes: Mode[] = ["text", "photo", "camera", "blend"];
  function onTabKey(event: KeyboardEvent) {
    const delta = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!delta) return;
    event.preventDefault();
    mode = modes[(modes.indexOf(mode) + delta + modes.length) % modes.length];
    queueMicrotask(() => document.getElementById(`search-tab-${mode}`)?.focus());
  }
</script>

<svelte:window onpaste={onPaste} />

<div class="search">
  <div class="tabs" role="tablist" aria-label="Search by">
    {#each [["text", "Describe"], ["photo", "Photo"], ["camera", "Camera"], ["blend", "Blend"]] as [value, label] (value)}
      <button
        role="tab"
        id="search-tab-{value}"
        aria-controls="search-panel"
        aria-selected={mode === value}
        tabindex={mode === value ? 0 : -1}
        class:active={mode === value}
        onclick={() => (mode = value as Mode)}
        onkeydown={onTabKey}
      >
        {label}
      </button>
    {/each}
    <button class="surprise" onclick={onsurprise} {disabled} title="Search with a random photo from the collection" aria-label="Surprise me">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5" /></svg>
      <span>Surprise me</span>
    </button>
  </div>

  <div class="panel" id="search-panel" role="tabpanel" aria-labelledby="search-tab-{mode}">

  {#if mode === "text"}
    <form class="field" onsubmit={(e) => (e.preventDefault(), searchText())}>
      <svg class="lead" viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
      <input
        bind:this={input}
        bind:value={text}
        type="search"
        placeholder={textEncoder.status === "ready"
          ? "Describe a photo…"
          : textEncoder.status === "error"
            ? "The text model couldn't load. Reload to retry."
            : `Loading the text model… ${pct(textEncoder.progress)}`}
        aria-label="Describe a photo to search for"
        disabled={disabled || textEncoder.status !== "ready"}
        autocomplete="off"
        spellcheck="false"
      />
      {#if busy}<span class="spinner" aria-hidden="true"></span>{:else}<kbd aria-hidden="true">/</kbd>{/if}
    </form>
    <div class="chips" aria-label="Example searches">
      {#each examples as example (example)}
        <button onclick={() => searchText(example)} disabled={disabled || textEncoder.status !== "ready"}>{example}</button>
      {/each}
    </div>
  {:else if mode === "photo"}
    <label
      class="drop"
      class:dragging
      ondragover={(e) => (e.preventDefault(), (dragging = true))}
      ondragleave={() => (dragging = false)}
      ondrop={(e) => (e.preventDefault(), (dragging = false), onFiles(e.dataTransfer?.files))}
    >
      <input type="file" accept="image/*" onchange={(e) => onFiles((e.currentTarget as HTMLInputElement).files)} {disabled} />
      {#if preview}
        <img src={preview} alt="Your query" />
      {:else}
        <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="3" /><circle cx="9" cy="10" r="1.8" /><path d="m21 16-5-5-8 8" /></svg>
      {/if}
      <span>
        <strong>Drop a photo</strong>, choose one, or paste from the clipboard.
        {#if imageEncoder.status === "loading"}<br /><em>Loading the CLIP image model… {pct(imageEncoder.progress)}</em>{/if}
        {#if imageEncoder.status === "error"}<br /><em class="bad">The image model couldn't load. Reload to retry.</em>{/if}
      </span>
      {#if busy}<span class="spinner" aria-hidden="true"></span>{/if}
    </label>
  {:else if mode === "blend"}
    <BlendBox
      base={blendBase}
      {onsearch}
      {disabled}
      initialWords={initialBlend?.words ?? ""}
      initialMix={initialBlend?.mix ?? 0.5}
    />
  {:else}
    <div class="camera">
      {#if cameraOn}
        <video bind:this={video} muted playsinline aria-label="Camera preview"></video>
        <div class="camera-actions">
          <button class="btn btn-primary" onclick={capture} disabled={busy || imageEncoder.status !== "ready"}>Capture & search</button>
          <button class="btn btn-ghost" class:on={live} onclick={toggleLive} disabled={imageEncoder.status !== "ready"} aria-pressed={live}>
            <span class="dot" aria-hidden="true"></span>{live ? "Live" : "Go live"}
          </button>
        </div>
      {:else}
        <button class="start" onclick={startCamera}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h3l2-2.5h6L17 7h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z" /><circle cx="12" cy="13" r="3.6" /></svg>
          <span><strong>Point your camera at something</strong><br />Frames are embedded by CLIP on your device; nothing is uploaded.</span>
        </button>
      {/if}
      {#if imageEncoder.status === "loading"}<p class="note">Loading the CLIP image model… {pct(imageEncoder.progress)}</p>{/if}
      {#if imageEncoder.status === "error"}<p class="note bad">The image model couldn't load. Reload to retry.</p>{/if}
    </div>
  {/if}

  </div>

  {#if error}<p class="error" role="alert">{error}</p>{/if}
</div>

<style>
  .search {
    display: grid;
    grid-template-columns: minmax(0, 1fr); /* never grow past the panel for wide content */
    gap: 12px;
    container-type: inline-size;
  }
  .tabs {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
  }
  .tabs button {
    padding: 6px 10px;
    border: 0;
    border-radius: 8px;
    background: transparent;
    color: var(--text-2);
    font-size: 13.5px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
  }
  .tabs button:hover {
    color: var(--text);
  }
  .tabs button.active {
    background: var(--surface-3);
    color: var(--text);
    box-shadow: 0 0 0 1px var(--border-strong);
  }
  .tabs .surprise {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
    white-space: nowrap;
  }
  .panel {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 12px;
  }
  .bad {
    color: var(--warn);
  }
  /* The panel is 400px wide at most: keep the four controls on one line. */
  /* Four tabs and "Surprise me" don't fit the 400px panel: the shuffle button goes icon-only
     (it keeps its accessible name), and the tabs tighten further on the smallest phones. */
  @container (max-width: 400px) {
    .surprise span {
      display: none;
    }
  }
  @container (max-width: 320px) {
    .tabs button {
      padding: 6px 7px;
      font-size: 13px;
    }
  }
  @container (max-width: 275px) {
    .tabs button {
      padding: 6px 5px;
      font-size: 12.5px;
    }
  }
  .surprise svg {
    width: 14px;
    height: 14px;
    fill: none;
    stroke: currentColor;
    stroke-width: 2;
    stroke-linecap: round;
  }
  .field {
    display: flex;
    align-items: center;
    gap: 10px;
    height: 50px;
    padding: 0 14px;
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    background: var(--surface-solid);
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .field:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 4px var(--accent-wash);
  }
  .lead {
    width: 18px;
    height: 18px;
    fill: none;
    stroke: var(--muted);
    stroke-width: 2;
    stroke-linecap: round;
  }
  input[type="search"] {
    flex: 1;
    width: 0; /* the input's default size would otherwise set the field's minimum width */
    min-width: 0;
    border: 0;
    outline: 0;
    background: transparent;
    color: var(--text);
    font: 16px var(--font);
  }
  input::placeholder {
    color: var(--muted);
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
    transition: border-color 0.2s, color 0.2s;
  }
  .chips button:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--text);
  }
  .chips button:disabled {
    opacity: 0.5;
    cursor: progress;
  }
  .drop {
    position: relative;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px;
    border: 1.5px dashed var(--border-strong);
    border-radius: 14px;
    background: var(--surface);
    color: var(--text-2);
    font-size: 13.5px;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
  }
  .drop:hover,
  .drop.dragging {
    border-color: var(--accent);
    background: var(--accent-wash);
  }
  .drop input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }
  .drop img,
  .drop svg {
    flex: none;
    width: 56px;
    height: 56px;
    border-radius: 10px;
    object-fit: cover;
  }
  .drop svg {
    padding: 12px;
    background: var(--surface-2);
    fill: none;
    stroke: var(--text-2);
    stroke-width: 1.6;
  }
  .drop strong,
  .start strong {
    color: var(--text);
  }
  .drop em {
    color: var(--muted);
    font-style: normal;
  }
  .camera {
    display: grid;
    gap: 10px;
  }
  video {
    width: 100%;
    aspect-ratio: 4 / 3;
    border-radius: 14px;
    background: #000;
    object-fit: cover;
  }
  .camera-actions {
    display: flex;
    gap: 8px;
  }
  .camera-actions .btn {
    flex: 1;
    height: 40px;
    font-size: 14px;
  }
  .btn.on {
    border-color: var(--result);
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--muted);
  }
  .btn.on .dot {
    background: var(--result);
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--result) 25%, transparent);
    animation: pulse 1.2s ease-in-out infinite;
  }
  .start {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px;
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    background: var(--surface);
    color: var(--text-2);
    font-size: 13.5px;
    text-align: left;
    cursor: pointer;
  }
  .start:hover {
    border-color: var(--accent);
  }
  .start svg {
    flex: none;
    width: 40px;
    height: 40px;
    fill: none;
    stroke: var(--accent-strong);
    stroke-width: 1.6;
  }
  .note {
    margin: 0;
    color: var(--muted);
    font-size: 12.5px;
  }
  .error {
    margin: 0;
    color: var(--result);
    font-size: 13px;
  }
  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .drop .spinner {
    margin-left: auto;
  }
  @keyframes pulse {
    50% {
      opacity: 0.5;
    }
  }
</style>
