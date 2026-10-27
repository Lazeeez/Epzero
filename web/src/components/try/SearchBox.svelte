<script lang="ts">
  import { untrack } from "svelte";
  import { imageEncoder, textEncoder } from "../../lib/engine/encoders.svelte";
  import { engine } from "../../lib/engine/engine.svelte";
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
  let stream = $state.raw<MediaStream | null>(null);
  let cameraOn = $state(false);
  let hasFrame = $state(false); // the preview is showing frames: until then there is nothing to capture
  let starting = $state(false);
  let live = $state(false);
  let liveTimer = 0;
  // Phones start on the back camera (point it at things); laptops have one, facing the user.
  let facing = $state<"user" | "environment">(matchMedia("(pointer: coarse)").matches ? "environment" : "user");
  let cameraCount = $state(0);
  // A camera facing the user is shown mirrored, like every video-call app; captures are not.
  let mirrored = $state(false);

  // Recent descriptions, kept on this device only (a convenience: fine if storage is blocked).
  const RECENT_KEY = "epzero-recent-searches";
  let recent = $state<string[]>(loadRecent());
  function loadRecent(): string[] {
    try {
      const value = JSON.parse(localStorage.getItem(RECENT_KEY) ?? "[]");
      return Array.isArray(value) ? value.filter((v) => typeof v === "string").slice(0, 6) : [];
    } catch {
      return [];
    }
  }
  function saveRecent(next: string[]) {
    recent = next;
    try {
      localStorage.setItem(RECENT_KEY, JSON.stringify(next));
    } catch {
      /* storage blocked */
    }
  }

  const examples = ["a dog on the beach", "snowy mountains", "city lights at night", "a bowl of fruit", "vintage black and white photo", "a red sports car"];

  // Models load once the index is ready: searching needs the index anyway, and loading a 67-89 MB
  // model at the same time would slow the index's own download on the critical path.
  $effect(() => {
    if (engine.status !== "ready") return;
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

  // Stop the camera when leaving camera mode or the page. Depends on `mode` only: stopCamera reads
  // the stream, and tracking that would stop every new stream the moment it started.
  $effect(() => {
    if (mode !== "camera") untrack(stopCamera);
    return () => untrack(stopCamera);
  });

  async function run(task: () => Promise<Query | null>, live = false) {
    if (busy) return;
    busy = true;
    error = "";
    try {
      const query = await task();
      if (query) await onsearch(query, { live });
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  // Re-runs the current description, if there is one, so the switch shows its effect at once.
  function toggleHybrid(on: boolean) {
    engine.setHybrid(on);
    if (text.trim() && textEncoder.status === "ready" && !busy) searchText();
  }

  function searchText(value = text) {
    const trimmed = value.trim();
    if (!trimmed) return;
    text = trimmed;
    saveRecent([trimmed, ...recent.filter((r) => r !== trimmed)].slice(0, 6));
    run(async () => ({ kind: "vector", source: "text", text: trimmed, vector: await textEncoder.encode(trimmed) }));
  }

  // Camera frames: live mode makes one every few hundred milliseconds. Release old ones, but only
  // once they can no longer be on screen (the results header shows the latest query's frame).
  const frameUrls: string[] = [];

  function searchImage(blob: Blob, source: "photo" | "camera", liveFrame = false) {
    const url = URL.createObjectURL(blob);
    if (source === "photo") {
      if (preview) URL.revokeObjectURL(preview);
      preview = url;
    } else {
      frameUrls.push(url);
      while (frameUrls.length > 3) URL.revokeObjectURL(frameUrls.shift()!);
    }
    run(
      async () => ({
        kind: "vector",
        source,
        text: source === "photo" ? "your photo" : "camera frame",
        preview: url,
        vector: await imageEncoder.encode(blob),
      }),
      liveFrame,
    );
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

  function cameraError(e: unknown): string {
    const name = e instanceof DOMException || e instanceof Error ? e.name : "";
    switch (name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Camera access was blocked. Allow the camera for this site in your browser's address bar, then try again.";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No camera was found on this device.";
      case "NotReadableError":
      case "AbortError":
        return "The camera is busy in another app or tab. Close it there and try again.";
      case "NotSupportedError":
        return "This browser window can't open the camera (in-app browsers often can't). Try it in Chrome, Safari, Edge or Firefox.";
      default:
        return `The camera couldn't start${e instanceof Error && e.message ? `: ${e.message}` : "."}`;
    }
  }

  async function startCamera(want: "user" | "environment" = facing) {
    error = "";
    if (!navigator.mediaDevices?.getUserMedia) {
      error = "This browser can't use a camera on this page.";
      return;
    }
    starting = true;
    releaseStream();
    try {
      const next = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: want }, width: { ideal: 640 }, height: { ideal: 480 } },
      });
      if (mode !== "camera") {
        next.getTracks().forEach((t) => t.stop()); // the viewer left the tab while it was starting
        return;
      }
      facing = want;
      const track = next.getVideoTracks()[0];
      // Laptops don't report a facing mode; their one camera faces the user.
      mirrored = track?.getSettings().facingMode !== "environment";
      track?.addEventListener("ended", () => {
        if (stream === next) {
          stopCamera();
          error = "The camera stopped (it was unplugged or its permission was revoked).";
        }
      });
      stream = next;
      cameraOn = true;
      // Device labels and counts are only available once permission is granted.
      cameraCount = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput").length;
    } catch (e) {
      error = cameraError(e);
    } finally {
      starting = false;
    }
  }

  // Attach the stream once both it and the <video> element exist (the element only renders after
  // cameraOn flips, so doing it right after getUserMedia could race the render).
  $effect(() => {
    if (!video || !stream) return;
    video.srcObject = stream;
    video.play().catch(() => {
      /* autoplay of a muted inline video is allowed; a rejection means the element went away */
    });
  });

  function releaseStream() {
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    hasFrame = false;
  }

  function stopCamera() {
    live = false;
    liveGeneration++;
    clearTimeout(liveTimer);
    releaseStream();
    cameraOn = false;
  }

  function grabFrame(): Promise<Blob | null> {
    if (!video || !video.videoWidth) {
      error = "The camera is still starting. Try again in a moment.";
      return Promise.resolve(null);
    }
    const canvas = document.createElement("canvas");
    // A centered square crop: what CLIP sees anyway (it center-crops to 224x224).
    const side = Math.min(video.videoWidth, video.videoHeight);
    canvas.width = canvas.height = 320;
    canvas.getContext("2d")!.drawImage(video, (video.videoWidth - side) / 2, (video.videoHeight - side) / 2, side, side, 0, 0, 320, 320);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  }

  async function capture(liveFrame = false) {
    const blob = await grabFrame();
    if (blob) searchImage(blob, "camera", liveFrame);
  }

  // Live mode: search the camera feed continuously (as fast as the encoder allows). Each start gets
  // a new generation, so toggling quickly never leaves two loops running.
  let liveGeneration = 0;
  async function liveLoop(generation: number) {
    if (!live || !cameraOn || generation !== liveGeneration) return;
    // Live frames keep the 3D camera still and replay fast (no fly-to on every frame).
    if (!busy && !document.hidden && video?.videoWidth) await capture(true);
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
  <!-- The tab list holds only the tabs (ARIA requires it); the shuffle button sits beside it. -->
  <div class="tabs">
    <div class="tablist" role="tablist" aria-label="Search by">
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
    </div>
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
    {#if recent.length}
      <div class="chips recent" role="group" aria-label="Recent searches">
        <span class="label">Recent</span>
        {#each recent as item (item)}
          <button onclick={() => searchText(item)} disabled={disabled || textEncoder.status !== "ready"}>{item}</button>
        {/each}
        <button class="clear" onclick={() => saveRecent([])} aria-label="Clear recent searches" title="Clear recent searches">×</button>
      </div>
    {/if}
    <div class="chips" role="group" aria-label="Example searches">
      {#each examples as example (example)}
        <button onclick={() => searchText(example)} disabled={disabled || textEncoder.status !== "ready"}>{example}</button>
      {/each}
    </div>
    {#if engine.source !== "mine"}
    <label class="hybrid" title="Hybrid search: the words are also matched against the photos' captions (BM25), and the two rankings are merged">
      <input type="checkbox" role="switch" checked={engine.hybrid} onchange={(e) => toggleHybrid(e.currentTarget.checked)} />
      <span>Match words in captions too <small>hybrid search</small></span>
    </label>
    {/if}
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
        <div class="viewfinder">
          <video bind:this={video} class:mirrored autoplay muted playsinline aria-label="Camera preview" onloadeddata={() => (hasFrame = true)}></video>
          {#if live}<span class="live-badge" aria-hidden="true">LIVE</span>{/if}
          <div class="corner-actions">
            {#if cameraCount > 1}
              <button
                class="round"
                onclick={() => startCamera(facing === "user" ? "environment" : "user")}
                disabled={starting}
                aria-label="Switch camera"
                title="Switch camera"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8h3l2-2.5h6L17 8h3v11H4zM9 13a3 3 0 0 1 5.2-2M15 13a3 3 0 0 1-5.2 2M14 9.5l.4 1.6-1.6.3M10 16.5l-.4-1.6 1.6-.3" /></svg>
              </button>
            {/if}
            <button class="round" onclick={stopCamera} aria-label="Turn the camera off" title="Turn the camera off">
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" /></svg>
            </button>
          </div>
        </div>
        <div class="camera-actions">
          <button class="btn btn-primary" onclick={() => capture()} disabled={busy || imageEncoder.status !== "ready" || !hasFrame}>
            {imageEncoder.status !== "ready" ? `Loading model… ${pct(imageEncoder.progress)}` : hasFrame ? "Capture & search" : "Starting camera…"}
          </button>
          <button class="btn btn-ghost" class:on={live} onclick={toggleLive} disabled={imageEncoder.status !== "ready" || !hasFrame} aria-pressed={live}>
            <span class="dot" aria-hidden="true"></span>{live ? "Live" : "Go live"}
          </button>
        </div>
      {:else}
        <button class="start" onclick={() => startCamera()} disabled={starting}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h3l2-2.5h6L17 7h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z" /><circle cx="12" cy="13" r="3.6" /></svg>
          <span>
            <strong>{starting ? "Starting the camera…" : "Point your camera at something"}</strong><br />Frames are embedded by CLIP on
            your device; nothing is uploaded.
          </span>
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
  .tablist {
    display: flex;
    align-items: center;
    gap: 4px;
    min-width: 0;
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
  .hybrid {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    width: fit-content;
    color: var(--text-2);
    font-size: 13px;
    cursor: pointer;
  }
  .hybrid input {
    width: 16px;
    height: 16px;
    margin: 0;
    accent-color: var(--accent);
  }
  .hybrid small {
    margin-left: 4px;
    color: var(--muted);
    font-size: 11.5px;
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
  .viewfinder {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    background: #000;
  }
  video {
    display: block;
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
  }
  video.mirrored {
    transform: scaleX(-1);
  }
  .live-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    padding: 2px 8px;
    border-radius: 6px;
    background: var(--result);
    color: #fff;
    font: 700 10.5px var(--mono);
    letter-spacing: 0.08em;
  }
  .corner-actions {
    position: absolute;
    top: 8px;
    right: 8px;
    display: flex;
    gap: 6px;
  }
  .round {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border: 0;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.55);
    color: #fff;
    cursor: pointer;
  }
  .round svg {
    width: 17px;
    height: 17px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
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
  .recent {
    align-items: center;
  }
  .recent .label {
    color: var(--muted);
    font: 600 10.5px var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .recent button {
    border-style: dashed;
  }
  .recent .clear {
    padding: 2px 9px;
    border-style: solid;
    color: var(--muted);
  }
  /* Phones: each chip group is one row that scrolls sideways (bleeding to the sheet's edges),
     instead of wrapping into three or four lines that push the results down; bigger taps. */
  @media (max-width: 900px) {
    .chips {
      flex-wrap: nowrap;
      overflow-x: auto;
      margin-inline: -16px;
      padding-inline: 16px;
      scrollbar-width: none;
    }
    .chips::-webkit-scrollbar {
      display: none;
    }
    .chips > :global(*) {
      flex: none;
    }
    .chips button {
      padding: 8px 13px;
      font-size: 13.5px;
    }
  }
</style>
