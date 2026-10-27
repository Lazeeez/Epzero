// Model files for the CLIP workers, each downloaded once, all at the same time.
//
// transformers.js fetches its files one after another, and with a progress callback it first
// checks each file's size with a plain GET whose body it never reads. For files served by this
// site that meant every model downloaded twice at once, each copy at half the bandwidth. Here the
// worker names the files it needs, they download in parallel (with the ONNX runtime), and
// transformers.js reads them from memory through its custom-cache hook.
import { env } from "@huggingface/transformers";

const base = new URL(import.meta.env.BASE_URL, self.location.origin).href;
// Without the site's service worker (it caches models/ itself), keep the files in this cache.
const CACHE = "epzero-models-v1";
// Sizes size the buffers and the progress bar; if a file differs, the buffer grows or is trimmed.
const RUNTIME = { "ort/ort-wasm-simd-threaded.wasm": 14_264_838, "ort/ort-wasm-simd-threaded.mjs": 24_381 };

export interface ModelFiles {
  /** Starts every download; resolves when all of them have arrived. */
  ready: Promise<void>;
}

/**
 * Downloads `files` (paths under models/, with their sizes in bytes) and the ONNX runtime, and points
 * transformers.js at them. `onprogress` gets the bytes received so far and the total.
 */
export function loadModelFiles(files: Record<string, number>, onprogress: (loaded: number, total: number) => void): ModelFiles {
  const total = [...Object.values(files), ...Object.values(RUNTIME)].reduce((sum, n) => sum + n, 0);
  const received = new Map<string, number>();
  const report = (file: string, bytes: number) => {
    received.set(file, bytes);
    let loaded = 0;
    for (const n of received.values()) loaded += n;
    onprogress(loaded, Math.max(loaded, total));
  };
  const persist = ownCache();
  // The runtime (its 14 MB wasm and the code that loads it) arrives alongside the model instead of
  // after it: transformers.js asks the same cache for it.
  const all: Record<string, number> = { ...RUNTIME };
  for (const [file, size] of Object.entries(files)) all[`models/${file}`] = size;
  const downloads = new Map(
    Object.entries(all).map(([path, size]) => [path, download(`${base}${path}`, size, (n) => report(path, n), persist)]),
  );
  env.useBrowserCache = false;
  env.useCustomCache = true;
  env.customCache = {
    async match(key: string) {
      const file = [...downloads.keys()].find((path) => key.endsWith(`/${path}`));
      if (!file) return undefined;
      const bytes = await downloads.get(file)!;
      return new Response(bytes, { headers: { "Content-Length": String(bytes.byteLength) } });
    },
    async put() {},
  };
  return { ready: Promise.all(downloads.values()).then(() => undefined) };
}

// The service worker caches everything under models/; when it's not running (first visit before it
// takes control, or a browser without one), keep a copy here so the next visit is offline-fast.
async function ownCache(): Promise<Cache | null> {
  // transformers.js's own copies from earlier visits (up to 154 MB) are no longer read.
  caches.delete("transformers-cache").catch(() => {});
  try {
    const names = await caches.keys();
    if (names.some((name) => name.startsWith("epzero-shell-"))) return null;
    return await caches.open(CACHE);
  } catch {
    return null;
  }
}

async function download(url: string, size: number, onbytes: (n: number) => void, persist: Promise<Cache | null>): Promise<Uint8Array<ArrayBuffer>> {
  const cache = await persist;
  const hit = await cache?.match(url).catch(() => undefined);
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    onbytes(bytes.byteLength);
    return bytes;
  }
  // A 65 MB download can be cut off mid-stream (flaky Wi-Fi, a proxy): retry with backoff.
  for (let attempt = 1; ; attempt++) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText} for ${url}`);
      const bytes = await read(response, size, onbytes);
      if (cache) cache.put(url, new Response(bytes)).catch(() => {});
      return bytes;
    } catch (error) {
      onbytes(0);
      if (attempt >= 4) throw error;
      await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** (attempt - 1)));
    }
  }
}

// Reads the body into one buffer of the expected size, reporting progress. Content-Length can't size
// it: a compressed response gives the compressed size, while the body arrives decompressed.
async function read(response: Response, size: number, onbytes: (n: number) => void): Promise<Uint8Array<ArrayBuffer>> {
  let buffer: Uint8Array<ArrayBuffer> = new Uint8Array(size);
  let length = 0;
  const reader = response.body!.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (length + value.byteLength > buffer.byteLength) {
      const grown = new Uint8Array(Math.max(Math.ceil(buffer.byteLength * 1.25), length + value.byteLength));
      grown.set(buffer.subarray(0, length));
      buffer = grown;
    }
    buffer.set(value, length);
    length += value.byteLength;
    onbytes(length);
  }
  return length === buffer.byteLength ? buffer : buffer.slice(0, length);
}
