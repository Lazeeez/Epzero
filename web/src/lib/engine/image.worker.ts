// CLIP ViT-B/32 image encoder (int8 ONNX) in its own worker: turns a photo or a webcam frame into a
// vector in the same space as the indexed photos, so "find photos like this one" is just a search.
// The model is served from this site (see scripts/fetch-assets.mjs) and cached after first use.
import { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } from "@huggingface/transformers";

export type ToImage = { type: "encode"; seq: number; image: Blob };
export type FromImage =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; ms: number }
  | { type: "embedding"; seq: number; vector: Float32Array; ms: number }
  | { type: "error"; seq?: number; message: string };

const post = (message: FromImage, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(message, transfer);

const base = new URL(import.meta.env.BASE_URL, self.location.origin).href;
env.allowRemoteModels = false;
env.allowLocalModels = true;
// Root-relative on purpose: transformers.js skips its local-file check for absolute http(s) paths.
env.localModelPath = `${import.meta.env.BASE_URL}models/`;
const onnx = env.backends.onnx as { wasm?: { wasmPaths?: unknown; numThreads?: number } };
if (onnx.wasm) {
  onnx.wasm.wasmPaths = { mjs: `${base}ort/ort-wasm-simd-threaded.mjs`, wasm: `${base}ort/ort-wasm-simd-threaded.wasm` };
  // Multithreaded ORT stalled its session setup inside our workers (see text.worker.ts); one
  // thread embeds an image in a few hundred milliseconds, which is fine for interactive use.
  onnx.wasm.numThreads = 1;
}

const MODEL = "clip-vit-base-patch32";
const EXPECTED_BYTES = 89_150_000; // int8 vision model + preprocessor config
const files = new Map<string, number>();
const started = performance.now();

const ready = (async () => {
  // The site's service worker already caches everything under models/. When it's active, don't
  // let transformers.js keep a second copy of the model in its own cache: that doubled the disk
  // writes on a first visit and the reads on every later one. Without it, keep the library cache.
  try {
    env.useBrowserCache = !(await caches.keys()).some((name) => name.startsWith("epzero-shell-"));
  } catch {
    env.useBrowserCache = true;
  }
  const progress_callback = (event: { status: string; file?: string; loaded?: number }) => {
    if (event.status !== "progress" || !event.file) return;
    files.set(event.file, event.loaded ?? 0);
    let loaded = 0;
    for (const bytes of files.values()) loaded += bytes;
    post({ type: "progress", loaded, total: EXPECTED_BYTES });
  };
  for (let attempt = 1; ; attempt++) {
    try {
      const processor = await AutoProcessor.from_pretrained(MODEL, { progress_callback });
      const model = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { dtype: "q8", device: "wasm", progress_callback });
      post({ type: "ready", ms: performance.now() - started });
      return { processor, model };
    } catch (error) {
      if (attempt >= 4) throw error;
      files.clear();
      await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** (attempt - 1)));
    }
  }
})();
ready.catch((error) => post({ type: "error", message: `Image model failed to load: ${error}` }));

self.onmessage = async (event: MessageEvent<ToImage>) => {
  const { seq, image } = event.data;
  try {
    const { processor, model } = await ready;
    const start = performance.now();
    const raw = await RawImage.fromBlob(image);
    const inputs = await processor(raw);
    const { image_embeds } = await model(inputs);
    const vector = Float32Array.from(image_embeds.data as Float32Array);
    let norm = 0;
    for (const x of vector) norm += x * x;
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < vector.length; i++) vector[i] /= norm;
    post({ type: "embedding", seq, vector, ms: performance.now() - start }, [vector.buffer]);
  } catch (error) {
    post({ type: "error", seq, message: String(error) });
  }
};
