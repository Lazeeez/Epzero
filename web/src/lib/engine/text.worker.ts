// CLIP ViT-B/32 text encoder (int8 ONNX) running in its own worker via ONNX Runtime Web. It maps a
// description into the same 512-d space as the photo embeddings, so a text query is just another
// vector for the HNSW index. The model and runtime are served from this site (see
// scripts/fetch-assets.mjs), so text search works offline after the first visit.
import { AutoTokenizer, CLIPTextModelWithProjection, env } from "@huggingface/transformers";

export type ToText = { type: "encode"; seq: number; text: string };
export type FromText =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; ms: number }
  | { type: "embedding"; seq: number; text: string; vector: Float32Array; ms: number }
  | { type: "error"; seq?: number; message: string };

const post = (message: FromText, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(message, transfer);

const base = new URL(import.meta.env.BASE_URL, self.location.origin).href;
env.allowRemoteModels = false;
env.allowLocalModels = true;
// Root-relative on purpose: transformers.js skips its local-file check for absolute http(s) paths,
// which makes the tokenizer look missing (the model loader is unaffected).
env.localModelPath = `${import.meta.env.BASE_URL}models/`;
const onnx = env.backends.onnx as { wasm?: { wasmPaths?: unknown; numThreads?: number } };
if (onnx.wasm) {
  onnx.wasm.wasmPaths = { mjs: `${base}ort/ort-wasm-simd-threaded.mjs`, wasm: `${base}ort/ort-wasm-simd-threaded.wasm` };
  // A query is at most 77 tokens, so one thread encodes it in milliseconds; extra ORT threads would
  // only compete with the vector index for cores.
  onnx.wasm.numThreads = 1;
}

const MODEL = "clip-vit-base-patch32";
const EXPECTED_BYTES = 66_750_000; // int8 text model (64.5 MB) + tokenizer (2.2 MB)
const files = new Map<string, { loaded: number; total: number }>();
const started = performance.now();

const ready = (async () => {
  const progress_callback = (event: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (event.status !== "progress" || !event.file) return;
    files.set(event.file, { loaded: event.loaded ?? 0, total: event.total ?? 0 });
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    // Servers that compress or omit Content-Length make `total` grow with the download, so
    // measure against the known size of the model files instead.
    post({ type: "progress", loaded, total: Math.max(total, EXPECTED_BYTES) });
  };
  // A 67 MB download can be cut off mid-stream (flaky Wi-Fi, a proxy, the service worker that
  // adds the isolation headers). Retry with backoff; files that already finished come from the
  // browser cache, so a retry only redoes the part that failed.
  for (let attempt = 1; ; attempt++) {
    try {
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL, { progress_callback });
      const model = await CLIPTextModelWithProjection.from_pretrained(MODEL, { dtype: "q8", device: "wasm", progress_callback });
      post({ type: "ready", ms: performance.now() - started });
      return { tokenizer, model };
    } catch (error) {
      if (attempt >= 4) throw error;
      files.clear();
      await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** (attempt - 1)));
    }
  }
})();
ready.catch((error) => post({ type: "error", message: `Text model failed to load: ${error}` }));

self.onmessage = async (event: MessageEvent<ToText>) => {
  const { seq, text } = event.data;
  try {
    const { tokenizer, model } = await ready;
    const start = performance.now();
    const inputs = tokenizer([text], { padding: true, truncation: true });
    const { text_embeds } = await model(inputs);
    const vector = Float32Array.from(text_embeds.data as Float32Array);
    let norm = 0;
    for (const x of vector) norm += x * x;
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < vector.length; i++) vector[i] /= norm;
    post({ type: "embedding", seq, text, vector, ms: performance.now() - start }, [vector.buffer]);
  } catch (error) {
    post({ type: "error", seq, message: String(error) });
  }
};
