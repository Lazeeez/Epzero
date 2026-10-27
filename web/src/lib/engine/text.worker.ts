// CLIP ViT-B/32 text encoder (int8 ONNX) running in its own worker via ONNX Runtime Web. It maps a
// description into the same 512-d space as the photo embeddings, so a text query is just another
// vector for the HNSW index. The model and runtime are served from this site (see
// scripts/fetch-assets.mjs) and downloaded by model-files.ts, so text search works offline after the
// first visit.
import { AutoTokenizer, CLIPTextModelWithProjection, env } from "@huggingface/transformers";
import { loadModelFiles } from "./model-files";

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
const started = performance.now();

const ready = (async () => {
  const files = loadModelFiles(
    {
      [`${MODEL}/tokenizer.json`]: 2_224_119,
      [`${MODEL}/tokenizer_config.json`]: 775,
      [`${MODEL}/config.json`]: 4_524,
      [`${MODEL}/onnx/text_model_quantized.onnx`]: 64_504_507,
    },
    (loaded, total) => post({ type: "progress", loaded, total }),
  );
  await files.ready;
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(MODEL),
    CLIPTextModelWithProjection.from_pretrained(MODEL, { dtype: "q8", device: "wasm" }),
  ]);
  post({ type: "ready", ms: performance.now() - started });
  return { tokenizer, model };
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
