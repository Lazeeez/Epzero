// CLIP ViT-B/32 image encoder (int8 ONNX) in its own worker: turns a photo or a webcam frame into a
// vector in the same space as the indexed photos, so "find photos like this one" is just a search.
// It also embeds the visitor's own photos in bulk ("embed"), several workers side by side.
// The model is served from this site (see scripts/fetch-assets.mjs) and cached after first use.
import { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } from "@huggingface/transformers";

export type ToImage =
  | { type: "encode"; seq: number; image: Blob }
  /** A batch of the visitor's photos: each is embedded and given a thumbnail (`thumb` px, longest side). */
  | { type: "embed"; seq: number; files: Blob[]; thumb: number };
export interface Embedded {
  /** null: the browser couldn't decode this file (e.g. HEIC outside Safari). */
  vector: Float32Array | null;
  thumb: Blob | null;
  width: number;
  height: number;
}
export type FromImage =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; ms: number }
  | { type: "embedding"; seq: number; vector: Float32Array; ms: number }
  | { type: "embedded"; seq: number; items: Embedded[]; ms: number }
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
  // thread embeds an image in a few hundred milliseconds, which is fine for interactive use. Bulk
  // embedding runs several of these workers instead.
  onnx.wasm.numThreads = 1;
}

const MODEL = "clip-vit-base-patch32";
const EXPECTED_BYTES = 89_150_000; // int8 vision model + preprocessor config
const CLIP_SIDE = 224; // the model's input: shortest side 224, centre-cropped by the processor
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

function normalized(data: Float32Array, offset: number, dim: number): Float32Array {
  const vector = data.slice(offset, offset + dim);
  let norm = 0;
  for (const x of vector) norm += x * x;
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) vector[i] /= norm;
  return vector;
}

// Draws `source` scaled by `scale` onto a new canvas. Big reductions go in halving steps: one
// drawImage from 12 megapixels straight to a thumbnail aliases visibly.
function scaled(source: ImageBitmap | OffscreenCanvas, width: number, height: number, scale: number): OffscreenCanvas {
  let current: ImageBitmap | OffscreenCanvas = source;
  let w = width;
  let h = height;
  const targetW = Math.max(1, Math.round(width * scale));
  const targetH = Math.max(1, Math.round(height * scale));
  do {
    const nextW = Math.max(targetW, Math.round(w / 2));
    const nextH = Math.max(targetH, Math.round(h / 2));
    const canvas = new OffscreenCanvas(nextW, nextH);
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(current, 0, 0, nextW, nextH);
    current = canvas;
    w = nextW;
    h = nextH;
  } while (w > targetW || h > targetH);
  return current as OffscreenCanvas;
}

// One decode per photo (with its EXIF orientation applied): a thumbnail for the page, and the
// model's input scaled from that thumbnail.
async function prepare(file: Blob, thumbSide: number) {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  try {
    const { width, height } = bitmap;
    const thumbScale = Math.min(1, thumbSide / Math.max(width, height));
    const thumbCanvas = scaled(bitmap, width, height, thumbScale);
    const clipScale = CLIP_SIDE / Math.min(thumbCanvas.width, thumbCanvas.height);
    const input = clipScale < 1 ? scaled(thumbCanvas, thumbCanvas.width, thumbCanvas.height, clipScale) : thumbCanvas;
    const pixels = input.getContext("2d")!.getImageData(0, 0, input.width, input.height);
    const raw = new RawImage(pixels.data, input.width, input.height, 4);
    const thumb = await thumbCanvas.convertToBlob({ type: "image/jpeg", quality: 0.82 });
    return { raw, thumb, width, height };
  } finally {
    bitmap.close();
  }
}

self.onmessage = async (event: MessageEvent<ToImage>) => {
  const message = event.data;
  const { seq } = message;
  try {
    const { processor, model } = await ready;
    const start = performance.now();
    if (message.type === "encode") {
      const raw = await RawImage.fromBlob(message.image);
      const inputs = await processor(raw);
      const { image_embeds } = await model(inputs);
      const vector = normalized(image_embeds.data as Float32Array, 0, image_embeds.dims[1]);
      post({ type: "embedding", seq, vector, ms: performance.now() - start }, [vector.buffer]);
      return;
    }
    // Bulk: decode every file (skipping those the browser can't read), then one batched run.
    const prepared = await Promise.all(message.files.map((file) => prepare(file, message.thumb).catch(() => null)));
    const ok = prepared.filter((p) => p !== null);
    const items: Embedded[] = prepared.map((p) => ({ vector: null, thumb: p?.thumb ?? null, width: p?.width ?? 0, height: p?.height ?? 0 }));
    if (ok.length) {
      const inputs = await processor(ok.map((p) => p!.raw));
      const { image_embeds } = await model(inputs);
      const dim = image_embeds.dims[1] as number;
      const data = image_embeds.data as Float32Array;
      let row = 0;
      prepared.forEach((p, i) => {
        if (p) items[i].vector = normalized(data, row++ * dim, dim);
      });
    }
    post(
      { type: "embedded", seq, items, ms: performance.now() - start },
      items.flatMap((item) => (item.vector ? [item.vector.buffer] : [])),
    );
  } catch (error) {
    post({ type: "error", seq, message: String(error) });
  }
};
