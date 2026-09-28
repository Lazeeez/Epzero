// Owns the Oryx engine for the explorer: downloads the int8 CLIP vectors, builds the HNSW graph on
// every core, and answers searches with full traversal traces. Nothing here touches the DOM.
import type { FromWorker, ToWorker, Trace } from "./protocol";

interface Collection {
  load(vectors: Int8Array, scales: Float32Array, dim: number): void;
  beginBuild(M: number, efConstruction: number): void;
  buildStep(count: number, threads: number): number;
  levels(): Uint8Array;
  edges(layer: number): Uint32Array;
  search(query: Float32Array, k: number, ef: number): Trace;
  searchById(id: number, k: number, ef: number): Trace;
  info(): { maxLevel: number; memoryBytes: number };
}

const post = (message: FromWorker, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(message, transfer);

const variant: "mt" | "st" = self.crossOriginIsolated ? "mt" : "st";
const threads = variant === "mt" ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1;
const engine = import(/* @vite-ignore */ `${import.meta.env.BASE_URL}wasm/${variant}/oryx.js`).then(
  (module) => module.default() as Promise<{ Collection: new () => Collection }>,
);
let collection: Collection | null = null;

// Streams a binary file, reporting bytes as they arrive.
async function download(url: string, onBytes: (bytes: number) => void): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("Content-Length")) || 0;
  const buffer = new Uint8Array(total || 1 << 20);
  let length = 0;
  const reader = response.body.getReader();
  let grown = buffer;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (length + value.length > grown.length) {
      const bigger = new Uint8Array(Math.max(grown.length * 2, length + value.length));
      bigger.set(grown.subarray(0, length));
      grown = bigger;
    }
    grown.set(value, length);
    length += value.length;
    onBytes(length);
  }
  return grown.buffer.slice(0, length);
}

async function load(message: Extract<ToWorker, { type: "load" }>) {
  const bytes = message.count * message.dim;
  const [vectors, scales, module] = await Promise.all([
    download(`${message.base}vectors.i8`, (done) => post({ type: "progress", stage: "download", done, total: bytes })),
    download(`${message.base}scales.f32`, () => {}),
    engine,
  ]);
  collection = new module.Collection();
  collection.load(new Int8Array(vectors), new Float32Array(scales), message.dim);

  // Build in slices so the page can animate nodes arriving while every core inserts.
  collection.beginBuild(message.M, message.efConstruction);
  const start = performance.now();
  const slice = Math.max(500, Math.round(message.count / 60));
  for (let built = 0; built < message.count; ) {
    built = collection.buildStep(slice, threads);
    post({ type: "progress", stage: "build", done: built, total: message.count });
  }
  const buildMs = performance.now() - start;
  const info = collection.info();
  const edges: Uint32Array[] = [new Uint32Array(0)];
  for (let layer = 1; layer <= info.maxLevel; layer++) edges.push(collection.edges(layer));
  const levels = collection.levels();
  post(
    {
      type: "built",
      graph: { levels, edges, maxLevel: info.maxLevel, buildMs, memoryBytes: info.memoryBytes, threads, variant },
    },
    [levels.buffer, ...edges.map((e) => e.buffer)],
  );
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  try {
    if (message.type === "load") {
      await load(message);
    } else if (message.type === "search") {
      if (!collection) throw new Error("the index is not built yet");
      const { query } = message;
      const trace =
        query.kind === "id"
          ? collection.searchById(query.id, message.k, message.ef)
          : collection.search(query.vector, message.k, message.ef);
      post({ type: "result", seq: message.seq, query, trace });
    }
  } catch (error) {
    const seq = message.type === "search" ? message.seq : undefined;
    post({ type: "error", seq, message: error instanceof Error ? error.message : String(error) });
  }
};
