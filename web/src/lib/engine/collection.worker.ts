// Owns the Oryx engine for the explorer: downloads the int8 CLIP vectors, builds the HNSW graph on
// every core, and answers searches with full traversal traces. Nothing here touches the DOM.
import type { Evaluation, FromWorker, Neighborhood, Request, ToWorker, Trace } from "./protocol";

interface Collection {
  load(vectors: Int8Array, scales: Float32Array, dim: number): void;
  beginBuild(M: number, efConstruction: number): void;
  buildStep(count: number, threads: number): number;
  levels(): Uint8Array;
  edges(layer: number): Uint32Array;
  search(query: Float32Array, k: number, ef: number): Trace;
  searchById(id: number, k: number, ef: number): Trace;
  evaluate(samples: number, k: number, ef: number, threads: number): Omit<Evaluation, "ef">;
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
// The int8 codes and per-dimension scales, kept to answer vector and similarity requests in JS.
let codes = new Int8Array(0);
let scales = new Float32Array(0);
let dim = 0;
let count = 0;
// Base-layer adjacency (CSR), built on first use and dropped on rebuild.
let adjacency: { offsets: Uint32Array; targets: Uint32Array } | null = null;

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
  const [vectorBytes, scaleBytes, module] = await Promise.all([
    download(`${message.base}vectors.i8`, (done) => post({ type: "progress", stage: "download", done, total: bytes })),
    download(`${message.base}scales.f32`, () => {}),
    engine,
  ]);
  codes = new Int8Array(vectorBytes);
  scales = new Float32Array(scaleBytes);
  dim = message.dim;
  count = codes.length / dim;
  collection = new module.Collection();
  collection.load(codes, scales, dim);
  build(message.M, message.efConstruction);
}

// Builds the graph in slices so the page can animate nodes arriving while every core inserts.
function build(M: number, efConstruction: number) {
  const index = collection!;
  adjacency = null;
  index.beginBuild(M, efConstruction);
  const start = performance.now();
  const slice = Math.max(500, Math.round(count / 60));
  for (let built = 0; built < count; ) {
    built = index.buildStep(slice, threads);
    post({ type: "progress", stage: "build", done: built, total: count });
  }
  const buildMs = performance.now() - start;
  const info = index.info();
  const edges: Uint32Array[] = [new Uint32Array(0)];
  for (let layer = 1; layer <= info.maxLevel; layer++) edges.push(index.edges(layer));
  const levels = index.levels();
  post(
    {
      type: "built",
      graph: { levels, edges, maxLevel: info.maxLevel, buildMs, memoryBytes: info.memoryBytes, threads, variant, M, efConstruction },
    },
    [levels.buffer, ...edges.map((e) => e.buffer)],
  );
}

// A photo's vector, dequantized exactly as the engine does it (per-dimension scale, normalized).
function vectorOf(id: number): Float32Array {
  if (!(id >= 0 && id < count)) throw new Error(`no photo #${id}`);
  const out = new Float32Array(dim);
  let norm = 0;
  for (let d = 0; d < dim; d++) {
    const x = (codes[id * dim + d] * scales[d]) / 127;
    out[d] = x;
    norm += x * x;
  }
  norm = Math.sqrt(norm) || 1;
  for (let d = 0; d < dim; d++) out[d] /= norm;
  return out;
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let d = 0; d < a.length; d++) sum += a[d] * b[d];
  return sum;
}

function neighborsOf(id: number): Uint32Array {
  if (!adjacency) {
    const edges = collection!.edges(0);
    const offsets = new Uint32Array(count + 1);
    for (let e = 0; e < edges.length; e++) offsets[edges[e] + 1]++;
    for (let i = 0; i < count; i++) offsets[i + 1] += offsets[i];
    const targets = new Uint32Array(edges.length);
    const fill = offsets.slice(0, count);
    for (let e = 0; e < edges.length; e += 2) {
      targets[fill[edges[e]]++] = edges[e + 1];
      targets[fill[edges[e + 1]]++] = edges[e];
    }
    adjacency = { offsets, targets };
  }
  return adjacency.targets.slice(adjacency.offsets[id], adjacency.offsets[id + 1]);
}

function answer(request: Request): { value: unknown; transfer: Transferable[] } {
  if (!collection) throw new Error("the index is not built yet");
  switch (request.method) {
    case "vector": {
      const vector = vectorOf(request.id);
      return { value: vector, transfer: [vector.buffer] };
    }
    case "neighbors": {
      const ids = neighborsOf(request.id);
      const target = vectorOf(request.target);
      const similarities = Float32Array.from(ids, (n) => dot(vectorOf(n), target));
      const value: Neighborhood = { ids, similarities };
      return { value, transfer: [ids.buffer, similarities.buffer] };
    }
    case "similarity":
      return { value: dot(vectorOf(request.a), vectorOf(request.b)), transfer: [] };
    case "evaluate": {
      const value: Evaluation = { ...collection.evaluate(request.samples, 10, request.ef, threads), ef: request.ef };
      return { value, transfer: [] };
    }
  }
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  try {
    if (message.type === "load") {
      await load(message);
    } else if (message.type === "rebuild") {
      if (!collection) throw new Error("the index is not built yet");
      build(message.M, message.efConstruction);
    } else if (message.type === "request") {
      const { value, transfer } = answer(message.request);
      post({ type: "reply", seq: message.seq, value }, transfer);
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
    const seq = message.type === "search" || message.type === "request" ? message.seq : undefined;
    post({ type: "error", seq, message: error instanceof Error ? error.message : String(error) });
  }
};
